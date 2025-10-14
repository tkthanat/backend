# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException, Response, \
    Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import os, io, csv, asyncio, base64

from .db_models import get_db, UserFace, User
from .camera_handler import CameraManager, discover_local_devices
from .ai_engine import refresh_facebank_from_db, load_facebank

app = FastAPI(title="Offline Attendance (Minimal)")

# ====== กล้อง: ตั้งค่า source (เว็บแคมโน้ตบุ๊ก test ก่อน) ======
CAMERA_SOURCES = {"entrance": "0", "exit": "1"}  # ถ้ามีตัวเดียว ให้ตั้งเป็น "0" ทั้งคู่ได้
cam_mgr = CameraManager(CAMERA_SOURCES, fps=10, width=640, height=480)

# ====== โหลด facebank ตอนบูต ======
@app.on_event("startup")
async def _startup():
    cnt = load_facebank()
    print(f"[facebank] loaded users={cnt}")

# ---------- กลุ่มอัปโหลดรูปฝึก / รีเฟรชโมเดล ----------
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "./data/faces/train")
os.makedirs(MEDIA_ROOT, exist_ok=True)

@app.post("/faces/upload")
async def upload_faces(user_id: int = Form(...), images: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    อัปโหลดรูปหลายไฟล์ => เซฟที่ MEDIA_ROOT/{user_id}/filename และบันทึก path ลง DB (UserFace)
    """
    saved, items = 0, []
    user_dir = os.path.join(MEDIA_ROOT, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    for f in images:
        # กันชื่อไฟล์แปลกๆ นิดหน่อย
        name = os.path.basename(f.filename)
        dest = os.path.join(user_dir, name)
        content = await f.read()
        with open(dest, "wb") as wf:
            wf.write(content)
        # บันทึก DB: เก็บเป็นชื่อไฟล์พอ (train จะ map กลับเอง)
        uf = UserFace(user_id=user_id, file_path=name)
        db.add(uf)
        items.append({"file": name})
        saved += 1
    db.commit()
    return {"saved": saved, "items": items}

@app.post("/train/refresh")
def train_refresh(db: Session = Depends(get_db)):
    """
    สร้าง/อัปเดต facebank จากรายการรูปทั้งหมดใน user_faces
    """
    rows = (
        db.query(UserFace.user_id, UserFace.file_path, User.name)
        .join(User, User.user_id == UserFace.user_id)
        .all()
    )
    users, total = refresh_facebank_from_db(rows)
    # โหลดเข้า RAM ทับอีกครั้ง
    cnt = load_facebank()
    return {"message": "facebank updated", "users": users, "images_used": total, "loaded": cnt}

# ---------- กลุ่มกล้อง (มี overlay) ----------
@app.get("/cameras")
def list_cameras():
    return {"cams": cam_mgr.list()}

@app.post("/cameras/{cam_id}/open")
def open_camera(cam_id: str):
    cam_mgr.open(cam_id)
    return {"message": f"camera '{cam_id}' opened"}

@app.post("/cameras/{cam_id}/close")
def close_camera(cam_id: str):
    cam_mgr.close(cam_id)
    return {"message": f"camera '{cam_id}' closed"}

@app.get("/cameras/{cam_id}/snapshot", responses={200: {"content": {"image/jpeg": {}}}})
def camera_snapshot(cam_id: str):
    jpg = cam_mgr.get_jpeg(cam_id)
    return Response(content=jpg, media_type="image/jpeg")

@app.get("/cameras/{cam_id}/mjpeg")
def camera_mjpeg(cam_id: str):
    boundary = "frame"

    async def gen():
        while True:
            try:
                jpg = cam_mgr.get_jpeg(cam_id)
                yield (
                        b"--" + boundary.encode() + b"\r\n"
                                                    b"Content-Type: image/jpeg\r\n"
                                                    b"Cache-Control: no-cache\r\n"
                                                    b"Pragma: no-cache\r\n"
                                                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                        + jpg + b"\r\n"
                )
            except Exception as e:
                # ถ้า error ให้พักสั้นๆ
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.06)  # ~16fps
    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                             headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Connection": "keep-alive"})

# (ทางเลือก) ส่งภาพเป็น base64 ผ่าน WS
@app.websocket("/ws/cameras/{cam_id}")
async def ws_camera(ws: WebSocket, cam_id: str):
    await ws.accept()
    try:
        cam_mgr.open(cam_id)
    except Exception:
        pass
    try:
        while True:
            await asyncio.sleep(0.1)  # ~10fps
            try:
                jpg = cam_mgr.get_jpeg(cam_id)
                b64 = base64.b64encode(jpg).decode("ascii")
                await ws.send_json({"type": "frame", "cam_id": cam_id, "data": b64})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass


# ======= NEW: ค้นหากล้องที่ต่ออยู่ =======
@app.get("/cameras/discover")
def cameras_discover(max_index: int = 10, test_frame: bool = True):
    """
    สำรวจ device ที่เปิดได้/อ่านได้
    - Windows/macOS: ลอง index 0..max_index
    - Linux: ไล่ /dev/video*
    """
    devs = discover_local_devices(max_index=max_index, test_frame=test_frame)
    return {"devices": devs}

# ======= NEW: อ่าน/ตั้งค่า mapping กล้อง =======
@app.get("/cameras/config")
def get_camera_config():
    return {"mapping": {k: v.src for k, v in cam_mgr.sources.items()}}

@app.post("/cameras/config")
def set_camera_config(mapping: dict = Body(..., example={"entrance": "0", "exit": "1"})):
    """
    ตั้งค่ากล้องใหม่ เช่น
    {
      "entrance": "2",   # ต่อกล้อง USB ตัวใหม่เป็น entrance
      "exit": "0"
    }
    """
    # ปิดกล้องเก่าก่อน + เซ็ต source ใหม่
    cam_mgr.reconfigure(mapping)
    # (ออปชัน) เปิดทันทีเพื่อเทสต์
    for cam_id in mapping.keys():
        try:
            cam_mgr.open(cam_id)
        except Exception as e:
            # ไม่เป็นไร ถ้าเปิดไม่ได้ก็ให้ไปเปิดเองผ่าน /open
            pass
    return {"message": "camera mapping updated", "mapping": mapping}

# ---------- 👤 สร้างผู้ใช้ใหม่ (Register) ----------
class UserCreate(BaseModel):
    student_code: Optional[str] = None
    name: str
    role: str                    # admin / operator / viewer
    user_type_id: Optional[int] = None
    subject_id: Optional[int] = None
    password_hash: Optional[str] = None   # optional

@app.post("/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    # ตรวจ role ว่าถูกต้องไหม
    if payload.role not in ["admin", "operator", "viewer"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    user = User(
        student_code=payload.student_code,
        name=payload.name,
        role=payload.role,
        user_type_id=payload.user_type_id,
        subject_id=payload.subject_id,
        password_hash=payload.password_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "message": "User created",
        "user": {
            "user_id": user.user_id,
            "student_code": user.student_code,
            "name": user.name,
            "role": user.role,
            "user_type_id": user.user_type_id,
            "subject_id": user.subject_id
        }
    }

# ---------- 👥 ดึงรายชื่อผู้ใช้ ----------
@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "user_id": u.user_id,
            "student_code": u.student_code,
            "name": u.name,
            "role": u.role,
            "user_type_id": u.user_type_id,
            "subject_id": u.subject_id,
        }
        for u in users
    ]