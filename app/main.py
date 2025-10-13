from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from .db_models import get_db, Subject, User
from .schemas import SubjectCreate, UserEnroll
from fastapi import Response
from fastapi.responses import StreamingResponse
from .camera_handler import CameraManager
import asyncio
import base64

app = FastAPI(title="Offline Attendance")

# ตั้งค่า source กล้องเริ่มต้น (โน้ตบุ๊กเทส webcam ก่อน)
# "0" = default webcam; เปลี่ยนเป็น "rtsp://..." ได้
CAMERA_SOURCES = {
    "entrance": "0",
    "exit": "1",   # ถ้ามี webcam ตัวเดียว ใช้ "0" ทั้งสองก็ได้ แต่จะเป็นฟีดเดียวกัน
}
cam_mgr = CameraManager(CAMERA_SOURCES, fps=12, width=640, height=480)


class WSManager:
    def __init__(self):
        self.active=set()
    async def connect(self, ws: WebSocket):
        await ws.accept(); self.active.add(ws)
    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
    async def broadcast(self, data: dict):
        dead=[]
        for ws in list(self.active):
            try: await ws.send_json(data)
            except: dead.append(ws)
        for d in dead: self.disconnect(d)

ws_manager=WSManager()

@app.route('/')
def home():
    return "Hello, Flask!"

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

@app.post("/subjects")
def create_subject(payload: SubjectCreate, db: Session = Depends(get_db)):
    s=Subject(**payload.model_dump())
    db.add(s); db.commit(); db.refresh(s)
    return s

@app.get("/subjects")
def list_subjects(db: Session = Depends(get_db)):
    return db.query(Subject).all()

@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.post("/users")
def create_user(payload: UserEnroll, db: Session = Depends(get_db)):
    s=User(**payload.model_dump())
    db.add(s); db.commit(); db.refresh(s)
    return s


# ---------- Cameras API ----------
@app.get("/cameras")
def list_cameras():
    """
    ดูรายการกล้อง + สถานะ
    """
    return cam_mgr.list()

@app.post("/cameras/{cam_id}/open")
def open_camera(cam_id: str):
    cam_mgr.open(cam_id)
    return {"message": f"camera '{cam_id}' opened"}

@app.post("/cameras/{cam_id}/close")
def close_camera(cam_id: str):
    cam_mgr.close(cam_id)
    return {"message": f"camera '{cam_id}' closed"}

@app.get("/cameras/{cam_id}/snapshot", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response)
def camera_snapshot(cam_id: str):
    """
    ดึงภาพล่าสุดเป็น JPEG (เหมาะกับทดสอบใน Postman/Browser)
    """
    jpeg = cam_mgr.get_jpeg(cam_id)
    return Response(content=jpeg, media_type="image/jpeg")

@app.get("/cameras/{cam_id}/mjpeg")
def camera_mjpeg(cam_id: str):
    """
    สตรีม MJPEG (เปิดดูใน Browser เช่น Chrome/Firefox)
    Postman จะไม่พรีวิววิดีโอ แต่ request ได้
    """
    boundary = "frame"

    async def gen():
        while True:
            try:
                jpeg = cam_mgr.get_jpeg(cam_id)
                yield (
                        b"--" + boundary.encode() + b"\r\n"
                                                    b"Content-Type: image/jpeg\r\n"
                                                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        + jpeg + b"\r\n"
                )
            except Exception:
                # ถ้าเฟรมหาย พักนิดแล้วลองใหม่
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.03)  # ~30–33ms per frame (~30fps theoretical)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

# ---------- WebSocket ส่งเฟรมเป็น base64 ----------
@app.websocket("/ws/cameras/{cam_id}")
async def ws_camera(ws: WebSocket, cam_id: str):
    """
    ส่งเฟรม jpeg เป็น base64 JSON {"type":"frame","data":"..."} ~10–12fps
    (เหมาะกับ React Dashboard ฝั่งหน้าเว็บ)
    """
    await ws.accept()
    # ensure camera is opened
    try:
        cam_mgr.open(cam_id)
    except Exception:
        pass

    try:
        while True:
            await asyncio.sleep(1/10)  # ประมาณ 10 fps
            try:
                jpeg = cam_mgr.get_jpeg(cam_id)
                b64 = base64.b64encode(jpeg).decode("ascii")
                await ws.send_json({"type": "frame", "cam_id": cam_id, "data": b64})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass