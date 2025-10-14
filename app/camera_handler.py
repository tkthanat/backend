# app/camera_handler.py
import cv2
import threading
import time
import platform
from glob import glob
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from .ai_engine import annotate_and_match

def _backend_flag():
    # ลดปัญหาดีเลย์เปิดกล้องบน Windows
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW  # หรือ CAP_MSMF ก็ได้ตามไดรเวอร์
    return 0

@dataclass
class CameraSource:
    cam_id: str
    src: str                  # "0","1" หรือ RTSP URL
    is_open: bool = False
    cap: Optional[cv2.VideoCapture] = None
    last_frame: Optional[bytes] = None
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop: bool = field(default=False, repr=False)

class CameraManager:
    def __init__(self, sources: Dict[str, str], fps: int = 10, width: int = 640, height: int = 480):
        self.sources = {k: CameraSource(k, v) for k, v in sources.items()}
        self.interval = 1.0 / max(1, fps)
        self.width = width
        self.height = height

    def _open_cap(self, src: str) -> cv2.VideoCapture:
        backend = _backend_flag()
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), backend)
        else:
            cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def open(self, cam_id: str):
        cam = self.sources.get(cam_id)
        if not cam: raise KeyError(f"camera '{cam_id}' not found")
        if cam.is_open: return
        cam.cap = self._open_cap(cam.src)
        if not cam.cap or not cam.cap.isOpened():
            raise RuntimeError(f"Cannot open camera '{cam_id}' ({cam.src})")
        cam.is_open = True
        cam._stop = False
        cam._thread = threading.Thread(target=self._loop, args=(cam,), daemon=True)
        cam._thread.start()

    def close(self, cam_id: str):
        cam = self.sources.get(cam_id)
        if not cam: return
        cam._stop = True
        if cam._thread and cam._thread.is_alive():
            cam._thread.join(timeout=1.0)
        if cam.cap:
            cam.cap.release()
        cam.is_open = False
        cam._thread = None

    def _loop(self, cam: CameraSource):
        while not cam._stop and cam.cap and cam.cap.isOpened():
            ok, frame = cam.cap.read()
            if ok and frame is not None:
                try:
                    _ = annotate_and_match(frame)  # วาดกรอบ/ชื่อบนเฟรม
                except Exception:
                    pass
                ok2, jpg = cv2.imencode(".jpg", frame)
                if ok2:
                    cam.last_frame = jpg.tobytes()
            time.sleep(self.interval)

    def get_jpeg(self, cam_id: str) -> bytes:
        cam = self.sources.get(cam_id)
        if not cam:
            raise KeyError(f"camera '{cam_id}' not found")
        if not cam.is_open:
            self.open(cam_id)
        t0 = time.time()
        while cam.last_frame is None and time.time() - t0 < 2.0:
            time.sleep(0.05)
        if cam.last_frame is None:
            raise RuntimeError("no frame yet")
        return cam.last_frame

    def list(self):
        return [{"cam_id": c.cam_id, "src": c.src, "is_open": c.is_open} for c in self.sources.values()]

    # ------- NEW: reconfigure sources --------
    def reconfigure(self, new_sources: Dict[str, str]):
        # ปิดทุกตัวก่อน
        for k in list(self.sources.keys()):
            self.close(k)
        # ตั้ง src ใหม่
        self.sources = {k: CameraSource(k, v) for k, v in new_sources.items()}

# ------- NEW: device discovery helpers -------
def discover_local_devices(max_index: int = 10, test_frame: bool = True) -> List[dict]:
    """
    สำรวจกล้องที่ระบบน่าจะมี:
      - Linux: /dev/video*
      - Windows/macOS: ลอง index 0..max_index-1
    """
    devices: List[dict] = []
    backend = _backend_flag()

    candidates: List[str] = []
    if platform.system() == "Linux":
        # ดึง /dev/video* แล้ว map เป็น index ที่คุ้นเคย
        vids = sorted(glob("/dev/video*"))
        # เอาเลขท้ายไฟล์มาเป็นดัชนี
        for v in vids:
            try:
                idx = str(int(v.split("video")[-1]))
                candidates.append(idx)
            except:
                pass
    else:
        # Windows/macOS — brute force index
        candidates = [str(i) for i in range(max_index)]

    seen = set()
    for src in candidates:
        if src in seen: continue
        seen.add(src)
        cap = cv2.VideoCapture(int(src), backend) if src.isdigit() else cv2.VideoCapture(src)
        opened = cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0

        ok_frame = False
        if opened and test_frame:
            ok, frame = cap.read()
            ok_frame = bool(ok and frame is not None)

        if cap: cap.release()
        devices.append({
            "src": src,             # ใช้ค่านี้ไปตั้งใน config ได้เลย
            "opened": opened,
            "readable": ok_frame,
            "width": w, "height": h
        })
    return devices
