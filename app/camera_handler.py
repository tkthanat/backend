# app/camera_handler.py
import cv2
import threading
import time
from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class CameraSource:
    cam_id: str              # "entrance" | "exit" | custom
    src: str                 # "0", "1" (device index) หรือ RTSP URL
    is_open: bool = False
    cap: Optional[cv2.VideoCapture] = None
    last_frame: Optional[bytes] = None   # JPEG bytes
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop: bool = field(default=False, repr=False)

class CameraManager:
    def __init__(self, sources: Dict[str, str], fps: int = 12, width: int = 640, height: int = 480):
        """
        sources: {"entrance": "0", "exit":"1"} หรือ RTSP URL
        """
        self.sources: Dict[str, CameraSource] = {
            k: CameraSource(cam_id=k, src=v) for k, v in sources.items()
        }
        self.fps = fps
        self.interval = 1.0 / max(1, fps)
        self.width = width
        self.height = height

    def _open_cap(self, src: str) -> cv2.VideoCapture:
        # แยกกรณี src เป็นเลข (webcam index) หรือเป็น URL
        if src.isdigit():
            cap = cv2.VideoCapture(int(src))
        else:
            cap = cv2.VideoCapture(src)
        # set ขนาดเบื้องต้น (บาง driver อาจไม่รองรับ)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def open(self, cam_id: str):
        cam = self.sources.get(cam_id)
        if not cam:
            raise KeyError(f"Camera '{cam_id}' not found")
        if cam.is_open:
            return
        cam.cap = self._open_cap(cam.src)
        if not cam.cap or not cam.cap.isOpened():
            raise RuntimeError(f"Cannot open camera '{cam_id}' (src={cam.src})")
        cam.is_open = True
        cam._stop = False
        cam._thread = threading.Thread(target=self._reader_loop, args=(cam,), daemon=True)
        cam._thread.start()

    def close(self, cam_id: str):
        cam = self.sources.get(cam_id)
        if not cam:
            return
        cam._stop = True
        if cam._thread and cam._thread.is_alive():
            cam._thread.join(timeout=1.0)
        if cam.cap:
            cam.cap.release()
        cam.is_open = False
        cam._thread = None

    def _reader_loop(self, cam: CameraSource):
        while not cam._stop and cam.cap and cam.cap.isOpened():
            ok, frame = cam.cap.read()
            if ok and frame is not None:
                # encode JPEG
                ok2, jpeg = cv2.imencode(".jpg", frame)
                if ok2:
                    cam.last_frame = jpeg.tobytes()
            time.sleep(self.interval)

    def get_jpeg(self, cam_id: str) -> bytes:
        cam = self.sources.get(cam_id)
        if not cam:
            raise KeyError(f"Camera '{cam_id}' not found")
        if not cam.is_open:
            self.open(cam_id)
        # ถ้ายังไม่มีเฟรม ลองรอสักนิด
        t0 = time.time()
        while cam.last_frame is None and time.time() - t0 < 2.0:
            time.sleep(0.05)
        if cam.last_frame is None:
            raise RuntimeError("No frame yet")
        return cam.last_frame

    def list(self):
        return [
            {
                "cam_id": c.cam_id,
                "src": c.src,
                "is_open": c.is_open,
            }
            for c in self.sources.values()
        ]
