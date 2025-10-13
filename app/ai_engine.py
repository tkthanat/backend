import numpy as np
import cv2
from typing import List, Dict, Any
from insightface.app import FaceAnalysis

_face_app=None

def init_ai():
    global _face_app
    if _face_app is not None:
        return
    _face_app = FaceAnalysis(name="buffalo_l")
    _face_app.prepare(ctx_id=0, det_size=(640, 640))

def extract_faces(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    assert _face_app is not None, "AI engine not initialized"
    faces = _face_app.get(frame_bgr)
    out=[]
    for f in faces:
        bbox=f.bbox.astype(int).tolist()
        emb=f.normed_embedding.astype(float).tolist()
        gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
        x1,y1,x2,y2=bbox
        crop=gray[max(0,y1):y2,max(0,x1):x2]
        liveness=float(cv2.Laplacian(crop,cv2.CV_64F).var()/1000.0)
        out.append({"bbox":bbox,"embedding":emb,"liveness":liveness})
    return out