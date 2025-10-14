# app/ai_engine.py
# -*- coding: utf-8 -*-
"""
AI Engine สำหรับ Face Recognition ด้วย InsightFace
- โหลด/รีเฟรช facebank จากรูปในระบบ (เฉลี่ยเวคเตอร์ต่อ user)
- เก็บ cache ทั้ง embeddings และชื่อผู้ใช้ในหน่วยความจำ
- annotate_and_match(): ตรวจจับ + วาดกรอบ + แสดง "ชื่อ (เปอร์เซ็นต์)"
"""

from __future__ import annotations
import os
import cv2
import json
import numpy as np
from typing import Dict, Tuple, List, Optional
from insightface.app import FaceAnalysis

# ===== Config =====
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "./data/faces/train")   # รูปเก็บเป็น /{user_id}/{filename}
FACEBANK_PATH = os.getenv("FACEBANK_PATH", "./data/facebank.npz")
FACEBANK_NAMES = FACEBANK_PATH + ".names.json"               # เซฟชื่อผู้ใช้คู่กับ facebank
THRESH = float(os.getenv("RECOG_THRESHOLD", "0.45"))         # cosine similarity threshold (0..1)

# ===== Globals =====
_app: Optional[FaceAnalysis] = None
_facebank: Dict[int, np.ndarray] = {}   # {user_id: mean_embedding(512,)}
_usernames: Dict[int, str] = {}         # {user_id: name}


# ===== Utils =====
def _ensure_app() -> FaceAnalysis:
    """โหลดโมเดลครั้งเดียว (lazy)"""
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        # ctx_id=0: ถ้ามี GPU จะใช้ GPU (onnxruntime-gpu), ถ้าไม่มีจะวิ่ง CPU อัตโนมัติ
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# ===== Public API =====
def load_facebank() -> int:
    """
    โหลด embeddings (.npz) + รายชื่อ (.names.json) เข้าหน่วยความจำ
    เรียกตอน startup ของ FastAPI
    """
    global _facebank, _usernames

    # embeddings
    if os.path.exists(FACEBANK_PATH):
        data = np.load(FACEBANK_PATH)
        _facebank = {int(k): data[k] for k in data.files}
    else:
        _facebank = {}

    # names
    if os.path.exists(FACEBANK_NAMES):
        with open(FACEBANK_NAMES, "r", encoding="utf-8") as f:
            _usernames = {int(k): v for k, v in json.load(f).items()}
    else:
        _usernames = {}

    return len(_facebank)


def refresh_facebank_from_db(rows: List[Tuple[int, str, str]]) -> Tuple[int, int]:
    """
    สร้าง facebank จากรายการรูปใน DB แล้วเซฟเป็นไฟล์ + อัปเดต cache ในหน่วยความจำ
    rows: [(user_id, file_path, name), ...]
      - file_path อาจเป็นชื่อไฟล์เฉย ๆ: จะ map เป็น MEDIA_ROOT/{user_id}/{basename(file_path)}
      - ชื่อไฟล์ absolute ก็ใช้ตามนั้น
    return: (จำนวนผู้ใช้ที่ได้ facebank, จำนวนรูปที่ถูกใช้)
    """
    app = _ensure_app()

    files_by_uid: Dict[int, List[str]] = {}
    names: Dict[int, str] = {}

    # รวมไฟล์ต่อ user + สร้าง mapping ชื่อ
    for uid, fp, name in rows:
        base = os.path.basename(fp)
        full = fp if os.path.isabs(fp) else os.path.join(MEDIA_ROOT, str(uid), base)
        files_by_uid.setdefault(uid, []).append(full)
        if uid not in names:
            names[uid] = name or f"UID {uid}"

    facebank: Dict[int, np.ndarray] = {}
    total_imgs = 0

    for uid, files in files_by_uid.items():
        embs = []
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            emb = _normalize(faces[0].normed_embedding.astype(np.float32))
            embs.append(emb)
            total_imgs += 1
        if embs:
            facebank[uid] = np.mean(embs, axis=0).astype(np.float32)

    # สร้างโฟลเดอร์ปลายทาง
    dst_dir = os.path.dirname(FACEBANK_PATH) or "."
    os.makedirs(dst_dir, exist_ok=True)

    # เซฟไฟล์ facebank + names (ใช้ str key สำหรับ np.savez/json)
    np.savez(FACEBANK_PATH, **{str(k): v for k, v in facebank.items()})
    with open(FACEBANK_NAMES, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in names.items()}, f, ensure_ascii=False)

    # อัปเดต cache ในหน่วยความจำทันที
    global _facebank, _usernames
    _facebank = facebank
    _usernames = names

    return len(facebank), total_imgs


def _best_match(emb: np.ndarray) -> Tuple[Optional[int], float]:
    """
    คำนวณ similarity กับทุกคนใน facebank (cosine dot เพราะ normalized แล้ว)
    คืน (user_id, similarity)
    """
    if not _facebank:
        return None, -1.0
    best_uid, best_sim = None, -1.0
    for uid, ref in _facebank.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_sim, best_uid = sim, uid
    return best_uid, best_sim


def annotate_and_match(frame: np.ndarray) -> List[dict]:
    """
    ตรวจจับใบหน้าในเฟรม + จับคู่กับ facebank + วาดกรอบ/ชื่อบนภาพ
    return: รายการผลลัพธ์ (สำหรับ broadcast/บันทึก)
    """
    app = _ensure_app()
    faces = app.get(frame)
    results: List[dict] = []

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        emb = _normalize(f.normed_embedding.astype(np.float32))

        uid, sim = _best_match(emb)
        if uid is not None and sim >= THRESH:
            name = _usernames.get(uid, f"UID {uid}")
            label = f"{name} ({sim * 100:.1f}%)"
            color = (0, 200, 0)  # เขียว
            matched = True
        else:
            label = "Unknown"
            color = (0, 0, 255)  # แดง
            matched = False

        # วาด overlay
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        results.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "user_id": uid,
            "name": _usernames.get(uid) if uid is not None else None,
            "similarity": float(sim) if sim is not None else None,
            "matched": matched,
            "label": label,
        })

    return results
