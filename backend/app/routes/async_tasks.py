# backend/app/routes/async_tasks.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import asyncio
import logging

from app.services.async_task_manager import manager
from app import state as app_state

router = APIRouter(prefix="/async", tags=["async"])

logger = logging.getLogger("app.routes.async_tasks")

# temp dir should match manager.TMP_DIR if needed; allow overriding via env
TMP_DIR = Path(os.getenv("ASYNC_TMP_DIR", "/tmp/frp_async"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("ASYNC_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5MB default


@router.post("/face/search")
async def async_face_search(file: UploadFile = File(...), cam_id: str = Form(None)):
    """
    Queue a face search job. Accepts multipart file upload (image).
    Returns: {"job_id": "...", "status": "queued"}
    """
    # Basic validation: content length
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # Save to temp file
    tmp_path = TMP_DIR / f"{uuid_safe_filename(file.filename)}_{int(time.time())}.jpg"
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.exception("Failed to write temp upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save upload")

    # Metadata (e.g., camera id)
    metadata = {"cam_id": cam_id} if cam_id else {}

    # Enqueue job
    try:
        job_id = await manager.enqueue_face_search(str(tmp_path), metadata)
    except Exception as e:
        logger.exception("Failed to enqueue job: %s", e)
        raise HTTPException(status_code=500, detail="Failed to enqueue job")

    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    status = await manager.get_status(job_id)
    if status.get("error") == "job not found":
        raise HTTPException(status_code=404, detail="job not found")
    return status


# small helper
def uuid_safe_filename(filename: str) -> str:
    import uuid, re
    base = Path(filename).stem
    base = re.sub(r"[^a-zA-Z0-9_-]", "_", base)
    return f"{base}_{uuid.uuid4().hex[:8]}"
