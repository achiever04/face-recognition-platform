# backend/app/routes/snapshot.py
from fastapi import APIRouter, HTTPException, Response, Query
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional
import os
from ..utils.thumbnail_cache import get_snapshot, set_snapshot
from pathlib import Path
import aiofiles
import asyncio

router = APIRouter(prefix="/api/camera", tags=["camera_snapshot"])

# Snapshot storage fallback dir
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "/app/backend/data/snapshots")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "/app/backend/data/uploads")

@router.get("/{cam_id}/snapshot", summary="Get a snapshot for a camera (cached)")
async def camera_snapshot(cam_id: str, enhance: Optional[bool] = Query(False)):
    """
    Returns latest snapshot bytes for camera `cam_id`.
    - tries cache (redis/memory)
    - falls back to disk snapshot (SNAPSHOT_DIR)
    - falls back to uploads (UPLOADS_DIR/<cam_id>.*)
    If `enhance=true` is passed, the endpoint sets a response header to indicate an enhancement request
    (the actual enhancement pipeline is separate).
    """
    # Try cache
    data = await get_snapshot(cam_id)
    if data:
        headers = {}
        if enhance:
            headers["X-Enhance-Requested"] = "1"
        return Response(content=data, media_type="image/jpeg", headers=headers)

    # Try disk snapshot path
    hash_path = Path(SNAPSHOT_DIR) / f"{cam_id}.jpg"
    if hash_path.exists():
        return FileResponse(str(hash_path), media_type="image/jpeg")

    # Try uploads dir: find any image file that starts with cam_id
    p = Path(UPLOADS_DIR)
    if p.exists():
        for ext in ("jpg", "jpeg", "png", "bmp"):
            candidate = p / f"{cam_id}.{ext}"
            if candidate.exists():
                return FileResponse(str(candidate), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Snapshot not found")
