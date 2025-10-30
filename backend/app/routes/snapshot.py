# backend/app/routes/snapshot.py
"""
Snapshot endpoint for camera thumbnails.

Behavior:
 - Try cache (get_snapshot)
 - Try disk snapshots in SNAPSHOT_DIR
 - Try uploads in UPLOADS_DIR
 - If enhance=true: set a header and schedule a non-blocking background attempt
   to run any available enhancement pipeline (best-effort; no hard dependency).
 - Return appropriate caching headers (ETag, Last-Modified, Cache-Control)
 - Honor If-None-Match / If-Modified-Since -> return 304 if unchanged
 - If nothing found, return a small SVG placeholder (friendly for <img> tags)
"""

from fastapi import APIRouter, HTTPException, Response, Query, Request, status
from fastapi.responses import StreamingResponse
from typing import Optional
import os
from ..utils.thumbnail_cache import get_snapshot, set_snapshot
from pathlib import Path
import aiofiles
import asyncio
from app.utils.logger import get_logger
import hashlib
import email.utils

router = APIRouter(prefix="/api/camera", tags=["camera_snapshot"])

logger = get_logger("app.routes.snapshot")

# Snapshot storage fallback dir (configurable)
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "data/snapshots")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "data/uploads")

# Cache-control policy for snapshots (short-lived)
CACHE_CONTROL = os.getenv("SNAPSHOT_CACHE_CONTROL", "public, max-age=5")  # 5s default

def _file_etag(path: Path) -> str:
    try:
        st = path.stat()
        et = f"{int(st.st_mtime)}-{st.st_size}"
        return hashlib.sha1(et.encode("utf-8")).hexdigest()
    except Exception:
        return ""

def _http_date_from_timestamp(ts: float) -> str:
    return email.utils.formatdate(ts, usegmt=True)

async def _read_file_bytes_async(path: Path) -> bytes:
    async with aiofiles.open(path, "rb") as f:
        return await f.read()

async def _maybe_schedule_enhancement(cam_id: str, src_path: Optional[Path] = None):
    async def _task():
        try:
            try:
                from app import state as app_state
            except Exception:
                app_state = None

            if app_state is not None:
                mm = getattr(app_state, "model_manager", None)
                if mm is not None:
                    enh = getattr(mm, "enhance_snapshot", None)
                    if callable(enh):
                        logger.debug("Running model_manager.enhance_snapshot for cam %s", cam_id)
                        res = enh(cam_id, src_path) if src_path is not None else enh(cam_id)
                        if asyncio.iscoroutine(res):
                            await res
                        return
            try:
                enh_mod = __import__("app.services.enhancer", fromlist=["enhance_snapshot"])
                enh_func = getattr(enh_mod, "enhance_snapshot", None)
                if callable(enh_func):
                    logger.debug("Running app.services.enhancer.enhance_snapshot for cam %s", cam_id)
                    res = enh_func(cam_id, src_path) if src_path is not None else enh_func(cam_id)
                    if asyncio.iscoroutine(res):
                        await res
                    return
            except Exception:
                pass
            logger.debug("No enhancement pipeline available for camera %s (skip)", cam_id)
        except Exception as e:
            logger.exception("Background enhancement task for cam %s failed (ignored): %s", cam_id, e)

    try:
        asyncio.create_task(_task())
    except Exception:
        logger.debug("asyncio.create_task failed for enhancement; scheduling via loop.run_in_executor")
        try:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, lambda: None)
        except Exception:
            pass

_PLACEHOLDER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180">
  <rect width="100%" height="100%" fill="#f0f0f0"/>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#999" font-family="Arial" font-size="14">
    No snapshot available
  </text>
</svg>"""

@router.get("/{cam_id}/snapshot", summary="Get a snapshot for a camera (cached)")
async def camera_snapshot(request: Request, cam_id: str, enhance: Optional[bool] = Query(False)):
    try:
        cached = await get_snapshot(cam_id)
        if cached:
            etag = hashlib.sha1(cached).hexdigest()
            inm = request.headers.get("if-none-match")
            if inm and inm.strip('"') == etag:
                return Response(status_code=status.HTTP_304_NOT_MODIFIED)

            headers = {"ETag": f'"{etag}"', "Cache-Control": CACHE_CONTROL}
            if enhance:
                headers["X-Enhance-Requested"] = "1"
                await _maybe_schedule_enhancement(cam_id, None)
            return Response(content=cached, media_type="image/jpeg", headers=headers)

        disk_path = Path(SNAPSHOT_DIR) / f"{cam_id}.jpg"
        if disk_path.exists():
            etag = _file_etag(disk_path)
            last_mod = disk_path.stat().st_mtime
            last_mod_http = _http_date_from_timestamp(last_mod)
            inm = request.headers.get("if-none-match")
            if inm and inm.strip('"') == etag:
                return Response(status_code=status.HTTP_304_NOT_MODIFIED)
            ims = request.headers.get("if-modified-since")
            if ims:
                try:
                    ims_ts = email.utils.parsedate_to_datetime(ims).timestamp()
                    if last_mod <= ims_ts:
                        return Response(status_code=status.HTTP_304_NOT_MODIFIED)
                except Exception:
                    pass
            try:
                content = await _read_file_bytes_async(disk_path)
                try:
                    await set_snapshot(cam_id, content)
                except Exception:
                    logger.debug("set_snapshot failed (ignored) for cam %s", cam_id)
                headers = {"ETag": f'"{etag}"', "Last-Modified": last_mod_http, "Cache-Control": CACHE_CONTROL}
                if enhance:
                    headers["X-Enhance-Requested"] = "1"
                    await _maybe_schedule_enhancement(cam_id, disk_path)
                return Response(content=content, media_type="image/jpeg", headers=headers)
            except Exception as e:
                logger.exception("Failed to read snapshot file %s: %s", disk_path, e)

        p = Path(UPLOADS_DIR)
        if p.exists() and p.is_dir():
            for ext in ("jpg", "jpeg", "png", "bmp"):
                candidate = p / f"{cam_id}.{ext}"
                if candidate.exists():
                    try:
                        content = await _read_file_bytes_async(candidate)
                        try:
                            await set_snapshot(cam_id, content)
                        except Exception:
                            logger.debug("set_snapshot failed (ignored) for cam %s", cam_id)
                        headers = {"Cache-Control": CACHE_CONTROL}
                        if enhance:
                            headers["X-Enhance-Requested"] = "1"
                            await _maybe_schedule_enhancement(cam_id, candidate)
                        return Response(content=content, media_type="image/jpeg", headers=headers)
                    except Exception:
                        logger.exception("Failed to read upload candidate %s (ignored)", candidate)
                        continue

        logger.debug("Snapshot not found for cam %s (returning placeholder)", cam_id)
        headers = {"Cache-Control": "no-cache, no-store", "X-Placeholder": "1"}
        if enhance:
            await _maybe_schedule_enhancement(cam_id, None)
            headers["X-Enhance-Requested"] = "1"
        return Response(content=_PLACEHOLDER_SVG, media_type="image/svg+xml", status_code=status.HTTP_404_NOT_FOUND, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in camera_snapshot(%s): %s", cam_id, e)
        headers = {"Cache-Control": "no-cache, no-store", "X-Error": "1"}
        return Response(content=_PLACEHOLDER_SVG, media_type="image/svg+xml", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, headers=headers)
