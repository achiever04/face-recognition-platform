# backend/app/health_checks.py
"""
Health check background loop for camera availability.

Behavior:
- Periodically fetch camera list from DB and check snapshot endpoint for each camera.
- Maintains per-camera fields:
    - last_checked (UTC ISO)
    - last_seen (UTC ISO) when a healthy response seen
    - healthy (bool)
    - consecutive_failures (int)
    - next_check_at (UTC timestamp float) -> skip checks until reached (exponential backoff)
- Supports Motor (async) if installed; otherwise uses sync pymongo via executor.
- Exposes init_health_checks(app) to start background task at FastAPI startup.
"""

import os
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from app.utils.logger import get_logger

logger = get_logger("app.health_checks")

# Config (env)
CHECK_INTERVAL = int(os.getenv("CAMERA_HEALTH_INTERVAL", "30"))  # seconds between global loops
REQUEST_TIMEOUT = int(os.getenv("CAMERA_HEALTH_REQUEST_TIMEOUT", "4"))  # per-camera http timeout
CAMERAS_COLLECTION = os.getenv("CAMERAS_COLLECTION", "cameras")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB_NAME", "face_recognition_db")
FAILURE_BACKOFF_BASE = int(os.getenv("CAMERA_BACKOFF_BASE", "10"))  # seconds base for backoff
MAX_BACKOFF_SECONDS = int(os.getenv("CAMERA_BACKOFF_MAX", str(60 * 60)))  # up to 1 hour

# Try Motor first for async DB updates
try:
    from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
except Exception:
    AsyncIOMotorClient = None

# Fallback: use synchronous pymongo via helper if provided
try:
    from app.utils.db import get_mongo_client  # returns pymongo.MongoClient
except Exception:
    get_mongo_client = None

def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

async def _fetch_snapshot_ok(session: aiohttp.ClientSession, url: str, timeout: int = REQUEST_TIMEOUT) -> bool:
    try:
        async with session.get(url, timeout=timeout) as resp:
            # treat 200 or 304 as healthy (304 if conditional)
            return resp.status == 200 or resp.status == 304
    except asyncio.TimeoutError:
        return False
    except Exception:
        return False

async def _process_camera_doc(cam: Dict[str, Any], session: aiohttp.ClientSession, cameras_coll, use_motor: bool):
    """
    Check a single camera doc, apply backoff logic, and update DB.
    camera doc schema expected to at least contain an identifier field (_id or id or cam_id)
    """
    # Identify camera id string used in snapshot route
    cam_id = cam.get("cam_id") or cam.get("id") or cam.get("_id")
    doc_id = cam.get("_id")  # used for filtering DB updates
    if cam_id is None or doc_id is None:
        return

    # Backoff logic: if next_check_at exists and in future, skip check
    next_check_at = cam.get("next_check_at")
    if next_check_at:
        try:
            if float(next_check_at) > asyncio.get_event_loop().time():
                logger.debug("Skipping camera %s until next_check_at=%s", cam_id, next_check_at)
                return
        except Exception:
            pass

    snapshot_url = f"http://127.0.0.1:8000/api/camera/{cam_id}/snapshot"
    ok = await _fetch_snapshot_ok(session, snapshot_url)

    # Prepare DB update
    now_iso = _utc_iso_now()
    update = {"last_checked": now_iso, "healthy": bool(ok)}
    if ok:
        update["last_seen"] = now_iso
        update["consecutive_failures"] = 0
        update["next_check_at"] = None
    else:
        # increment failures
        current_failures = int(cam.get("consecutive_failures", 0)) + 1
        update["consecutive_failures"] = current_failures
        # compute exponential backoff (in seconds) and set next_check_at as loop.time() + backoff
        backoff = min(FAILURE_BACKOFF_BASE * (2 ** (current_failures - 1)), MAX_BACKOFF_SECONDS)
        next_ts = asyncio.get_event_loop().time() + backoff
        update["next_check_at"] = float(next_ts)
        logger.debug("Camera %s failed health check (#%d). next_check in %s sec", cam_id, current_failures, backoff)

    # Apply update (motor sync or pymongo within executor)
    try:
        if use_motor:
            await cameras_coll.update_one({"_id": doc_id}, {"$set": update})
        else:
            # synchronous client passed as cameras_coll; run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: cameras_coll.update_one({"_id": doc_id}, {"$set": update}))
    except Exception:
        logger.exception("Failed to persist health update for camera %s (ignored)", cam_id)

async def health_loop(app):
    """
    Background loop that periodically polls cameras list and checks snapshot endpoints.
    """
    logger.info("Starting health loop (interval=%s)", CHECK_INTERVAL)
    # Initialize DB client
    use_motor = False
    motor_client = None
    pymongo_db = None
    cameras_coll = None

    if AsyncIOMotorClient is not None:
        try:
            motor_client = AsyncIOMotorClient(MONGO_URI)
            cameras_coll = motor_client[MONGO_DB][CAMERAS_COLLECTION]
            use_motor = True
            logger.info("Using Motor async client for health checks")
        except Exception:
            motor_client = None
            use_motor = False

    if not use_motor:
        # try to use sync pymongo via helper
        if get_mongo_client is not None:
            try:
                client = get_mongo_client()
                pymongo_db = client[MONGO_DB]
                cameras_coll = pymongo_db[CAMERAS_COLLECTION]
                logger.info("Using sync pymongo client for health checks (run in executor)")
            except Exception:
                cameras_coll = None
        else:
            logger.warning("No DB client available for health checks; will still attempt HTTP checks but won't persist results")
            cameras_coll = None

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # load camera docs - if motor, use async cursor; else run in executor
                cam_docs = []
                if use_motor and cameras_coll is not None:
                    cursor = cameras_coll.find({})
                    async for doc in cursor:
                        cam_docs.append(doc)
                elif cameras_coll is not None:
                    # sync cursor in executor
                    loop = asyncio.get_event_loop()
                    def _fetch_sync():
                        return list(cameras_coll.find({}))
                    cam_docs = await loop.run_in_executor(None, _fetch_sync)
                else:
                    cam_docs = []

                # iterate through camera docs
                tasks = []
                for cam in cam_docs:
                    tasks.append(_process_camera_doc(cam, session, cameras_coll, use_motor))
                if tasks:
                    # run them with concurrency limit to avoid too many parallel connections
                    sem = asyncio.Semaphore(int(os.getenv("HEALTH_CONCURRENCY", "10")))
                    async def _wrap(task):
                        async with sem:
                            await task
                    await asyncio.gather(*[_wrap(t) for t in tasks], return_exceptions=True)

                # sleep until next loop
                await asyncio.sleep(CHECK_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Health loop cancelled - shutting down")
                break
            except Exception:
                logger.exception("Unhandled exception in health loop (continuing) - sleeping briefly")
                await asyncio.sleep(5)

def init_health_checks(app):
    """
    Start health checks loop. Call this once from FastAPI startup (main.py).
    """
    loop = asyncio.get_event_loop()
    task = loop.create_task(health_loop(app))
    if not hasattr(app.state, "bg_tasks"):
        app.state.bg_tasks = []
    app.state.bg_tasks.append(task)

    @app.on_event("shutdown")
    async def _cancel_bg_tasks():
        for t in getattr(app.state, "bg_tasks", []):
            t.cancel()
        await asyncio.gather(*getattr(app.state, "bg_tasks", []), return_exceptions=True)
