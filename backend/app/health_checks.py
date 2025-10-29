# backend/app/health_checks.py
import os
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone

try:
    from .utils.db import get_mongo_client  # optional helper if present
except Exception:
    get_mongo_client = None

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except Exception:
    AsyncIOMotorClient = None

logger = logging.getLogger("health_checks")
CHECK_INTERVAL = int(os.getenv("CAMERA_HEALTH_INTERVAL", "60"))  # seconds
CAMERAS_COLLECTION = os.getenv("CAMERAS_COLLECTION", "cameras")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB = os.getenv("MONGO_DB", "face_platform")


async def _fetch_snapshot(session: aiohttp.ClientSession, url: str, timeout: int = 5) -> bool:
    try:
        async with session.get(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


async def health_loop(app):
    # Initialize Mongo client (either via helper or direct)
    if get_mongo_client:
        mongo = get_mongo_client()
    else:
        if AsyncIOMotorClient is None:
            logger.error("motor not installed; health checks disabled")
            return
        mongo = AsyncIOMotorClient(MONGO_URI)

    db = mongo[MONGO_DB]
    cameras = db[CAMERAS_COLLECTION]

    logger.info("Health checks started, interval=%s", CHECK_INTERVAL)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # fetch list of cameras from DB
                cam_list = []
                async for cam in cameras.find({}):
                    cam_list.append(cam)

                # check each camera snapshot endpoint
                for cam in cam_list:
                    cam_id = cam.get("_id") or cam.get("cam_id") or cam.get("id")
                    if not cam_id:
                        continue
                    # build snapshot url (assumes the snapshot router we added)
                    snapshot_url = f"http://localhost:8000/api/camera/{cam_id}/snapshot"
                    ok = await _fetch_snapshot(session, snapshot_url)
                    update = {
                        "last_checked": datetime.now(timezone.utc),
                        "healthy": bool(ok),
                    }
                    if ok:
                        update["last_seen"] = datetime.now(timezone.utc)
                    await cameras.update_one({"_id": cam["_id"]}, {"$set": update})
                await asyncio.sleep(CHECK_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Health loop cancelled - shutting down")
                break
            except Exception as e:
                logger.exception("Exception in health loop: %s", e)
                await asyncio.sleep(10)


def init_health_checks(app):
    """
    Call this once from your FastAPI startup code (e.g. in main.py)
    Example:
        from app.health_checks import init_health_checks
        init_health_checks(app)
    """
    loop = asyncio.get_event_loop()
    # create a background task tied to the loop
    task = loop.create_task(health_loop(app))
    # store on app to cancel on shutdown
    if not hasattr(app.state, "bg_tasks"):
        app.state.bg_tasks = []
    app.state.bg_tasks.append(task)

    # register shutdown handler to cancel tasks
    @app.on_event("shutdown")
    async def _cancel_bg_tasks():
        for t in getattr(app.state, "bg_tasks", []):
            t.cancel()
        # optionally wait
        await asyncio.gather(*getattr(app.state, "bg_tasks", []), return_exceptions=True)
