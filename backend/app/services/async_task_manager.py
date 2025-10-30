# backend/app/services/async_task_manager.py
"""
Async Task Manager

Provides:
- enqueue_job(func, kwargs) -> job_id
- get_job(job_id) -> status/result
- background execution in a ThreadPoolExecutor with safe limits.

Designed to offload CPU-bound and blocking ML inference tasks (face recognition,
enhancement, anti-spoof checks) from the FastAPI request/response loop.

This module is intentionally conservative and additive. It integrates with:
- app.state.model_manager (lazy model loading)
- app.state.emit_event(...) to send Socket.IO messages safely

Usage:
    from app.services.async_task_manager import manager
    job_id = await manager.enqueue_job("face_search", payload)
"""

import os
import time
import uuid
import logging
import traceback
import asyncio
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future

from pathlib import Path

# Local app imports (guarded and read-only for protected modules)
from app import state as app_state
from app.state import model_manager, emit_event

# Optional: face_service (protected) - we'll call it conservatively
try:
    from app.services import face_service
except Exception:
    face_service = None

# Optional DB hook (non-fatal if not present)
try:
    from app.utils import db as db_utils  # type: ignore
    _has_db = True
except Exception:
    db_utils = None
    _has_db = False

logger = logging.getLogger("app.async_task_manager")

# Configurable via env
ASYNC_MAX_WORKERS = int(os.getenv("ASYNC_MAX_WORKERS", "1"))
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("ASYNC_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB default
JOB_RETENTION_SECONDS = int(os.getenv("ASYNC_JOB_RETENTION", str(60 * 60)))  # 1 hour default

# In-memory job registry
# job_id -> {
#   "status": "queued"|"running"|"finished"|"failed",
#   "created_at": ts,
#   "started_at": ts|None,
#   "finished_at": ts|None,
#   "result": {...} or None,
#   "error": "trace" or None,
#   "future": Future
# }
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()

# Executor for running blocking jobs
_executor = ThreadPoolExecutor(max_workers=ASYNC_MAX_WORKERS, thread_name_prefix="async-task")

# Ensure a temp dir for job uploads
TMP_DIR = Path(os.getenv("ASYNC_TMP_DIR", "/tmp/frp_async"))  # default temp dir
TMP_DIR.mkdir(parents=True, exist_ok=True)


def _generate_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


async def _register_job(job_id: str, meta: Dict[str, Any]):
    async with _jobs_lock:
        _jobs[job_id] = meta


async def _update_job(job_id: str, **updates):
    async with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


async def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    async with _jobs_lock:
        return _jobs.get(job_id)


def _cleanup_old_jobs():
    """
    Remove job entries older than JOB_RETENTION_SECONDS to keep memory bounded.
    """
    now = time.time()
    stale = []
    for jid, info in list(_jobs.items()):
        created = info.get("created_at", now)
        if now - created > JOB_RETENTION_SECONDS:
            stale.append(jid)
    for jid in stale:
        try:
            _jobs.pop(jid, None)
        except Exception:
            logger.exception("Failed to cleanup stale job %s", jid)


def _safe_call_face_service_search(image_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conservative wrapper that attempts to call a suitable function in face_service.
    Returns a result dict or raises an Exception.
    """
    if face_service is None:
        raise RuntimeError("face_service is not available in this environment")

    # Prefer common function names, try them in order
    candidate_names = ["search_face", "find_matches", "recognize", "process_image", "process_frame"]
    for name in candidate_names:
        fn = getattr(face_service, name, None)
        if callable(fn):
            try:
                # Some implementations expect (image_path, metadata) or just image bytes
                # We try common call patterns (non-invasive).
                try:
                    return fn(image_path, metadata)
                except TypeError:
                    # maybe expects bytes or single arg
                    try:
                        with open(image_path, "rb") as f:
                            data = f.read()
                        return fn(data)
                    except Exception:
                        # last resort: call with no args
                        return fn()
            except Exception:
                # If this candidate failed, keep trying others but log
                logger.exception("Candidate face_service.%s raised exception", name)
                raise
    raise RuntimeError("No suitable face_service function found (expected one of: %s)" % ", ".join(candidate_names))


def _ensure_models_for_face():
    """
    Attempt to ensure some common models are loaded (non-fatal).
    This helps avoid on-demand load inside worker which might increase latency.
    """
    async def _inner():
        # Try to request retinaface & deepfake models if loaders registered
        try:
            # model_manager.get_model is async
            if "retinaface" in getattr(model_manager, "_loaders", {}):
                await model_manager.get_model("retinaface")
        except Exception:
            logger.exception("retinaface loader failed (continuing)")

        try:
            if "deepfake" in getattr(model_manager, "_loaders", {}):
                await model_manager.get_model("deepfake")
        except Exception:
            logger.exception("deepfake loader failed (continuing)")

    # schedule it and wait in a safe manner (worker calls should run in executor)
    loop = asyncio.get_event_loop()
    coro = _inner()
    try:
        loop.run_until_complete(coro)
    except Exception:
        # if the event loop is closed or we are in thread, run coro synchronously
        try:
            asyncio.run(coro)
        except Exception:
            logger.exception("Failed to eagerly load models (non-fatal)")


def _run_blocking_job(func: Callable, *args, **kwargs) -> Any:
    """
    Run a blocking callable in a safe wrapper; this is executed inside threadpool.
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        logger.exception("Exception in blocking job")
        raise


async def enqueue_face_search(file_path: str, metadata: Dict[str, Any]) -> str:
    """
    Enqueue a face-search job using the provided local file_path (image saved on disk).
    Returns job_id immediately (async).
    """
    _cleanup_old_jobs()
    job_id = _generate_job_id()
    now = time.time()
    meta = {
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    await _register_job(job_id, meta)

    # Ensure models are warmed up in non-fatal way (best-effort)
    # We run this in the event loop (it's async). It may load models lazily.
    try:
        # schedule model readiness check but do not block enqueue
        asyncio.create_task(model_manager.get_model("retinaface"))  # best-effort
    except Exception:
        # ignore
        pass

    loop = asyncio.get_event_loop()

    def _job_runner(path, jobid, meta_in):
        """
        Actual job runner executed in a thread.
        Steps:
         - mark job running
         - try to call face_service in conservative manner
         - store result or error
         - emit socket events via app_state.emit_event (async-safe)
        """
        try:
            # synchronous update of job
            # Mark as running (we update via the async update helper)
            asyncio.run(_update_job(jobid, status="running", started_at=time.time()))
        except Exception:
            # If run inside thread without event loop, fallback to setting dict directly
            _jobs[jobid]["status"] = "running"
            _jobs[jobid]["started_at"] = time.time()

        # Emit job_started event (best-effort)
        try:
            asyncio.run(emit_event("job_started", {"job_id": jobid, "type": "face_search"}))
        except Exception:
            # swallow emit errors
            pass

        try:
            # Attempt to call face service (blocking call inside thread)
            # We call a wrapper to pick appropriate function
            result = _safe_call_face_service_search(path, metadata)
            # On success
            try:
                asyncio.run(_update_job(jobid, status="finished", finished_at=time.time(), result=result))
            except Exception:
                _jobs[jobid]["status"] = "finished"
                _jobs[jobid]["finished_at"] = time.time()
                _jobs[jobid]["result"] = result

            # Emit finished event
            try:
                asyncio.run(emit_event("job_finished", {"job_id": jobid, "result": result}))
            except Exception:
                pass

            # Optionally persist result to DB if available
            if _has_db and db_utils:
                try:
                    db = db_utils.get_db()
                    # store minimal job metadata
                    doc = {
                        "job_id": jobid,
                        "type": "face_search",
                        "created_at": meta_in.get("created_at"),
                        "finished_at": time.time(),
                        "result": result,
                    }
                    db["async_jobs"].insert_one(doc)
                except Exception:
                    logger.exception("Failed to persist job result to Mongo (non-fatal)")

            return result

        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("Job %s failed: %s", jobid, exc)
            try:
                asyncio.run(_update_job(jobid, status="failed", finished_at=time.time(), error=str(tb)))
            except Exception:
                _jobs[jobid]["status"] = "failed"
                _jobs[jobid]["finished_at"] = time.time()
                _jobs[jobid]["error"] = str(tb)

            try:
                asyncio.run(emit_event("job_failed", {"job_id": jobid, "error": str(exc)}))
            except Exception:
                pass
            return None

    # Submit the job to executor
    try:
        fut: Future = _executor.submit(_run_blocking_job, _job_runner, file_path, job_id, meta)
        # store future for potential cancellation / inspection
        async with _jobs_lock:
            _jobs[job_id]["future"] = fut
    except Exception as e:
        await _update_job(job_id, status="failed", error=str(e))
        raise

    return job_id


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Return a compact job status for polling.
    """
    job = await _get_job(job_id)
    if not job:
        return {"error": "job not found", "job_id": job_id}
    # copy minimal fields
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


# Expose a singleton manager for convenience
class AsyncTaskManager:
    async def enqueue_face_search(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if metadata is None:
            metadata = {}
        return await enqueue_face_search(file_path, metadata)

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        return await get_job_status(job_id)

    def list_jobs(self):
        # synchronous snapshot
        return list(_jobs.keys())


manager = AsyncTaskManager()
