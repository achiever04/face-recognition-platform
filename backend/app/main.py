# backend/app/main.py
# --- IMPORTS ---
import os
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Socket.IO
import socketio

# Local app imports
from dotenv import load_dotenv

# Protected & existing routers / modules (kept unchanged)
from app.routes import camera, federated, deepfake, face, alerts
from app.routes import snapshot as snapshot_router  # new snapshot router
from app.health_checks import init_health_checks

# State and helpers
from app import state as app_state
from app.state import init_cameras, UPLOAD_DIR, SIO_MANAGER, model_manager, MODEL_IDLE_UNLOAD_SECONDS

# Optional logger setup helper (kept non-mandatory)
from app.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# --- Logging setup (optional) ---
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "false").lower() in ("1", "true", "yes")
if ENABLE_LOGGING:
    try:
        setup_logger()
    except Exception:
        logging.getLogger("app.main").exception("setup_logger() failed (continuing without file logger)")

logger = logging.getLogger("app.main")

# --- FastAPI app creation ---
api = FastAPI(title="Multi-Camera Face Recognition Platform")

# --- CORS configuration (configurable via env) ---
_frontend_origins = os.getenv("FRONTEND_ORIGINS", "*")
if _frontend_origins and _frontend_origins != "*":
    # allow comma-separated origins
    allow_origins = [o.strip() for o in _frontend_origins.split(",") if o.strip()]
else:
    # if wildcard, pass "*" which is accepted by both CORSMiddleware and socketio
    allow_origins = ["*"]

api.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Socket.IO server ---
# Use the shared AsyncManager from app.state for cross-module emission support
# Important: provide cors_allowed_origins so engineio/socketio handshake allows frontends
sio = socketio.AsyncServer(
    async_mode="asgi",
    client_manager=SIO_MANAGER,
    cors_allowed_origins=allow_origins,  # allow same origins as FastAPI CORS
)

# --- Ensure upload directory exists ---
os.makedirs(app_state.UPLOAD_DIR, exist_ok=True)

# --- Initialize cameras (non-destructive, uses app.state.init_cameras) ---
# These camera configs are examples; keep them as-is (you can modify via env or DB)
camera_configs = [
    {"id": 0, "name": "Shivaji Nagar Chauk 1", "geo": (18.555, 73.808)},
    {"id": 1, "name": "Pune Station", "geo": (18.528, 73.847)},
    {"id": 2, "name": "FC Road Signal", "geo": (18.516, 73.841)},
    {"id": 3, "name": "Kothrud Square", "geo": (18.504, 73.823)},
    {"id": 4, "name": "Swargate Bus Stop", "geo": (18.501, 73.862)},
]
# Initialize cameras conservatively (init_cameras is safe and additive)
try:
    init_cameras(camera_configs)
except Exception:
    logger.exception("init_cameras failed (continuing with best-effort)")

# --- Include FastAPI routers (keeps your original routers intact) ---
# Note: camera.router will be mounted at /camera (as before)
api.include_router(camera.router, prefix="/camera")
api.include_router(federated.router)
api.include_router(deepfake.router)
api.include_router(alerts.router)
api.include_router(face.router)  # keep order as before

# Include the snapshot router you added (it defines its own prefix inside the file)
try:
    api.include_router(snapshot_router.router)
except Exception:
    logger.exception("Failed to include snapshot router (snapshot endpoints unavailable)")

# --- Root health endpoint ---
@api.get("/")
def read_root():
    return {"message": "FastAPI backend is running"}

# --- Camera status endpoint (keeps original behavior but uses app_state) ---
@api.get("/camera/status")
def camera_status():
    status = {}
    for cam_id, cap in app_state.CAMERAS.items():
        state_str = "error"
        try:
            if cap and hasattr(cap, "isOpened") and cap.isOpened():
                state_str = "ok"
        except Exception:
            state_str = "error"
        meta = app_state.CAMERA_METADATA.get(cam_id, {}) or {}
        status[cam_id] = {
            "state": state_str,
            "name": meta.get("name"),
            "geo": meta.get("geo"),
        }
    return {"status": status}

# --- Socket.IO event handlers (preserve your existing handlers) ---
# Improved connect handler: check origin header and allow only configured origins (or "*")
@sio.event
async def connect(sid, environ, auth=None):
    """
    Handle new client connection with tolerant origin checking.

    We normalize localhost <-> 127.0.0.1 (and strip trailing slash) so browser
    origin variations do not cause a 403 during the Socket.IO handshake.
    """
    # Try common header locations for origin
    origin = environ.get("HTTP_ORIGIN") or environ.get("origin") or ""
    # Normalize common host differences (localhost vs 127.0.0.1)
    orig_norm = origin.replace("http://localhost", "http://127.0.0.1").replace(
        "https://localhost", "https://127.0.0.1"
    ).rstrip("/")

    # Normalize the allow_origins configured earlier
    allowed_norm = [
        o.replace("http://localhost", "http://127.0.0.1")
         .replace("https://localhost", "https://127.0.0.1")
         .rstrip("/")
        for o in allow_origins
    ]

    # If allow_origins is not a wildcard and origin is present, enforce membership
    if allowed_norm and "*" not in allowed_norm and orig_norm:
        if orig_norm not in allowed_norm:
            logger.warning(
                "Socket connection rejected from origin=%s not in allow_origins=%s",
                origin,
                allow_origins,
            )
            # refuse connection; Socket.IO will return 403/connection refused
            raise ConnectionRefusedError("origin not allowed")

    # Accepted
    print(f"[Socket.IO] Client connected: {sid} origin={origin}")
    return True


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    print(f"[Socket.IO] Client disconnected: {sid}")

# --- Startup & shutdown lifecycle events ---
@api.on_event("startup")
async def startup_event():
    """
    Application startup logic:
      - Log startup summary
      - Attach shared SIO manager to app state for other modules
      - Initialize health checks background task (non-blocking)
      - Schedule periodic model cleanup to keep memory bounded on CPU machines
    """
    print("=" * 50)
    print("Multi-Camera Face Recognition Platform - Starting Up")
    print("=" * 50)
    print(f"Cameras initialized: {len([c for c in app_state.CAMERAS.values() if c is not None])} active / {len(app_state.CAMERAS)} total configured")
    print(f"In-memory faces loaded: {len(app_state.ENCODINGS)}")
    print("Backend ready at http://127.0.0.1:8000")
    print("Socket.IO server attached.")
    print("=" * 50)

    # Expose the SIO manager on the FastAPI app state for other modules to use
    try:
        api.state.sio_manager = SIO_MANAGER
    except Exception:
        # best-effort: if this fails, continue without attaching
        logger.exception("Failed to set api.state.sio_manager (continuing)")

    # Start health checks (registers background task internally)
    try:
        init_health_checks(api)
    except Exception:
        logger.exception("init_health_checks failed (health checks disabled)")

    # Start a periodic background task to cleanup idle models to limit memory usage.
    # This is conservative and helpful on CPU-only, low-RAM machines.
    async def _model_cleanup_loop():
        while True:
            try:
                await asyncio.sleep(MODEL_IDLE_UNLOAD_SECONDS)
                # call cleanup from threadpool if it uses blocking operations
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, model_manager.cleanup_idle_models, None)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Exception in model cleanup loop (continuing)")

    # schedule cleanup loop as a background task
    try:
        api.state._model_cleanup_task = asyncio.create_task(_model_cleanup_loop())
    except Exception:
        logger.exception("Failed to create model cleanup background task (continuing)")

@api.on_event("shutdown")
async def shutdown_event():
    """
    Graceful shutdown: release camera handles and cancel background tasks.
    """
    print("\nShutting down...")
    try:
        # Close cameras in a safe, additive manner
        try:
            app_state.close_all_cameras()
        except Exception:
            logger.exception("close_all_cameras failed (continuing)")

        # Cancel the model cleanup task if scheduled
        task = getattr(api.state, "_model_cleanup_task", None)
        if task:
            try:
                task.cancel()
                await task
            except Exception:
                logger.exception("Error cancelling model cleanup task")

    except Exception:
        logger.exception("Exception during shutdown (continuing)")
    finally:
        print("Shutdown complete")

# --- Wrap FastAPI app with Socket.IO ASGIApp (preserve previous behavior) ---
# After registering routers and events on the FastAPI instance, wrap it
# so the Socket.IO server handles socket endpoints while passing other
# requests to FastAPI.
app = socketio.ASGIApp(sio, api)
