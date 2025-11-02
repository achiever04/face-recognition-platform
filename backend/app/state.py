# backend/app/state.py
"""
Shared runtime state for the application.

This module centralizes small amounts of global runtime state (camera handles,
encodings cache, federated-learning placeholders, and model references), while
providing safer, higher-level helpers:

  * ModelManager - lazy-loading/unloading + optional idle cleanup
  * Camera helpers - init_cameras, get_camera, release_camera, close_all_cameras
  * Lightweight diagnostics & memory heuristics

Design constraints / goals:
  - Backwards-compatible: all previously-exported symbols (ENCODINGS, CAMERAS, ...)
    remain available so existing code keeps working.
  - Lazy model imports: heavy libraries (torch, retinaface, etc.) are imported
    only when their loader runs (reduces memory & import time on CPU laptops).
  - Non-blocking and safe: get_model() is async-friendly; loaders can be sync
    or async functions.
  - Optional psutil usage for accurate memory checks; fallback to resource on Linux.
"""

from typing import Dict, Any, Optional, Callable, Coroutine
import os
import time
import threading
import asyncio
import logging

# Optional imports (guarded)
try:
    import psutil  # optional, for better memory checks
except Exception:
    psutil = None

# Avoid importing heavy ML libraries at module import time
# (they will be imported lazily inside loader functions)

# -------------------------------
# Socket.IO Asynchronous Server Manager
# -------------------------------
import socketio  # type: ignore

# Keep SIO_MANAGER compatible with current code that expects this symbol.
# Note: this is an AsyncManager instance; other parts of the app may create
# an AsyncServer/ASGI wrapper around it as needed.
SIO_MANAGER = socketio.AsyncManager()  # intentionally minimal and compatible

# Helper to attempt emitting an event via SIO_MANAGER. This is additive only.
async def emit_event(event: str, data: dict, namespace: Optional[str] = None):
    """
    Safe helper to emit Socket.IO events. If SIO_MANAGER has an 'emit' coroutine
    (e.g., if it's wrapped by an AsyncServer/ASGI app), this will call it.
    This helper swallows exceptions to avoid crashing business logic.
    """
    try:
        # If SIO_MANAGER has an 'emit' coroutine method (server), use it
        emit_coro = getattr(SIO_MANAGER, "emit", None)
        if asyncio.iscoroutinefunction(emit_coro):
            await emit_coro(event, data, namespace=namespace)
        else:
            # AsyncManager alone may not expose emit as coroutine; attempt safe call
            # In typical setups, the app wraps AsyncManager with AsyncServer which
            # provides an emit coroutine. So this is a no-op fallback.
            return
    except Exception:
        logging.getLogger("app.state").exception("emit_event failed (ignored)")

# -------------------------------
# Upload directory
# -------------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")

# -------------------------------
# Encoded face storage: filename -> encoding list (JSON serializable)
# Embeddings may be encrypted for privacy-preserving recognition
# -------------------------------
ENCODINGS: Dict[str, list] = {}

# -------------------------------
# Camera handles: camera_id -> cv2.VideoCapture (or None if failed)
# Use int or string keys (camera IDs may be numbers or strings)
# -------------------------------
import cv2  # local import - opencv should be installed for camera support

CAMERAS: Dict[Any, Optional[cv2.VideoCapture]] = {}
# Use a lock for thread-safe camera access in multi-threaded contexts
_CAMERAS_LOCK = threading.RLock()

# -------------------------------
# Camera metadata: camera_id -> {name, geo, status}
# -------------------------------
CAMERA_METADATA: Dict[Any, dict] = {}

# -------------------------------
# Track person locations: face_name -> camera_id
# -------------------------------
PERSON_LOCATIONS: Dict[str, Any] = {}

# -------------------------------
# Federated Learning structures
# -------------------------------
FL_WEIGHTS: Dict[str, dict] = {}  # client_id -> {param_name: np.array}
FL_MODEL_VERSIONS: Dict[str, int] = {}  # client_id -> version

# -------------------------------
# DeepFake + Face Detection Models (placeholders)
# Keep the old symbols for backward compatibility; the ModelManager
# will populate these when the corresponding model is loaded.
# -------------------------------
DEEPFAKE_MODEL = None
RETINAFACE_DETECTOR = None

# -------------------------------
# Configuration knobs (environment-driven)
# -------------------------------
MODEL_CPU_MODE = os.getenv("MODEL_CPU_MODE", "true").lower() in ("1", "true", "yes")
try:
    MODEL_MAX_MEMORY_MB = int(os.getenv("MODEL_MAX_MEMORY_MB", "6400"))
except Exception:
    MODEL_MAX_MEMORY_MB = 6400
try:
    MODEL_IDLE_UNLOAD_SECONDS = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "600"))
except Exception:
    MODEL_IDLE_UNLOAD_SECONDS = 600

# -------------------------------
# Logging
# -------------------------------
logger = logging.getLogger("app.state")

# -------------------------------
# ModelManager: lazy loader + unload manager for heavy models
# -------------------------------
class ModelManager:
    """
    Manages lazy-loading and unloading of heavy models.

    Usage:
        model_manager.register_loader("deepfake", loader_fn)
        model = await model_manager.get_model("deepfake")

    loader_fn can be a sync function or an async coroutine. It must return
    the loaded model instance.

    The manager keeps track of last_used timestamps and can unload models
    that were idle for MODEL_IDLE_UNLOAD_SECONDS seconds via cleanup_idle_models().
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self._models: Dict[str, Any] = {}
        self._last_used: Dict[str, float] = {}

    def register_loader(self, name: str, loader: Callable[[], Any]):
        """
        Register a loader for model `name`. Loader should be a callable that
        returns the model instance (can be a coroutine function).
        """
        with self._lock:
            self._loaders[name] = loader
            logger.debug("Registered model loader: %s", name)

    async def get_model(self, name: str):
        """
        Return the model instance for `name`, loading it lazily if needed.
        Accepts sync or async loader functions.
        """
        # Fast path if already loaded
        with self._lock:
            model = self._models.get(name)
            if model is not None:
                self._last_used[name] = time.time()
                return model

            loader = self._loaders.get(name)
            if loader is None:
                raise KeyError(f"No loader registered for model '{name}'")

        # Call loader outside lock to avoid long blocking under lock
        loaded = None
        try:
            if asyncio.iscoroutinefunction(loader):
                loaded = await loader()
            else:
                # run sync loader in threadpool if it may block
                loop = asyncio.get_running_loop()
                loaded = await loop.run_in_executor(None, loader)
        except Exception:
            logger.exception("Exception while loading model %s", name)
            raise

        with self._lock:
            self._models[name] = loaded
            self._last_used[name] = time.time()
            # populate well-known globals for backwards compatibility if applicable
            global DEEPFAKE_MODEL, RETINAFACE_DETECTOR
            if name == "deepfake":
                DEEPFAKE_MODEL = loaded
            elif name == "retinaface":
                RETINAFACE_DETECTOR = loaded
            logger.info("Model '%s' loaded (lazy)", name)
        return loaded

    def unload_model(self, name: str):
        """
        Unload (remove reference to) model `name`. This will let GC reclaim memory
        provided there are no other references.
        """
        with self._lock:
            if name in self._models:
                try:
                    # If the model has a close() / cpu() / cuda() method, attempt graceful release
                    m = self._models[name]
                    if hasattr(m, "close"):
                        try:
                            m.close()
                        except Exception:
                            pass
                    # Attempt to call .cpu() if present to move tensors off GPU (safe no-op on CPU)
                    if hasattr(m, "cpu"):
                        try:
                            m.cpu()
                        except Exception:
                            pass
                finally:
                    del self._models[name]
                    self._last_used.pop(name, None)
                    logger.info("Model '%s' unloaded (explicit)", name)

    def list_loaded_models(self):
        with self._lock:
            return list(self._models.keys())

    def model_status(self, name: str) -> Dict[str, Any]:
        with self._lock:
            return {
                "loaded": name in self._models,
                "last_used": self._last_used.get(name),
            }

    def cleanup_idle_models(self, idle_seconds: Optional[int] = None):
        """
        Unload models which have been idle for longer than idle_seconds (defaults to env).
        This method can be scheduled by the application (e.g., in a background task).
        """
        if idle_seconds is None:
            idle_seconds = MODEL_IDLE_UNLOAD_SECONDS
        cutoff = time.time() - idle_seconds
        to_unload = []
        with self._lock:
            for name, last in list(self._last_used.items()):
                if last < cutoff:
                    to_unload.append(name)
        for name in to_unload:
            try:
                self.unload_model(name)
            except Exception:
                logger.exception("Failed to cleanup model %s", name)

# Instantiate a global manager
model_manager = ModelManager()

# -------------------------------
# Built-in (conservative) loader registrations
# These loaders import heavy libraries lazily so that normal process startup
# on a small laptop is light-weight. The loaders are conservative and provide a
# CPU-only default where possible.
# -------------------------------
def _load_deepfake_model_sync():
    """
    Example loader for a deepfake/liveness model. This function imports heavy
    libraries only when executed. Replace with your real model load logic.
    """
    # Local imports to avoid import-time cost
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    # Placeholder: return a simple callable or object; replace with actual model load
    class DummyDeepfakeModel:
        def predict(self, image):
            # Always return not-deepfake with low confidence as default placeholder
            return {"spoof": False, "score": 0.01}

    # In real code, load a trained model here (CPU mode by default)
    return DummyDeepfakeModel()

async def _load_retinaface_async():
    """
    Example async loader for RetinaFace or similar detector. Replace with your
    real detector initialization (maybe using a small CPU model).
    """
    # Local imports
    try:
        # attempt to import a retinaface implementation, else fall back
        # to a tiny MTCNN or even dlib fallback
        from retinaface import RetinaFace  # type: ignore
        # Construct a small detector if available
        detector = RetinaFace
        return detector
    except Exception:
        # Fallback placeholder detector
        class DummyDetector:
            def detect_faces(self, image):
                return []  # no detections
        return DummyDetector()

# Register these conservative loaders
model_manager.register_loader("deepfake", _load_deepfake_model_sync)
model_manager.register_loader("retinaface", _load_retinaface_async)

# -------------------------------
# Utilities: memory checks (optional psutil)
# -------------------------------
def memory_info() -> Dict[str, Any]:
    """
    Return a small dict with system memory info. Uses psutil if present,
    otherwise falls back to resource.getrusage on Linux.
    """
    try:
        if psutil:
            mem = psutil.virtual_memory()
            return {
                "total_mb": int(mem.total / 1024 / 1024),
                "available_mb": int(mem.available / 1024 / 1024),
                "used_mb": int(mem.used / 1024 / 1024),
            }
        else:
            import resource
            import os
            # Best-effort: return current process rss in MB if available
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss = getattr(usage, "ru_maxrss", 0)
            # On Linux ru_maxrss is in kilobytes, convert:
            if rss > 0:
                rss_mb = int(rss / 1024)
            else:
                rss_mb = 0
            return {"process_rss_mb": rss_mb}
    except Exception:
        return {"error": "memory info unavailable"}

# -------------------------------
# Camera management helpers
# -------------------------------
def init_cameras(cam_configs=None, open_timeout: float = 2.0):
"""
Initialize multiple cv2.VideoCapture instances with metadata.
FIXED: Better error handling, prevents storing None in CAMERAS dict
"""
if cam_configs is None:
    cam_configs = [{"id": 0, "name": "Default Cam", "geo": (0.0, 0.0)}]

for cam in cam_configs:
    cam_id = cam["id"]
    
    with _CAMERAS_LOCK:
        # Skip if already initialized and working
        if cam_id in CAMERAS:
            existing_cap = CAMERAS.get(cam_id)
            if existing_cap is not None and existing_cap.isOpened():
                logger.info("Camera %s already initialized and working", cam_id)
                continue
    
    try:
        # ✅ FIX: Try to open camera
        cap = cv2.VideoCapture(cam_id)
        time.sleep(0.1)  # Small delay for camera to initialize
        
        if not cap.isOpened():
            logger.warning("Camera %s (%s) not available - will NOT store in CAMERAS", cam_id, cam.get("name"))
            
            # ✅ CRITICAL: Release failed capture
            try:
                cap.release()
            except Exception:
                pass
            
            # ✅ FIX: Store metadata but mark as unavailable (don't store None cap)
            with _CAMERAS_LOCK:
                # Don't add to CAMERAS dict if it failed to open
                CAMERA_METADATA[cam_id] = {
                    "name": cam.get("name"),
                    "geo": cam.get("geo"),
                    "status": "unavailable",
                    "source": cam_id,
                    "error": "Failed to open camera"
                }
        else:
            # ✅ SUCCESS: Camera opened properly
            with _CAMERAS_LOCK:
                CAMERAS[cam_id] = cap
                CAMERA_METADATA[cam_id] = {
                    "name": cam.get("name"),
                    "geo": cam.get("geo"),
                    "status": "ok",
                    "source": cam_id
                }
            logger.info("Successfully initialized Camera %s (%s)", cam_id, cam.get("name"))
            
    except Exception as e:
        logger.exception("Error initializing camera %s: %s", cam_id, e)
        with _CAMERAS_LOCK:
            # Don't add to CAMERAS on exception
            CAMERA_METADATA[cam_id] = {
                "name": cam.get("name"),
                "geo": cam.get("geo"),
                "status": "error",
                "source": cam_id,
                "error": str(e)
            }

def get_camera(cam_id):
    """
    Return the VideoCapture instance for cam_id or None if unavailable.
    """
    with _CAMERAS_LOCK:
        return CAMERAS.get(cam_id)

def release_camera(cam_id):
    """
    Release a camera VideoCapture if present and set its entry to None.
    """
    with _CAMERAS_LOCK:
        cap = CAMERAS.get(cam_id)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.exception("Error releasing camera %s", cam_id)
        CAMERAS[cam_id] = None
        CAMERA_METADATA[cam_id] = CAMERA_METADATA.get(cam_id, {})
        CAMERA_METADATA[cam_id]["status"] = "released"

def close_all_cameras():
    """
    Close all open camera handles; used at shutdown.
    """
    with _CAMERAS_LOCK:
        for k, cap in list(CAMERAS.items()):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    logger.exception("Exception releasing camera %s", k)
            CAMERAS[k] = None
            CAMERA_METADATA[k] = CAMERA_METADATA.get(k, {})
            CAMERA_METADATA[k]["status"] = "released"

# -------------------------------
# Deprecated placeholders (kept for compatibility)
# -------------------------------
import numpy as np  # some helpers below use numpy

def encrypt_embedding(embedding: np.ndarray) -> list:
    """
    --- DEPRECATED ---
    Placeholder: real encryption lives in app/utils/db.py (cryptography).
    Left in place for backward compatibility with older code that may call it.
    """
    try:
        return (-embedding).tolist()
    except Exception:
        return list(embedding.tolist())

def decrypt_embedding(enc_embedding: list) -> np.ndarray:
    """
    --- DEPRECATED ---
    Placeholder.
    """
    try:
        return -np.array(enc_embedding)
    except Exception:
        return np.array(enc_embedding)

def save_fl_weights(client_id: str):
    """
    --- DEPRECATED ---
    Saves FL weights from FL_WEIGHTS dict to disk (JSON). Kept as a convenience
    helper but routes/federated.py should be the canonical implementation.
    """
    import json
    import os

    FL_DIR = os.getenv("FL_DIR", "data/fl_weights")
    os.makedirs(FL_DIR, exist_ok=True)
    if client_id in FL_WEIGHTS:
        weights_json = {k: v.tolist() for k, v in FL_WEIGHTS[client_id].items()}
        with open(os.path.join(FL_DIR, f"{client_id}.json"), "w", encoding="utf-8") as f:
            json.dump(weights_json, f)

def load_fl_weights(client_id: str):
    """
    --- DEPRECATED ---
    Loads FL weights from disk (if present) into FL_WEIGHTS.
    """
    import json
    import os

    FL_DIR = os.getenv("FL_DIR", "data/fl_weights")
    path = os.path.join(FL_DIR, f"{client_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            weights_dict = json.load(f)
        # convert lists back to numpy arrays
        FL_WEIGHTS[client_id] = {k: np.array(v) for k, v in weights_dict.items()}

# -------------------------------
# Diagnostics helpers
# -------------------------------
def model_manager_status() -> Dict[str, Any]:
    """
    Return a compact dict describing loaded models and memory info.
    """
    try:
        return {
            "loaded_models": model_manager.list_loaded_models(),
            "memory": memory_info(),
        }
    except Exception:
        return {"error": "status unavailable"}

# -------------------------------
# End of module
# -------------------------------
