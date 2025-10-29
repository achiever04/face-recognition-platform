"""
Shared runtime state for the application.
Includes multiple camera support with metadata and federated learning support.
"""

import cv2
from typing import Dict, Any
import numpy as np
import socketio  # --- NEW: Import Socket.IO ---

# -------------------------------
# NEW: Socket.IO Asynchronous Server Manager
# -------------------------------
# This is the central object that manages all WebSocket connections
# and allows emitting events from anywhere in the app.
# --- FIX: Removed cors_allowed_origins argument ---
SIO_MANAGER = socketio.AsyncManager() # Remove cors_allowed_origins="*"

# -------------------------------
# Upload directory
# -------------------------------
UPLOAD_DIR = "data/uploads"

# -------------------------------
# Encoded face storage: filename -> encoding list (JSON serializable)
# Embeddings may be encrypted for privacy-preserving recognition
# -------------------------------
ENCODINGS: Dict[str, list] = {}

# -------------------------------
# Camera handles: camera_id -> cv2.VideoCapture
# -------------------------------
CAMERAS: Dict[int, cv2.VideoCapture] = {}

# -------------------------------
# Camera metadata: camera_id -> {name, geo}
# -------------------------------
CAMERA_METADATA: Dict[int, dict] = {}

# -------------------------------
# Track person locations: face_name -> camera_id
# -------------------------------
PERSON_LOCATIONS: Dict[str, int] = {}

# -------------------------------
# Federated Learning structures
# client_id -> model weights (dict of numpy arrays)
# -------------------------------
FL_WEIGHTS: Dict[str, dict] = {}  # stores per-client model updates

# Optional: local model version tracking
FL_MODEL_VERSIONS: Dict[str, int] = {}  # client_id -> version

# -------------------------------
# DeepFake + Face Detection Models (placeholders)
# -------------------------------
DEEPFAKE_MODEL = None        # Will hold MobileNetV3-based DeepFake detector
RETINAFACE_DETECTOR = None   # Will hold RetinaFace detector instance

# -------------------------------
# Initialize multiple cameras with metadata
# --- UPGRADED ---
# -------------------------------
def init_cameras(cam_configs=None):
    """
    Initialize multiple cv2.VideoCapture instances with metadata.

    IMPROVEMENT: Now checks if camera isOpened() to prevent errors
    from disconnected or unavailable cameras.
    """
    if cam_configs is None:
        cam_configs = [{"id": 0, "name": "Default Cam", "geo": (0.0, 0.0)}]

    for cam in cam_configs:
        cam_id = cam["id"]
        if cam_id not in CAMERAS:
            try:
                cap = cv2.VideoCapture(cam_id)
                # --- NEW: Check if camera is accessible ---
                if not cap.isOpened():
                    print(f"Warning: Camera {cam_id} ({cam['name']}) is not available or failed to open.")
                    cap.release()
                    CAMERAS[cam_id] = None # Store None to mark as failed
                    CAMERA_METADATA[cam_id] = {"name": cam["name"], "geo": cam["geo"], "status": "error"}
                else:
                    CAMERAS[cam_id] = cap
                    CAMERA_METADATA[cam_id] = {"name": cam["name"], "geo": cam["geo"], "status": "ok"}
                    print(f"Successfully initialized Camera {cam_id} ({cam['name']})")
            except Exception as e:
                print(f"Error initializing camera {cam_id}: {e}")
                CAMERAS[cam_id] = None
                CAMERA_METADATA[cam_id] = {"name": cam["name"], "geo": cam["geo"], "status": "error"}


# -------------------------------
# Utility: encrypt/decrypt embeddings (optional)
# --- DEPRECATED ---
# -------------------------------
def encrypt_embedding(embedding: np.ndarray) -> list:
    """
    --- DEPRECATED ---
    This is a non-functional placeholder.
    The REAL encryption logic is in 'app/utils/db.py'
    using the 'cryptography' library. This function is not used.
    """
    # For demonstration: convert to list and multiply by -1 (dummy encryption)
    return (-embedding).tolist()

def decrypt_embedding(enc_embedding: list) -> np.ndarray:
    """
    --- DEPRECATED ---
    This is a non-functional placeholder.
    The REAL decryption logic is in 'app/utils/db.py'
    using the 'cryptography' library. This function is not used.
    """
    return -np.array(enc_embedding)

# -------------------------------
# Utility: save/load FL weights to/from disk
# --- DEPRECATED ---
# -------------------------------
def save_fl_weights(client_id: str):
    """
    --- DEPRECATED ---
    This is a redundant placeholder.
    The REAL logic for saving weights is implemented directly in
    'app/routes/federated.py' inside the 'upload_fl_weights' endpoint.
    """
    import os, json
    FL_DIR = "data/fl_weights"
    os.makedirs(FL_DIR, exist_ok=True)
    if client_id in FL_WEIGHTS:
        weights_json = {k: v.tolist() for k, v in FL_WEIGHTS[client_id].items()}
        with open(os.path.join(FL_DIR, f"{client_id}.json"), "w", encoding="utf-8") as f:
            json.dump(weights_json, f)

def load_fl_weights(client_id: str):
    """
    --- DEPRECATED ---
    This is a redundant placeholder.
    The REAL logic for loading weights is implemented directly in
    'app/routes/federated.py' inside the 'get_fl_status' endpoint.
    """
    import os, json
    FL_DIR = "data/fl_weights"
    path = os.path.join(FL_DIR, f"{client_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            weights_dict = json.load(f)
        # convert lists back to numpy arrays
        FL_WEIGHTS[client_id] = {k: np.array(v) for k, v in weights_dict.items()}