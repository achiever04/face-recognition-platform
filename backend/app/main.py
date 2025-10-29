# backend/app/main.py

# --- IMPORTS ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import socketio # Import the main library
# backend/app/main.py (insert near other imports)
from app.routes import snapshot as snapshot_router  # new file we added
from app.health_checks import init_health_checks


# --- State and Routers ---
# Import the state module itself to access variables within functions
from app import state as app_state
# Import specific functions needed at the top level
from app.state import init_cameras, UPLOAD_DIR, SIO_MANAGER
# Import routers
from app.routes import camera, federated, deepfake, face, alerts

# --- .env and Logging ---
from dotenv import load_dotenv
from app.utils.logger import setup_logger # Import the setup function
load_dotenv() # Load .env variables
# setup_logger() # Initialize logging - UNCOMMENT THIS LINE if you want file/console logging

app = FastAPI(title="Multi-Camera Face Recognition Platform")

# --- Create the actual Socket.IO Async Server ---
sio = socketio.AsyncServer(async_mode="asgi", client_manager=SIO_MANAGER)

# --- CORS Middleware (Handles HTTP requests) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Ensure upload directory exists ---
# Access UPLOAD_DIR through the imported state module
os.makedirs(app_state.UPLOAD_DIR, exist_ok=True)

# --- Initialize cameras ---
camera_configs = [
    {"id": 0, "name": "Shivaji Nagar Chauk 1", "geo": (18.555, 73.808)},
    {"id": 1, "name": "Pune Station", "geo": (18.528, 73.847)},
    {"id": 2, "name": "FC Road Signal", "geo": (18.516, 73.841)},
    {"id": 3, "name": "Kothrud Square", "geo": (18.504, 73.823)},
    {"id": 4, "name": "Swargate Bus Stop", "geo": (18.501, 73.862)},
]
# Call the init_cameras function (which modifies state.CAMERAS and state.CAMERA_METADATA)
init_cameras(camera_configs)

# --- Include FastAPI routers ---
app.include_router(camera.router, prefix="/camera")
app.include_router(federated.router)
app.include_router(deepfake.router)
app.include_router(alerts.router)
app.include_router(face.router) # Correctly included

# --- Health Check ---
@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running"}

# --- Camera Status ---
@app.get("/camera/status")
def camera_status():
    status = {}
    # --- FIX: Access CAMERAS and CAMERA_METADATA via app_state ---
    for cam_id, cap in app_state.CAMERAS.items():
        status_str = "error"
        # Check if cap object exists and is opened
        if cap and cap.isOpened():
            status_str = "ok"
        status[cam_id] = {
            "state": status_str,
            "name": app_state.CAMERA_METADATA.get(cam_id, {}).get("name"),
            "geo": app_state.CAMERA_METADATA.get(cam_id, {}).get("geo"),
        }
    return {"status": status}

# --- REMOVED: Conflicting /face/upload and /face/compare endpoints ---

# --- Define Socket.IO event handlers on the 'sio' server instance ---
@sio.event
async def connect(sid, environ):
    """Handle new client connection."""
    print(f"[Socket.IO] Client connected: {sid}")

@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    print(f"[Socket.IO] Client disconnected: {sid}")

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("Multi-Camera Face Recognition Platform - Starting Up")
    print("=" * 50)
    # --- FIX: Access CAMERAS and ENCODINGS via app_state ---
    print(f"Cameras initialized: {len([c for c in app_state.CAMERAS.values() if c is not None])} active / {len(app_state.CAMERAS)} total configured")
    print(f"In-memory faces loaded: {len(app_state.ENCODINGS)}") # Access via app_state.ENCODINGS
    print("Backend ready at http://127.0.0.1:8000")
    print("Socket.IO server attached.")
    print("=" * 50)

# --- Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    print("\nShutting down...")
    # --- FIX: Access CAMERAS via app_state ---
    for cam_id, cap in app_state.CAMERAS.items():
        if cap: # Only try to release if it's not None
            try:
                cap.release()
                print(f"Released camera {cam_id}")
            except Exception as e:
                print(f"Error releasing camera {cam_id}: {e}")
    print("Shutdown complete")

# --- Wrap FastAPI app with Socket.IO ASGIApp ---
# Pass the 'sio' server instance here
app = socketio.ASGIApp(sio, app)