# backend/app/routes/__init__.py

"""
API Routes Registration
Handles importing and exposing API routers to the main FastAPI application.
"""

# Import the router objects from your route modules
from app.routes import camera
from app.routes import face
from app.routes import deepfake
from app.routes import federated # Keep this one (the functional version)
from app.routes import alerts
# from app.routes import face_fl # --- CRITICAL FIX: REMOVED / COMMENTED OUT ---
                                # This conflicts with 'federated.py'

# The __all__ list tells Python what symbols to export when 'from app.routes import *' is used.
# More importantly, it serves as a clear declaration of which routers are active.
__all__ = [
    'camera',
    'face',       # Now active after fixing main.py
    'deepfake',
    'federated',  # The correct FL router
    'alerts'
    # 'face_fl'   # Ensure the redundant/conflicting one is NOT included
]

print("[Routes Init] Loaded routers:", __all__) # Add a print statement for confirmation