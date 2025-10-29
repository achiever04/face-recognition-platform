"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! CRITICAL WARNING: FILE CONFLICT                                      !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This file (face_fl.py) is REDUNDANT and conflicts with:
    
    backend/app/routes/federated.py

Both files are trying to register the same API prefix: "/face/fl".

The other file (federated.py) is the correct, more complete version that
saves weights to disk and is used by your frontend.

This file is a simpler, non-functional placeholder and should NOT be
used.

To fix your application, this file (face_fl) MUST BE REMOVED from the
'__all__' list in:

    backend/app/routes/__init__.py

Leaving both will cause your application to crash or fail.
No new code will be added here to avoid making the conflict worse.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict

# Optional: import your db utils if you want to store FL weights
# from app.utils import db  # Uncomment if db.store_fl_weights or similar exists

router = APIRouter(prefix="/face/fl", tags=["Federated Learning"])

# -------------------------------
# Pydantic model for weight upload
# -------------------------------
class FLWeightsUpload(BaseModel):
    target: str
    weights: Dict[str, list]  # e.g., {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}


# -------------------------------
# GET /face/fl/status
# -------------------------------
@router.get("/status")
async def get_status(client_id: str = Query(..., alias="target")):
    """
    Endpoint to check federated learning weights/status.
    Accepts either ?client_id=client_1 or ?target=client_1 (alias for frontend compatibility).
    """
    # TODO: Replace with real stored weights retrieval if needed
    stored_weights = {}  # Replace with db.retrieve_fl_weights(client_id) if implemented
    return {"status": "success", "client_id": client_id, "weights": stored_weights}


# -------------------------------
# Alias route to handle old frontend calls like /face/fl/get_weights?target=client_1
# -------------------------------
@router.get("/get_weights")
async def get_weights(target: str = Query(...)):
    """
    Backward compatibility route.
    Maps ?target=client_1 to the same logic as /status.
    """
    return await get_status(client_id=target)


# -------------------------------
# POST /face/fl/upload_weights
# -------------------------------
@router.post("/upload_weights")
async def upload_weights(payload: FLWeightsUpload):
    """
    Endpoint to upload federated learning weights.
    Expects JSON body:
    {
        "target": "client_1",
        "weights": {"layer1": [..], "layer2": [..]}
    }
    """
    try:
        # Optional: store weights in DB (replace with actual logic if implemented)
        # db.store_fl_weights(payload.target, payload.weights)

        # For demo, just print/log
        print(f"[FL] Received weights for {payload.target}: {payload.weights}")

        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": f"Weights uploaded for {payload.target}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload weights: {str(e)}")