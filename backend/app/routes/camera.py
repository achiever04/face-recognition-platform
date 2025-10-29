# backend/app/routes/camera.py

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse, Response
import face_recognition
import numpy as np
import cv2
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import logging
import time

from app.state import ENCODINGS, CAMERAS, CAMERA_METADATA, init_cameras
from app.utils.db import retrieve_embedding
from app.services.face_service import face_service
from app.services.tracking_service import tracking_service
from app.services.alert_service import alert_service

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Thread pool for CPU-bound operations (EXISTING - kept as is)
executor = ThreadPoolExecutor(max_workers=4)

# Initialize cameras if not done (EXISTING - kept as is)
if not CAMERAS:
    init_cameras(list(CAMERA_METADATA.keys()))

# -------------------------------
# NEW: Performance tracking
# -------------------------------
camera_performance = {}  # Track FPS and processing times per camera

# -------------------------------
# NEW: Pydantic Models for Request Validation
# -------------------------------
class CameraConfig(BaseModel):
    """Model for camera configuration"""
    camera_id: int = Field(..., ge=0, description="Unique camera ID")
    name: str = Field(..., min_length=1, max_length=255, description="Camera name")
    source: str = Field(..., description="Camera source (device ID, RTSP URL, or HTTP stream)")
    geo: tuple = Field(default=(0.0, 0.0), description="Geographic coordinates (lat, lon)")
    enabled: bool = Field(default=True, description="Whether camera is active")
    fps_limit: Optional[int] = Field(default=None, ge=1, le=60, description="Max FPS for processing")

class CameraUpdate(BaseModel):
    """Model for updating camera settings"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    geo: Optional[tuple] = None
    enabled: Optional[bool] = None
    fps_limit: Optional[int] = Field(None, ge=1, le=60)

class DetectionConfig(BaseModel):
    """Model for detection configuration"""
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Face matching threshold")
    frame_skip: int = Field(default=1, ge=1, le=30, description="Process every Nth frame")
    max_faces: int = Field(default=10, ge=1, le=100, description="Max faces to detect per frame")

# -------------------------------
# EXISTING: Frame generator (KEPT AS IS)
# -------------------------------
def gen_frames_from_cap(cap: cv2.VideoCapture):
    """Generator that yields MJPEG frames from a cv2.VideoCapture object."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------------------------------
# EXISTING: MJPEG Camera Feed (KEPT AS IS)
# -------------------------------
@router.get("/{camera_id}/feed")
def camera_feed(camera_id: int):
    """MJPEG stream for camera feed: /camera/{camera_id}/feed"""
    try:
        if camera_id not in CAMERAS:
            logger.warning(f"Camera feed requested for non-existent camera: {camera_id}")
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        cap = CAMERAS[camera_id]
        
        # Check if camera is accessible
        if not cap.isOpened():
            logger.error(f"Camera {camera_id} is not opened")
            raise HTTPException(status_code=503, detail=f"Camera {camera_id} is not available")
        
        logger.debug(f"Streaming feed for camera {camera_id}")
        
        return StreamingResponse(
            gen_frames_from_cap(cap),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming camera {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stream camera feed: {str(e)}")

# -------------------------------
# NEW: Camera Snapshot Endpoint
# -------------------------------
@router.get("/{camera_id}/snapshot")
async def camera_snapshot(camera_id: int, quality: int = Query(95, ge=1, le=100, description="JPEG quality")):
    """
    Capture a single snapshot from camera.
    
    **NEW ENDPOINT**
    
    Returns a JPEG image.
    """
    try:
        if camera_id not in CAMERAS:
            logger.warning(f"Snapshot requested for non-existent camera: {camera_id}")
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        cap = CAMERAS[camera_id]
        
        if not cap.isOpened():
            logger.error(f"Camera {camera_id} is not opened for snapshot")
            raise HTTPException(status_code=503, detail=f"Camera {camera_id} is not available")
        
        # Capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            logger.error(f"Failed to capture frame from camera {camera_id}")
            raise HTTPException(status_code=500, detail="Failed to capture frame")
        
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        logger.info(f"Snapshot captured from camera {camera_id} with quality {quality}")
        
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=camera_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error capturing snapshot from camera {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to capture snapshot: {str(e)}")

# -------------------------------
# EXISTING: Process single camera (ENHANCED)
# -------------------------------
def process_camera_sync(cam_id: int, cap: cv2.VideoCapture, config: Dict = None):
    """
    Synchronous camera processing (runs in thread pool)
    
    ENHANCED: Added configuration support, better error handling, performance tracking
    """
    results = []
    start_time = time.time()
    
    try:
        # Get configuration (NEW)
        confidence_threshold = config.get("confidence_threshold", 0.6) if config else 0.6
        max_faces = config.get("max_faces", 10) if config else 10
        
        # Check camera health (NEW)
        if not cap.isOpened():
            logger.warning(f"Camera {cam_id} is not opened, attempting to reconnect")
            # Attempt reconnection (NEW)
            try:
                cap.open(cam_id)
                if not cap.isOpened():
                    logger.error(f"Failed to reconnect camera {cam_id}")
                    return results
            except Exception as reconnect_error:
                logger.error(f"Reconnection failed for camera {cam_id}: {reconnect_error}")
                return results
        
        # Capture frame (EXISTING)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.debug(f"No frame captured from camera {cam_id}")
            return results
        
        # Convert to RGB (EXISTING)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces (EXISTING but with limit - NEW)
        try:
            face_locations = face_recognition.face_locations(rgb)
            
            # Limit number of faces (NEW)
            if len(face_locations) > max_faces:
                logger.warning(f"Camera {cam_id}: Detected {len(face_locations)} faces, limiting to {max_faces}")
                face_locations = face_locations[:max_faces]
            
            faces_enc = face_recognition.face_encodings(rgb, face_locations)
        except Exception as face_error:
            logger.error(f"Face detection error on camera {cam_id}: {face_error}")
            return results
        
        # Compare against all stored faces (EXISTING)
        for face_encoding in faces_enc:
            try:
                # Use face_service for comparison (EXISTING)
                matches = face_service.compare_faces(
                    face_encoding,
                    target_names=None,  # Check all targets
                    return_distances=True
                )
                
                # Process matches with threshold (ENHANCED)
                for match in matches:
                    if match["match"] and match["distance"] <= confidence_threshold:  # NEW: threshold check
                        results.append({
                            "camera_id": cam_id,
                            "target": match["target"],
                            "distance": match["distance"],
                            "confidence": match["confidence"]
                        })
            except Exception as match_error:
                logger.error(f"Face matching error on camera {cam_id}: {match_error}")
                continue
        
        # Track performance (NEW)
        processing_time = time.time() - start_time
        if cam_id not in camera_performance:
            camera_performance[cam_id] = {
                "total_frames": 0,
                "total_time": 0,
                "avg_fps": 0
            }
        
        camera_performance[cam_id]["total_frames"] += 1
        camera_performance[cam_id]["total_time"] += processing_time
        camera_performance[cam_id]["avg_fps"] = (
            camera_performance[cam_id]["total_frames"] / 
            camera_performance[cam_id]["total_time"]
        )
        
    except Exception as e:
        logger.error(f"Error processing camera {cam_id}: {e}", exc_info=True)
    
    return results

# -------------------------------
# EXISTING: Async wrapper (ENHANCED)
# -------------------------------
async def process_camera_async(cam_id: int, cap: cv2.VideoCapture, config: Dict = None):
    """
    Process camera in thread pool
    
    ENHANCED: Added configuration passthrough
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, process_camera_sync, cam_id, cap, config)

# -------------------------------
# EXISTING: Real-Time Alerts & Movement Tracking (ENHANCED)
# -------------------------------
@router.get("/alerts")
async def camera_alerts(
    confidence_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Face matching threshold"),
    frame_skip: int = Query(1, ge=1, le=30, description="Process every Nth frame"),
    max_faces: int = Query(10, ge=1, le=100, description="Max faces per frame")
):
    """
    Scan all initialized cameras and return matches against ENCODINGS.
    Uses services for tracking, alerts, and face recognition.
    
    ENHANCED: Added configurable detection parameters
    """
    try:
        start_time = time.time()
        
        logger.debug(f"Starting camera alerts scan with threshold={confidence_threshold}, frame_skip={frame_skip}")
        
        # Prepare detection config (NEW)
        detection_config = {
            "confidence_threshold": confidence_threshold,
            "frame_skip": frame_skip,
            "max_faces": max_faces
        }
        
        # Process all cameras in parallel (EXISTING but with config - ENHANCED)
        tasks = [
            process_camera_async(cam_id, cap, detection_config)
            for cam_id, cap in CAMERAS.items()
        ]
        
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results (EXISTING)
        raw_detections = []
        for camera_results in all_results:
            raw_detections.extend(camera_results)
        
        logger.info(f"Detected {len(raw_detections)} face matches across {len(CAMERAS)} cameras")
        
        # Process each detection (EXISTING)
        for detection in raw_detections:
            cam_id = detection["camera_id"]
            target = detection["target"]
            distance = detection["distance"]
            confidence = detection["confidence"]
            
            # Get camera metadata (EXISTING)
            camera_info = CAMERA_METADATA.get(cam_id, {})
            camera_name = camera_info.get("name", f"Camera {cam_id}")
            geo = camera_info.get("geo", (0.0, 0.0))
            
            # Record in tracking service (EXISTING)
            tracking_result = tracking_service.record_detection(
                person_name=target,
                camera_id=cam_id,
                distance=distance,
                timestamp=datetime.now()
            )
            
            # Generate alert if not duplicate (EXISTING)
            if tracking_result["recorded"]:
                alert_service.generate_alert(
                    target_name=target,
                    camera_id=cam_id,
                    distance=distance,
                    timestamp=datetime.now(),
                    metadata={
                        "camera_name": camera_name,
                        "geo": geo,
                        "confidence": confidence,
                        "is_new_location": tracking_result["is_new_location"]
                    }
                )
        
        # Get aggregated data from services (EXISTING)
        alerts = alert_service.get_alerts(limit=50)
        movement_log = tracking_service.get_all_movements(limit_per_person=10)
        current_locations = tracking_service.get_current_locations()
        
        # Format for frontend (EXISTING)
        latest_detection = None
        if alerts:
            latest_detection = alerts[0]  # Most recent alert
        
        # Group alerts by person (EXISTING)
        grouped_alerts = {}
        for alert in alerts:
            target = alert["target"]
            if target not in grouped_alerts:
                grouped_alerts[target] = alert
        
        processing_time = time.time() - start_time
        
        logger.info(f"Camera alerts scan completed in {processing_time:.2f}s")
        
        return JSONResponse({
            "status": "success",
            "alerts": list(grouped_alerts.values()),
            "history": movement_log,
            "movement_log": [
                {
                    "target": person,
                    "camera_id": info["camera_id"],
                    "camera_name": info["camera_name"],
                    "geo": info["geo"],
                    "timestamp": info["last_seen"]
                }
                for person, info in current_locations.items()
            ],
            "latest_detection": latest_detection,
            "metadata": {  # NEW: Performance metadata
                "processing_time": round(processing_time, 3),
                "cameras_scanned": len(CAMERAS),
                "detections": len(raw_detections),
                "config": detection_config
            }
        })
    
    except Exception as e:
        logger.error(f"Error in camera alerts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to scan cameras: {str(e)}")

# -------------------------------
# EXISTING: Get tracking statistics (KEPT AS IS)
# -------------------------------
@router.get("/stats")
async def get_tracking_stats():
    """Get tracking and alert statistics"""
    try:
        logger.debug("Fetching tracking statistics")
        
        return JSONResponse({
            "status": "success",
            "tracking": tracking_service.get_statistics(),
            "alerts": alert_service.get_statistics()
        })
    
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# -------------------------------
# EXISTING: Get movement history (KEPT AS IS)
# -------------------------------
@router.get("/movement/{person_name}")
async def get_person_movement(person_name: str, limit: int = 20):
    """Get movement history for a specific person"""
    try:
        logger.debug(f"Fetching movement history for: {person_name}")
        
        history = tracking_service.get_movement_history(person_name, limit=limit)
        path = tracking_service.get_movement_path(person_name, include_duplicates=False)
        
        return JSONResponse({
            "status": "success",
            "person": person_name,
            "history": history,
            "path": path
        })
    
    except Exception as e:
        logger.error(f"Error fetching movement for {person_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get movement history: {str(e)}")

# -------------------------------
# EXISTING: Analyze patterns (KEPT AS IS)
# -------------------------------
@router.get("/analyze/{person_name}")
async def analyze_patterns(person_name: str):
    """Analyze movement patterns for suspicious behavior"""
    try:
        logger.debug(f"Analyzing patterns for: {person_name}")
        
        analysis = tracking_service.detect_suspicious_patterns(person_name)
        
        return JSONResponse({
            "status": "success",
            "person": person_name,
            "analysis": analysis
        })
    
    except Exception as e:
        logger.error(f"Error analyzing patterns for {person_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze patterns: {str(e)}")

# -------------------------------
# NEW: Get camera list with details
# -------------------------------
@router.get("/list")
async def list_cameras(include_performance: bool = Query(False, description="Include performance metrics")):
    """
    Get list of all cameras with their details.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug("Listing all cameras")
        
        camera_list = []
        
        for cam_id, cap in CAMERAS.items():
            metadata = CAMERA_METADATA.get(cam_id, {})
            
            # Check camera status
            is_opened = cap.isOpened()
            
            camera_info = {
                "camera_id": cam_id,
                "name": metadata.get("name", f"Camera {cam_id}"),
                "geo": metadata.get("geo", (0.0, 0.0)),
                "status": "online" if is_opened else "offline",
                "enabled": metadata.get("enabled", True)
            }
            
            # Add performance metrics if requested
            if include_performance and cam_id in camera_performance:
                perf = camera_performance[cam_id]
                camera_info["performance"] = {
                    "total_frames": perf["total_frames"],
                    "avg_fps": round(perf["avg_fps"], 2)
                }
            
            camera_list.append(camera_info)
        
        return JSONResponse({
            "status": "success",
            "count": len(camera_list),
            "cameras": camera_list
        })
    
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list cameras: {str(e)}")

# -------------------------------
# NEW: Get specific camera details
# -------------------------------
@router.get("/{camera_id}/info")
async def get_camera_info(camera_id: int):
    """
    Get detailed information about a specific camera.
    
    **NEW ENDPOINT**
    """
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        logger.debug(f"Fetching info for camera {camera_id}")
        
        cap = CAMERAS[camera_id]
        metadata = CAMERA_METADATA.get(camera_id, {})
        
        # Get camera properties
        is_opened = cap.isOpened()
        
        camera_info = {
            "camera_id": camera_id,
            "name": metadata.get("name", f"Camera {camera_id}"),
            "geo": metadata.get("geo", (0.0, 0.0)),
            "status": "online" if is_opened else "offline",
            "enabled": metadata.get("enabled", True)
        }
        
        # Add technical details if camera is open
        if is_opened:
            camera_info["properties"] = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "backend": cap.getBackendName()
            }
        
        # Add performance metrics
        if camera_id in camera_performance:
            perf = camera_performance[camera_id]
            camera_info["performance"] = {
                "total_frames_processed": perf["total_frames"],
                "avg_processing_fps": round(perf["avg_fps"], 2),
                "total_processing_time": round(perf["total_time"], 2)
            }
        
        return JSONResponse({
            "status": "success",
            "camera": camera_info
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera info for {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get camera info: {str(e)}")

# -------------------------------
# NEW: Add new camera
# -------------------------------
@router.post("/add")
async def add_camera(camera: CameraConfig):
    """
    Add a new camera to the system.
    
    **NEW ENDPOINT**
    
    Body:
```json
    {
        "camera_id": 5,
        "name": "Main Entrance",
        "source": "rtsp://192.168.1.100:554/stream",
        "geo": [18.555, 73.808],
        "enabled": true,
        "fps_limit": 15
    }
```
    """
    try:
        logger.info(f"Adding new camera {camera.camera_id}: {camera.name}")
        
        # Check if camera already exists
        if camera.camera_id in CAMERAS:
            raise HTTPException(
                status_code=400, 
                detail=f"Camera {camera.camera_id} already exists"
            )
        
        # Try to open camera
        try:
            # Convert source to int if it's a number (device ID)
            try:
                source = int(camera.source)
            except ValueError:
                source = camera.source  # Keep as string (URL)
            
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise Exception(f"Failed to open camera source: {camera.source}")
            
            # Add to global dictionaries
            CAMERAS[camera.camera_id] = cap
            CAMERA_METADATA[camera.camera_id] = {
                "name": camera.name,
                "geo": camera.geo,
                "enabled": camera.enabled,
                "fps_limit": camera.fps_limit,
                "source": camera.source,
                "added_at": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully added camera {camera.camera_id}")
            
            return JSONResponse({
                "status": "success",
                "message": f"Camera {camera.camera_id} added successfully",
                "camera": {
                    "camera_id": camera.camera_id,
                    "name": camera.name,
                    "source": camera.source,
                    "geo": camera.geo,
                    "enabled": camera.enabled
                }
            })
        
        except Exception as cam_error:
            logger.error(f"Failed to open camera {camera.camera_id}: {cam_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to open camera: {str(cam_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding camera: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add camera: {str(e)}")

# -------------------------------
# NEW: Update camera settings
# -------------------------------
@router.patch("/{camera_id}")
async def update_camera(camera_id: int, updates: CameraUpdate):
    """
    Update camera settings.
    
    **NEW ENDPOINT**
    
    Body:
```json
    {
        "name": "Updated Name",
        "geo": [18.555, 73.808],
        "enabled": true,
        "fps_limit": 10
    }
```
    """
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        logger.info(f"Updating camera {camera_id}")
        
        metadata = CAMERA_METADATA.get(camera_id, {})
        
        # Update fields
        if updates.name is not None:
            metadata["name"] = updates.name
        if updates.geo is not None:
            metadata["geo"] = updates.geo
        if updates.enabled is not None:
            metadata["enabled"] = updates.enabled
        if updates.fps_limit is not None:
            metadata["fps_limit"] = updates.fps_limit
        
        metadata["updated_at"] = datetime.now().isoformat()
        
        CAMERA_METADATA[camera_id] = metadata
        
        logger.info(f"Successfully updated camera {camera_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Camera {camera_id} updated successfully",
            "camera": {
                "camera_id": camera_id,
                **metadata
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating camera {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update camera: {str(e)}")

# -------------------------------
# NEW: Remove camera
# -------------------------------
@router.delete("/{camera_id}")
async def remove_camera(camera_id: int):
    """
    Remove a camera from the system.
    
    **NEW ENDPOINT**
    """
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        logger.info(f"Removing camera {camera_id}")
        
        # Release camera resource
        try:
            cap = CAMERAS[camera_id]
            cap.release()
        except Exception as release_error:
            logger.warning(f"Error releasing camera {camera_id}: {release_error}")
        
        # Remove from dictionaries
        del CAMERAS[camera_id]
        if camera_id in CAMERA_METADATA:
            del CAMERA_METADATA[camera_id]
        if camera_id in camera_performance:
            del camera_performance[camera_id]
        
        logger.info(f"Successfully removed camera {camera_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Camera {camera_id} removed successfully",
            "camera_id": camera_id
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing camera {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to remove camera: {str(e)}")

# -------------------------------
# NEW: Restart camera
# -------------------------------
@router.post("/{camera_id}/restart")
async def restart_camera(camera_id: int):
    """
    Restart a camera connection.
    
    **NEW ENDPOINT**
    
    Useful when camera becomes unresponsive.
    """
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        logger.info(f"Restarting camera {camera_id}")
        
        cap = CAMERAS[camera_id]
        metadata = CAMERA_METADATA.get(camera_id, {})
        
        # Release current connection
        try:
            cap.release()
        except Exception as release_error:
            logger.warning(f"Error releasing camera {camera_id}: {release_error}")
        
        # Reopen camera
        source = metadata.get("source", camera_id)
        try:
            source = int(source)
        except ValueError:
            pass  # Keep as string
        
        new_cap = cv2.VideoCapture(source)
        
        if not new_cap.isOpened():
            logger.error(f"Failed to restart camera {camera_id}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to restart camera {camera_id}"
            )
        
        # Replace in global dict
        CAMERAS[camera_id] = new_cap
        
        logger.info(f"Successfully restarted camera {camera_id}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Camera {camera_id} restarted successfully",
            "camera_id": camera_id
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting camera {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restart camera: {str(e)}")

# -------------------------------
# NEW: Get camera performance metrics
# -------------------------------
@router.get("/{camera_id}/performance")
async def get_camera_performance(camera_id: int):
    """
    Get performance metrics for a specific camera.
    
    **NEW ENDPOINT**
    """
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        logger.debug(f"Fetching performance metrics for camera {camera_id}")
        
        if camera_id not in camera_performance:
            return JSONResponse({
                "status": "success",
                "camera_id": camera_id,
                "message": "No performance data available yet",
                "performance": {
                    "total_frames": 0,
                    "avg_fps": 0,
                    "total_time": 0
                }
            })
        
        perf = camera_performance[camera_id]
        
        return JSONResponse({
            "status": "success",
            "camera_id": camera_id,
            "performance": {
                "total_frames_processed": perf["total_frames"],
                "total_processing_time_seconds": round(perf["total_time"], 2),
                "average_fps": round(perf["avg_fps"], 2),
                "last_updated": datetime.now().isoformat()
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching performance for camera {camera_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# -------------------------------
# NEW: Get all camera performance metrics
# -------------------------------
@router.get("/performance/all")
async def get_all_performance():
    """
    Get performance metrics for all cameras.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug("Fetching performance metrics for all cameras")
        
        performance_data = {}
        
        for cam_id in CAMERAS.keys():
            if cam_id in camera_performance:
                perf = camera_performance[cam_id]
                performance_data[cam_id] = {
                    "camera_id": cam_id,
                    "name": CAMERA_METADATA.get(cam_id, {}).get("name", f"Camera {cam_id}"),
                    "total_frames": perf["total_frames"],
                    "avg_fps": round(perf["avg_fps"], 2),
                    "total_time": round(perf["total_time"], 2)
                }
        
        return JSONResponse({
            "status": "success",
            "count": len(performance_data),
            "performance": performance_data,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching all performance metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# -------------------------------
# NEW: Health check for all cameras
# -------------------------------
@router.get("/health")
async def camera_health_check():
    """
    Check health status of all cameras.
    
    **NEW ENDPOINT**
    
    Returns online/offline status for each camera.
    """
    try:
        logger.debug("Performing health check on all cameras")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "total_cameras": len(CAMERAS),
            "cameras": []
        }
        
        online_count = 0
        offline_count = 0
        
        for cam_id, cap in CAMERAS.items():
            metadata = CAMERA_METADATA.get(cam_id, {})
            is_opened = cap.isOpened()
            
            status = "online" if is_opened else "offline"
            
            if is_opened:
                online_count += 1
            else:
                offline_count += 1
            
            camera_health = {
                "camera_id": cam_id,
                "name": metadata.get("name", f"Camera {cam_id}"),
                "status": status,
                "enabled": metadata.get("enabled", True)
            }
            
            # Try to read a frame to verify camera is actually working
            if is_opened:
                try:
                    ret, _ = cap.read()
                    camera_health["readable"] = ret
                    if not ret:
                        camera_health["status"] = "degraded"
                        camera_health["issue"] = "Cannot read frames"
                except Exception as read_error:
                    camera_health["readable"] = False
                    camera_health["status"] = "error"
                    camera_health["issue"] = str(read_error)
            
            health_status["cameras"].append(camera_health)
        
        health_status["summary"] = {
            "online": online_count,
            "offline": offline_count,
            "health_percentage": round((online_count / len(CAMERAS)) * 100, 2) if CAMERAS else 0
        }
        
        # Determine overall status
        if offline_count == 0:
            health_status["overall_status"] = "healthy"
        elif offline_count < len(CAMERAS) / 2:
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "critical"
        
        logger.info(f"Health check complete: {online_count} online, {offline_count} offline")
        
        return JSONResponse(health_status)
    
    except Exception as e:
        logger.error(f"Error during health check: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# -------------------------------
# NEW: Bulk camera operations
# -------------------------------
@router.post("/bulk/restart")
async def restart_all_cameras():
    """
    Restart all cameras.
    
    **NEW ENDPOINT**
    
    Useful for system-wide camera refresh.
    """
    try:
        logger.info("Restarting all cameras")
        
        results = {
            "success": [],
            "failed": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for cam_id, cap in list(CAMERAS.items()):
            try:
                metadata = CAMERA_METADATA.get(cam_id, {})
                
                # Release current connection
                cap.release()
                
                # Reopen
                source = metadata.get("source", cam_id)
                try:
                    source = int(source)
                except ValueError:
                    pass
                
                new_cap = cv2.VideoCapture(source)
                
                if new_cap.isOpened():
                    CAMERAS[cam_id] = new_cap
                    results["success"].append(cam_id)
                    logger.info(f"Camera {cam_id} restarted successfully")
                else:
                    results["failed"].append({
                        "camera_id": cam_id,
                        "reason": "Failed to open camera"
                    })
                    logger.error(f"Failed to restart camera {cam_id}")
            
            except Exception as cam_error:
                results["failed"].append({
                    "camera_id": cam_id,
                    "reason": str(cam_error)
                })
                logger.error(f"Error restarting camera {cam_id}: {cam_error}")
        
        return JSONResponse({
            "status": "completed",
            "message": f"Restarted {len(results['success'])} cameras, {len(results['failed'])} failed",
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in bulk restart: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bulk restart failed: {str(e)}")

# -------------------------------
# NEW: Clear performance metrics
# -------------------------------
@router.delete("/performance/clear")
async def clear_performance_metrics():
    """
    Clear all performance metrics.
    
    **NEW ENDPOINT**
    
    Resets performance tracking counters.
    """
    try:
        logger.info("Clearing all performance metrics")
        
        cleared_cameras = list(camera_performance.keys())
        camera_performance.clear()
        
        return JSONResponse({
            "status": "success",
            "message": f"Cleared performance metrics for {len(cleared_cameras)} cameras",
            "cleared_cameras": cleared_cameras
        })
    
    except Exception as e:
        logger.error(f"Error clearing performance metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear metrics: {str(e)}")

# -------------------------------
# NEW: Test camera connection
# -------------------------------
@router.post("/test")
async def test_camera_connection(source: str = Body(..., embed=True)):
    """
    Test a camera connection without adding it to the system.
    
    **NEW ENDPOINT**
    
    Body:
```json
    {
        "source": "rtsp://192.168.1.100:554/stream"
    }
```
    
    or
```json
    {
        "source": "0"
    }
```
    """
    try:
        logger.info(f"Testing camera connection: {source}")
        
        # Try to convert to int (device ID)
        try:
            test_source = int(source)
        except ValueError:
            test_source = source
        
        # Try to open camera
        test_cap = cv2.VideoCapture(test_source)
        
        if not test_cap.isOpened():
            logger.warning(f"Failed to open camera source: {source}")
            return JSONResponse({
                "status": "failed",
                "source": source,
                "message": "Failed to open camera source",
                "accessible": False
            })
        
        # Try to read a frame
        ret, frame = test_cap.read()
        
        result = {
            "status": "success",
            "source": source,
            "accessible": True,
            "can_read_frames": ret
        }
        
        # Get camera properties if successful
        if ret:
            result["properties"] = {
                "width": int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": test_cap.get(cv2.CAP_PROP_FPS),
                "backend": test_cap.getBackendName()
            }
        
        # Release test camera
        test_cap.release()
        
        logger.info(f"Camera test successful for source: {source}")
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Error testing camera connection: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "source": source,
            "message": str(e),
            "accessible": False
        })