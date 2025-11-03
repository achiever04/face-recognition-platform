# backend/app/routes/camera.py

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse, Response
import face_recognition
import numpy as np
import cv2
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Tuple, Any
from pydantic import BaseModel, Field
import logging
import time
import os
import threading

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

# Ensure cameras are initialized
if not CAMERAS:
    try:
        init_cameras(list(CAMERA_METADATA.keys()))
    except Exception as e:
        logger.debug("init_cameras failed or already initialized: %s", e)

# -------------------------------
# NEW: Performance tracking & thread-safety
# -------------------------------
camera_performance: Dict[int, Dict[str, Any]] = {}  # Track FPS and processing times per camera
_performance_lock = threading.RLock()

# -------------------------------
# NEW: Pydantic Models for Request Validation
# -------------------------------
class CameraConfig(BaseModel):
    """Model for camera configuration"""
    camera_id: int = Field(..., ge=0, description="Unique camera ID")
    name: str = Field(..., min_length=1, max_length=255, description="Camera name")
    source: str = Field(..., description="Camera source (device ID, RTSP URL, or HTTP stream)")
    geo: Tuple[float, float] = Field(default=(0.0, 0.0), description="Geographic coordinates (lat, lon)")
    enabled: bool = Field(default=True, description="Whether camera is active")
    fps_limit: Optional[int] = Field(default=None, ge=1, le=60, description="Max FPS for processing")

class CameraUpdate(BaseModel):
    """Model for updating camera settings"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    geo: Optional[Tuple[float, float]] = None
    enabled: Optional[bool] = None
    fps_limit: Optional[int] = Field(None, ge=1, le=60)

class DetectionConfig(BaseModel):
    """Model for detection configuration"""
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Face matching threshold")
    frame_skip: int = Field(default=1, ge=1, le=30, description="Process every Nth frame")
    max_faces: int = Field(default=10, ge=1, le=100, description="Max faces to detect per frame")

# -------------------------------
# EXISTING: Frame generator (ENHANCED)
# -------------------------------
def gen_frames_from_cap(cap: cv2.VideoCapture):
    """Generator that yields MJPEG frames from a cv2.VideoCapture object."""
    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break
            # Encode frame to JPEG safely
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.debug("Failed to encode frame to JPEG; skipping frame")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        # Client disconnected
        logger.debug("Client disconnected from MJPEG stream")
    except Exception as e:
        logger.error("Error in frame generator: %s", e, exc_info=True)

# -------------------------------
# EXISTING: MJPEG Camera Feed (KEPT AS IS)
# -------------------------------
@router.get("/{camera_id}/feed")
def camera_feed(camera_id: int):
    """MJPEG stream for camera feed: /camera/{camera_id}/feed"""
    try:
        if camera_id not in CAMERAS:
            logger.warning("Camera feed requested for non-existent camera: %s", camera_id)
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        cap = CAMERAS[camera_id]

        if not cap.isOpened():
            logger.error("Camera %s is not opened", camera_id)
            raise HTTPException(status_code=503, detail=f"Camera {camera_id} is not available")

        logger.debug("Streaming feed for camera %s", camera_id)

        return StreamingResponse(
            gen_frames_from_cap(cap),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error streaming camera %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stream camera feed: {str(e)}")

# -------------------------------
# NEW: Camera Snapshot Endpoint
# -------------------------------
@router.get("/{camera_id}/snapshot")
async def camera_snapshot(camera_id: int, quality: int = Query(95, ge=1, le=100, description="JPEG quality")):
    """
    Capture a single snapshot from camera.
    Returns a JPEG image.
    """
    try:
        if camera_id not in CAMERAS:
            logger.warning("Snapshot requested for non-existent camera: %s", camera_id)
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        cap = CAMERAS[camera_id]
        if not cap.isOpened():
            logger.error("Camera %s is not opened for snapshot", camera_id)
            raise HTTPException(status_code=503, detail=f"Camera {camera_id} is not available")

        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Failed to capture frame from camera %s", camera_id)
            raise HTTPException(status_code=500, detail="Failed to capture frame")

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            logger.error("Failed to encode snapshot from camera %s", camera_id)
            raise HTTPException(status_code=500, detail="Failed to encode image")

        logger.info("Snapshot captured from camera %s with quality %s", camera_id, quality)
        filename = f"camera_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error capturing snapshot from camera %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to capture snapshot: {str(e)}")

# -------------------------------
# EXISTING: Process single camera (ENHANCED)
# -------------------------------
def process_camera_sync(cam_id: int, cap: cv2.VideoCapture, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start_time = time.time()
    try:
        # ✅ CRITICAL FIX: NULL SAFETY CHECK
        if cap is None:
            logger.warning("Camera %s has null capture object - skipping", cam_id)
            return results
        
        confidence_threshold = config.get("confidence_threshold", 0.6) if config else 0.6
        frame_skip = max(1, config.get("frame_skip", 1)) if config else 1
        max_faces = config.get("max_faces", 10) if config else 10

        # ✅ FIXED: Check isOpened only after null check
        if not cap.isOpened():
            logger.warning("Camera %s is not opened, attempting reconnect", cam_id)
            try:
                # Try to reopen using stored metadata source if available
                source = CAMERA_METADATA.get(cam_id, {}).get("source", cam_id)
                try:
                    source_conv = int(source)
                except Exception:
                    source_conv = source
                cap.open(source_conv)
                if not cap.isOpened():
                    logger.error("Failed to reconnect camera %s", cam_id)
                    return results
            except Exception as reconnect_error:
                logger.error("Reconnection failed for camera %s: %s", cam_id, reconnect_error)
                return results

        # Read a frame; if frame_skip > 1, skip appropriate frames by reading and discarding
        frame = None
        for i in range(frame_skip):
            ret, frame_candidate = cap.read()
            if not ret:
                frame = None
                break
            frame = frame_candidate

        if frame is None:
            logger.debug("No frame captured from camera %s", cam_id)
            return results

        # Optionally enforce camera fps limit
        fps_limit = CAMERA_METADATA.get(cam_id, {}).get("fps_limit")
        if fps_limit:
            elapsed = time.time() - start_time
            min_interval = 1.0 / float(fps_limit)
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        # Convert to RGB for face_recognition
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as cvt_err:
            logger.error("Failed to convert frame color for camera %s: %s", cam_id, cvt_err)
            return results

        # Detect faces with protections
        try:
            face_locations = face_recognition.face_locations(rgb)
            if len(face_locations) > max_faces:
                logger.warning("Camera %s: Detected %d faces, limiting to %d", cam_id, len(face_locations), max_faces)
                face_locations = face_locations[:max_faces]

            faces_enc = face_recognition.face_encodings(rgb, face_locations)
        except Exception as face_error:
            logger.error("Face detection error on camera %s: %s", cam_id, face_error, exc_info=True)
            return results

        # Compare against stored faces using face_service
        for face_encoding in faces_enc:
            try:
                matches = face_service.compare_faces(face_encoding, target_names=None, return_distances=True)
                for match in matches:
                    distance = match.get("distance", 0.0)
                    confidence = match.get("confidence", 1.0)
                    target = match.get("target")
                    if match.get("match") and distance <= confidence_threshold:
                        results.append({
                            "camera_id": cam_id,
                            "target": target,
                            "distance": distance,
                            "confidence": confidence
                        })
            except Exception as match_error:
                logger.error("Face matching error on camera %s: %s", cam_id, match_error, exc_info=True)
                continue

        # Update performance metrics
        processing_time = time.time() - start_time
        with _performance_lock:
            perf = camera_performance.setdefault(cam_id, {"total_frames": 0, "total_time": 0.0, "avg_fps": 0.0})
            perf["total_frames"] += 1
            perf["total_time"] += processing_time
            perf["avg_fps"] = (perf["total_frames"] / perf["total_time"]) if perf["total_time"] > 0 else 0.0

    except Exception as e:
        logger.error("Error processing camera %s: %s", cam_id, e, exc_info=True)

    return results

# -------------------------------
# EXISTING: Async wrapper (ENHANCED)
# -------------------------------
async def process_camera_async(cam_id: int, cap: cv2.VideoCapture, config: Optional[Dict[str, Any]] = None):
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
    """
    try:
        start_time = time.time()
        logger.debug("Starting camera alerts scan with threshold=%s, frame_skip=%s", confidence_threshold, frame_skip)

        detection_config = {
            "confidence_threshold": confidence_threshold,
            "frame_skip": frame_skip,
            "max_faces": max_faces
        }

        tasks = [process_camera_async(cam_id, cap, detection_config) for cam_id, cap in CAMERAS.items()]
        all_results = await asyncio.gather(*tasks)
        raw_detections: List[Dict[str, Any]] = [r for camera_results in all_results for r in camera_results]

        logger.info("Detected %d face matches across %d cameras", len(raw_detections), len(CAMERAS))

        for detection in raw_detections:
            cam_id = detection.get("camera_id")
            target = detection.get("target")
            distance = detection.get("distance")
            confidence = detection.get("confidence")

            camera_info = CAMERA_METADATA.get(cam_id, {})
            camera_name = camera_info.get("name", f"Camera {cam_id}")
            geo = camera_info.get("geo", (0.0, 0.0))

            try:
                tracking_result = tracking_service.record_detection(person_name=target, camera_id=cam_id, distance=distance, timestamp=datetime.now())
            except Exception as te:
                logger.error("Tracking service failed for detection %s: %s", detection, te)
                tracking_result = {"recorded": False, "is_new_location": False}

            if tracking_result.get("recorded"):
                try:
                    alert_service.generate_alert(
                        target_name=target,
                        camera_id=cam_id,
                        distance=distance,
                        timestamp=datetime.now(),
                        metadata={
                            "camera_name": camera_name,
                            "geo": geo,
                            "confidence": confidence,
                            "is_new_location": tracking_result.get("is_new_location", False)
                        }
                    )
                except Exception as ae:
                    logger.error("Alert generation failed for %s on camera %s: %s", target, cam_id, ae)

        # Get aggregated data
        try:
            alerts = alert_service.get_alerts(limit=50)
        except Exception:
            alerts = []
        try:
            movement_log = tracking_service.get_all_movements(limit_per_person=10)
        except Exception:
            movement_log = []
        try:
            current_locations = tracking_service.get_current_locations()
        except Exception:
            current_locations = {}

        latest_detection = alerts[0] if alerts else None
        grouped_alerts: Dict[str, Any] = {}
        for alert in alerts:
            tgt = alert.get("target")
            if tgt and tgt not in grouped_alerts:
                grouped_alerts[tgt] = alert

        processing_time = time.time() - start_time
        logger.info("Camera alerts scan completed in %.2fs", processing_time)

        return JSONResponse({
            "status": "success",
            "alerts": list(grouped_alerts.values()),
            "history": movement_log,
            "movement_log": [
                {
                    "target": person,
                    "camera_id": info.get("camera_id"),
                    "camera_name": info.get("camera_name"),
                    "geo": info.get("geo"),
                    "timestamp": info.get("last_seen")
                } for person, info in current_locations.items()
            ],
            "latest_detection": latest_detection,
            "metadata": {
                "processing_time": round(processing_time, 3),
                "cameras_scanned": len(CAMERAS),
                "detections": len(raw_detections),
                "config": detection_config
            }
        })

    except Exception as e:
        logger.error("Error in camera alerts: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to scan cameras: {str(e)}")

# -------------------------------
# EXISTING: Get tracking statistics (KEPT AS IS)
# -------------------------------
@router.get("/stats")
async def get_tracking_stats():
    try:
        logger.debug("Fetching tracking statistics")
        return JSONResponse({
            "status": "success",
            "tracking": tracking_service.get_statistics(),
            "alerts": alert_service.get_statistics()
        })
    except Exception as e:
        logger.error("Error fetching stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# -------------------------------
# EXISTING: Get movement history (KEPT AS IS)
# -------------------------------
@router.get("/movement/{person_name}")
async def get_person_movement(person_name: str, limit: int = 20):
    try:
        logger.debug("Fetching movement history for: %s", person_name)
        history = tracking_service.get_movement_history(person_name, limit=limit)
        path = tracking_service.get_movement_path(person_name, include_duplicates=False)
        return JSONResponse({
            "status": "success",
            "person": person_name,
            "history": history,
            "path": path
        })
    except Exception as e:
        logger.error("Error fetching movement for %s: %s", person_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get movement history: {str(e)}")

# -------------------------------
# EXISTING: Analyze patterns (KEPT AS IS)
# -------------------------------
@router.get("/analyze/{person_name}")
async def analyze_patterns(person_name: str):
    try:
        logger.debug("Analyzing patterns for: %s", person_name)
        analysis = tracking_service.detect_suspicious_patterns(person_name)
        return JSONResponse({"status": "success", "person": person_name, "analysis": analysis})
    except Exception as e:
        logger.error("Error analyzing patterns for %s: %s", person_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze patterns: {str(e)}")

# -------------------------------
# NEW: Get camera list with details
# -------------------------------
@router.get("/list")
async def list_cameras(include_performance: bool = Query(False, description="Include performance metrics")):
    try:
        logger.debug("Listing all cameras")
        camera_list = []
        for cam_id, cap in CAMERAS.items():
            metadata = CAMERA_METADATA.get(cam_id, {})
            is_opened = cap.isOpened() if hasattr(cap, "isOpened") else False
            camera_info = {
                "camera_id": cam_id,
                "name": metadata.get("name", f"Camera {cam_id}"),
                "geo": metadata.get("geo", (0.0, 0.0)),
                "status": "online" if is_opened else "offline",
                "enabled": metadata.get("enabled", True)
            }
            if include_performance:
                with _performance_lock:
                    perf = camera_performance.get(cam_id)
                    if perf:
                        camera_info["performance"] = {
                            "total_frames": perf.get("total_frames", 0),
                            "avg_fps": round(perf.get("avg_fps", 0.0), 2)
                        }
            camera_list.append(camera_info)
        return JSONResponse({"status": "success", "count": len(camera_list), "cameras": camera_list})
    except Exception as e:
        logger.error("Error listing cameras: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list cameras: {str(e)}")

# -------------------------------
# NEW: Get specific camera details
# -------------------------------
@router.get("/{camera_id}/info")
async def get_camera_info(camera_id: int):
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        logger.debug("Fetching info for camera %s", camera_id)
        cap = CAMERAS[camera_id]
        metadata = CAMERA_METADATA.get(camera_id, {})
        is_opened = cap.isOpened() if hasattr(cap, "isOpened") else False
        camera_info = {
            "camera_id": camera_id,
            "name": metadata.get("name", f"Camera {camera_id}"),
            "geo": metadata.get("geo", (0.0, 0.0)),
            "status": "online" if is_opened else "offline",
            "enabled": metadata.get("enabled", True)
        }
        if is_opened:
            camera_info["properties"] = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "backend": cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
            }
        with _performance_lock:
            perf = camera_performance.get(camera_id)
            if perf:
                camera_info["performance"] = {
                    "total_frames_processed": perf.get("total_frames", 0),
                    "avg_processing_fps": round(perf.get("avg_fps", 0.0), 2),
                    "total_processing_time": round(perf.get("total_time", 0.0), 2)
                }
        return JSONResponse({"status": "success", "camera": camera_info})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting camera info for %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get camera info: {str(e)}")

# -------------------------------
# NEW: Add new camera
# -------------------------------
@router.post("/add")
async def add_camera(camera: CameraConfig):
    try:
        logger.info("Adding new camera %s: %s", camera.camera_id, camera.name)
        if camera.camera_id in CAMERAS:
            raise HTTPException(status_code=400, detail=f"Camera {camera.camera_id} already exists")
        try:
            try:
                source = int(camera.source)
            except Exception:
                source = camera.source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                # fallback: try with different backend flags? keep simple: fail
                raise Exception(f"Failed to open camera source: {camera.source}")
            CAMERAS[camera.camera_id] = cap
            CAMERA_METADATA[camera.camera_id] = {
                "name": camera.name,
                "geo": camera.geo,
                "enabled": camera.enabled,
                "fps_limit": camera.fps_limit,
                "source": camera.source,
                "added_at": datetime.now().isoformat()
            }
            logger.info("Successfully added camera %s", camera.camera_id)
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
            logger.error("Failed to open camera %s: %s", camera.camera_id, cam_error)
            raise HTTPException(status_code=500, detail=f"Failed to open camera: {str(cam_error)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error adding camera: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add camera: {str(e)}")

# -------------------------------
# NEW: Update camera settings
# -------------------------------
@router.patch("/{camera_id}")
async def update_camera(camera_id: int, updates: CameraUpdate):
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        logger.info("Updating camera %s", camera_id)
        metadata = CAMERA_METADATA.get(camera_id, {})
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
        logger.info("Successfully updated camera %s", camera_id)
        return JSONResponse({"status": "success", "message": f"Camera {camera_id} updated successfully", "camera": {"camera_id": camera_id, **metadata}})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating camera %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update camera: {str(e)}")

# -------------------------------
# NEW: Remove camera
# -------------------------------
@router.delete("/{camera_id}")
async def remove_camera(camera_id: int):
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        logger.info("Removing camera %s", camera_id)
        try:
            cap = CAMERAS[camera_id]
            try:
                cap.release()
            except Exception as release_error:
                logger.warning("Error releasing camera %s: %s", camera_id, release_error)
        except Exception:
            pass
        CAMERAS.pop(camera_id, None)
        CAMERA_METADATA.pop(camera_id, None)
        with _performance_lock:
            camera_performance.pop(camera_id, None)
        logger.info("Successfully removed camera %s", camera_id)
        return JSONResponse({"status": "success", "message": f"Camera {camera_id} removed successfully", "camera_id": camera_id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error removing camera %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to remove camera: {str(e)}")

# -------------------------------
# NEW: Restart camera
# -------------------------------
@router.post("/{camera_id}/restart")
async def restart_camera(camera_id: int):
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        logger.info("Restarting camera %s", camera_id)
        cap = CAMERAS[camera_id]
        metadata = CAMERA_METADATA.get(camera_id, {})
        try:
            cap.release()
        except Exception as release_error:
            logger.warning("Error releasing camera %s: %s", camera_id, release_error)
        source = metadata.get("source", camera_id)
        try:
            source_conv = int(source)
        except Exception:
            source_conv = source
        new_cap = cv2.VideoCapture(source_conv)
        if not new_cap.isOpened():
            logger.error("Failed to restart camera %s", camera_id)
            raise HTTPException(status_code=500, detail=f"Failed to restart camera {camera_id}")
        CAMERAS[camera_id] = new_cap
        logger.info("Successfully restarted camera %s", camera_id)
        return JSONResponse({"status": "success", "message": f"Camera {camera_id} restarted successfully", "camera_id": camera_id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error restarting camera %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restart camera: {str(e)}")

# -------------------------------
# NEW: Get camera performance metrics
# -------------------------------
@router.get("/{camera_id}/performance")
async def get_camera_performance(camera_id: int):
    try:
        if camera_id not in CAMERAS:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        logger.debug("Fetching performance metrics for camera %s", camera_id)
        with _performance_lock:
            perf = camera_performance.get(camera_id)
            if not perf:
                return JSONResponse({"status": "success", "camera_id": camera_id, "message": "No performance data available yet", "performance": {"total_frames": 0, "avg_fps": 0, "total_time": 0}})
            return JSONResponse({
                "status": "success",
                "camera_id": camera_id,
                "performance": {
                    "total_frames_processed": perf.get("total_frames", 0),
                    "total_processing_time_seconds": round(perf.get("total_time", 0.0), 2),
                    "average_fps": round(perf.get("avg_fps", 0.0), 2),
                    "last_updated": datetime.now().isoformat()
                }
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching performance for camera %s: %s", camera_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# -------------------------------
# NEW: Get all camera performance metrics
# -------------------------------
@router.get("/performance/all")
async def get_all_performance():
    try:
        logger.debug("Fetching performance metrics for all cameras")
        performance_data = {}
        with _performance_lock:
            for cam_id, perf in camera_performance.items():
                performance_data[cam_id] = {
                    "camera_id": cam_id,
                    "name": CAMERA_METADATA.get(cam_id, {}).get("name", f"Camera {cam_id}"),
                    "total_frames": perf.get("total_frames", 0),
                    "avg_fps": round(perf.get("avg_fps", 0.0), 2),
                    "total_time": round(perf.get("total_time", 0.0), 2)
                }
        return JSONResponse({"status": "success", "count": len(performance_data), "performance": performance_data, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logger.error("Error fetching all performance metrics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# -------------------------------
# NEW: Health check for all cameras
# -------------------------------
@router.get("/health")
async def camera_health_check():
    try:
        logger.debug("Performing health check on all cameras")
        health_status = {"timestamp": datetime.now().isoformat(), "total_cameras": len(CAMERAS), "cameras": []}
        online_count = offline_count = 0
        for cam_id, cap in CAMERAS.items():
            metadata = CAMERA_METADATA.get(cam_id, {})
            is_opened = cap.isOpened() if hasattr(cap, "isOpened") else False
            status = "online" if is_opened else "offline"
            if is_opened:
                online_count += 1
            else:
                offline_count += 1
            camera_health = {"camera_id": cam_id, "name": metadata.get("name", f"Camera {cam_id}"), "status": status, "enabled": metadata.get("enabled", True)}
            if is_opened:
                try:
                    ret, _ = cap.read()
                    camera_health["readable"] = bool(ret)
                    if not ret:
                        camera_health["status"] = "degraded"
                        camera_health["issue"] = "Cannot read frames"
                except Exception as read_error:
                    camera_health["readable"] = False
                    camera_health["status"] = "error"
                    camera_health["issue"] = str(read_error)
            health_status["cameras"].append(camera_health)
        health_status["summary"] = {"online": online_count, "offline": offline_count, "health_percentage": round((online_count / len(CAMERAS) * 100), 2) if CAMERAS else 0}
        if offline_count == 0:
            health_status["overall_status"] = "healthy"
        elif offline_count < len(CAMERAS) / 2:
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "critical"
        logger.info("Health check complete: %d online, %d offline", online_count, offline_count)
        return JSONResponse(health_status)
    except Exception as e:
        logger.error("Error during health check: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# -------------------------------
# NEW: Bulk camera operations
# -------------------------------
@router.post("/bulk/restart")
async def restart_all_cameras():
    try:
        logger.info("Restarting all cameras")
        results = {"success": [], "failed": [], "timestamp": datetime.now().isoformat()}
        for cam_id, cap in list(CAMERAS.items()):
            try:
                metadata = CAMERA_METADATA.get(cam_id, {})
                try:
                    cap.release()
                except Exception:
                    pass
                source = metadata.get("source", cam_id)
                try:
                    source_conv = int(source)
                except Exception:
                    source_conv = source
                new_cap = cv2.VideoCapture(source_conv)
                if new_cap.isOpened():
                    CAMERAS[cam_id] = new_cap
                    results["success"].append(cam_id)
                    logger.info("Camera %s restarted successfully", cam_id)
                else:
                    results["failed"].append({"camera_id": cam_id, "reason": "Failed to open camera"})
                    logger.error("Failed to restart camera %s", cam_id)
            except Exception as cam_error:
                results["failed"].append({"camera_id": cam_id, "reason": str(cam_error)})
                logger.error("Error restarting camera %s: %s", cam_id, cam_error)
        return JSONResponse({"status": "completed", "message": f"Restarted {len(results['success'])} cameras, {len(results['failed'])} failed", "results": results})
    except Exception as e:
        logger.error("Error in bulk restart: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bulk restart failed: {str(e)}")

# -------------------------------
# NEW: Clear performance metrics
# -------------------------------
@router.delete("/performance/clear")
async def clear_performance_metrics():
    try:
        logger.info("Clearing all performance metrics")
        with _performance_lock:
            cleared_cameras = list(camera_performance.keys())
            camera_performance.clear()
        return JSONResponse({"status": "success", "message": f"Cleared performance metrics for {len(cleared_cameras)} cameras", "cleared_cameras": cleared_cameras})
    except Exception as e:
        logger.error("Error clearing performance metrics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear metrics: {str(e)}")

# -------------------------------
# NEW: Test camera connection
# -------------------------------
@router.post("/test")
async def test_camera_connection(source: str = Body(..., embed=True)):
    try:
        logger.info("Testing camera connection: %s", source)
        try:
            test_source = int(source)
        except Exception:
            test_source = source
        test_cap = cv2.VideoCapture(test_source)
        if not test_cap.isOpened():
            logger.warning("Failed to open camera source: %s", source)
            return JSONResponse({"status": "failed", "source": source, "message": "Failed to open camera source", "accessible": False})
        ret, frame = test_cap.read()
        result = {"status": "success", "source": source, "accessible": True, "can_read_frames": bool(ret)}
        if ret:
            result["properties"] = {"width": int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": test_cap.get(cv2.CAP_PROP_FPS), "backend": test_cap.getBackendName() if hasattr(test_cap, "getBackendName") else "unknown"}
        try:
            test_cap.release()
        except Exception:
            pass
        logger.info("Camera test successful for source: %s", source)
        return JSONResponse(result)
    except Exception as e:
        logger.error("Error testing camera connection: %s", e, exc_info=True)
        return JSONResponse({"status": "error", "source": source, "message": str(e), "accessible": False})