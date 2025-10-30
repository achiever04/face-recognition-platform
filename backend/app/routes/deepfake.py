# backend/app/routes/deepfake.py

import os
import uuid
import shutil
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import json
from io import BytesIO
from collections import deque
import tempfile
import threading
import hashlib

from app.utils.deepfake_utils import DeepfakeDetector
from app.utils.cctv_utils import CCTVProcessor

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deepfake", tags=["Deepfake Detection"])

# Initialize detector instance (EXISTING - kept as is)
detector = DeepfakeDetector()

# Temporary folder for uploads (EXISTING - kept as is)
UPLOAD_DIR = os.getenv("DEEPFAKE_UPLOAD_DIR", "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Thread-safety for shared structures
_lock = threading.RLock()

# -------------------------------
# NEW: Detection history and statistics
# -------------------------------
detection_history = deque(maxlen=1000)  # Keep last 1000 detections
detection_stats = {
    "total_videos_processed": 0,
    "total_frames_analyzed": 0,
    "total_fake_detected": 0,
    "total_real_detected": 0,
    "average_processing_time": 0.0,
    "last_updated": None
}

# -------------------------------
# NEW: Cache for recent detections (avoid reprocessing same file)
# -------------------------------
detection_cache: Dict[str, Dict[str, Any]] = {}  # file_hash -> {"result": ..., "timestamp": datetime}
CACHE_EXPIRY_MINUTES = 30

# -------------------------------
# NEW: Pydantic Models for Request Validation
# -------------------------------
class DeepfakeConfig(BaseModel):
    """Configuration for deepfake detection"""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold for fake classification")
    max_frames: int = Field(default=20, ge=1, le=100, description="Maximum frames to sample")
    sampling_strategy: str = Field(default="uniform", description="Frame sampling strategy (uniform/random/keyframes)")

class CCTVConfig(BaseModel):
    """Configuration for CCTV processing"""
    cameras: str = Field(..., description="Comma-separated camera sources")
    max_frames_per_camera: int = Field(default=10, ge=1, le=100, description="Frames to process per camera")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection threshold")

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    status: str
    weights_loaded: bool
    description: str

# -------------------------------
# Helper utilities
# -------------------------------
def _now_iso() -> str:
    return datetime.now().isoformat()

def _safe_basename(filename: str) -> str:
    base = os.path.basename(filename or "")
    safe = base.replace("..", "").replace("/", "_").replace("\\", "_")
    return safe or f"upload_{int(datetime.now().timestamp())}"

def _atomic_write(path: str, data: bytes):
    """Atomically write bytes to disk using tmp file + replace"""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=".tmp_write_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file for cache key"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error("Error calculating file hash: %s", e)
        # fallback to uuid
        return str(uuid.uuid4())

def clean_detection_cache():
    """Remove expired entries from detection cache"""
    with _lock:
        now = datetime.now()
        expired = [k for k, v in detection_cache.items() if (now - v["timestamp"]).total_seconds() > CACHE_EXPIRY_MINUTES * 60]
        for k in expired:
            del detection_cache[k]
        if expired:
            logger.debug("Cleaned %d expired cache entries", len(expired))

# -------------------------------
# EXISTING: Video frame prediction helper (ENHANCED)
# -------------------------------
def process_video(video_path: str, config: Optional[DeepfakeConfig] = None) -> Dict[str, Any]:
    """
    Process video frame by frame and return deepfake analysis.

    ENHANCED: Added configuration support, better sampling, metadata extraction
    """
    if config is None:
        config = DeepfakeConfig()

    start_time = datetime.now()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file")

    probabilities: List[float] = []
    frame_results: List[Dict[str, Any]] = []

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0

        logger.info("Processing video: %d frames, %.2f FPS, %.2fs duration", frame_count, fps, duration)

        if frame_count <= 0:
            # Fall back to sequential read up to max_frames
            sample_indices = None
            max_frames = config.max_frames
        else:
            max_frames = int(min(config.max_frames, max(1, frame_count)))
            if config.sampling_strategy == "uniform":
                step = max(1, frame_count // max_frames)
                frame_indices = list(range(0, frame_count, step))[:max_frames]
                sample_indices = set(frame_indices)
            elif config.sampling_strategy == "random":
                # If max_frames >= frame_count, choose all
                if max_frames >= frame_count:
                    frame_indices = list(range(frame_count))
                else:
                    frame_indices = sorted(np.random.choice(frame_count, max_frames, replace=False).tolist())
                sample_indices = set(frame_indices)
            else:
                # fallback to uniform
                step = max(1, frame_count // max_frames)
                frame_indices = list(range(0, frame_count, step))[:max_frames]
                sample_indices = set(frame_indices)

        frames_processed = 0
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count <= 0:
                # unknown frame_count: process until we reach max_frames
                if frames_processed >= config.max_frames:
                    break
            else:
                if idx not in sample_indices:
                    idx += 1
                    continue

            try:
                results = detector.detect_and_classify(frame)
                for r in results:
                    # Probability: for fake detection use confidence, for real use 1 - confidence
                    prob = float(r.get("confidence", 0.0)) if r.get("is_fake") else (1.0 - float(r.get("confidence", 0.0)))
                    probabilities.append(prob)

                    timestamp = (idx / fps) if fps > 0 else 0.0

                    frame_results.append({
                        "frame_number": idx,
                        "timestamp": timestamp,
                        "is_fake": bool(r.get("is_fake")),
                        "confidence": float(r.get("confidence", 0.0)),
                        "bbox": r.get("bbox")
                    })

                frames_processed += 1

            except Exception as frame_error:
                logger.warning("Error processing frame %d: %s", idx, frame_error)

            idx += 1

            # If frame_count unknown, stop once we've read enough frames
            if frame_count <= 0 and frames_processed >= config.max_frames:
                break

    finally:
        cap.release()

    processing_time = (datetime.now() - start_time).total_seconds()

    if probabilities:
        avg_prob = float(np.mean(probabilities))
        max_prob = float(np.max(probabilities))
        min_prob = float(np.min(probabilities))
        std_dev = float(np.std(probabilities))
    else:
        avg_prob = 0.0
        max_prob = 0.0
        min_prob = 0.0
        std_dev = 0.0

    label = "fake" if avg_prob >= config.confidence_threshold else "real"

    # Confidence level (NEW)
    if avg_prob >= 0.8 or avg_prob <= 0.2:
        confidence_level = "high"
    elif avg_prob >= 0.6 or avg_prob <= 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    result = {
        "label": label,
        "probability": avg_prob,
        "statistics": {
            "mean_probability": avg_prob,
            "max_probability": max_prob,
            "min_probability": min_prob,
            "std_deviation": std_dev,
            "confidence_level": confidence_level
        },
        "metadata": {
            "total_frames": frame_count,
            "frames_analyzed": len(frame_results),
            "sampling_strategy": config.sampling_strategy,
            "fps": fps,
            "duration_seconds": duration,
            "resolution": f"{width}x{height}",
            "processing_time_seconds": round(processing_time, 2)
        },
        "frame_results": frame_results[:10]
    }

    logger.info("Video analysis complete: %s (probability: %.3f)", label, avg_prob)
    return result

# -------------------------------
# EXISTING: Detect DeepFake from uploaded video (ENHANCED)
# -------------------------------
@router.post("/detect")
async def detect_deepfake(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for fake classification"),
    max_frames: int = Query(20, ge=1, le=100, description="Maximum frames to sample"),
    sampling_strategy: str = Query("uniform", description="Sampling strategy (uniform/random)"),
    use_cache: bool = Query(True, description="Use cached results if available")
):
    """
    Upload a video and check if it is a deepfake.
    Returns detailed probability analysis and frame-level results.
    """
    file_id = str(uuid.uuid4())
    safe_name = _safe_basename(file.filename)
    file_path = None

    try:
        if not file.content_type or not file.content_type.startswith("video/"):
            logger.warning("Invalid file type uploaded: %s", file.content_type)
            raise HTTPException(status_code=400, detail="Only video files are allowed.")

        logger.info("Processing deepfake detection for file: %s", safe_name)

        # Save uploaded file temporarily (streaming copy to avoid large memory usage)
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")
        try:
            # If UploadFile provides a file-like, use shutil.copyfileobj for streaming
            with open(file_path, "wb") as buffer:
                try:
                    # some UploadFile implementations allow .file to be read synchronously
                    shutil.copyfileobj(file.file, buffer)
                except Exception:
                    # Fallback: read bytes (may use memory) but keep robust
                    content = await file.read()
                    buffer.write(content)
        except Exception as e:
            logger.error("Failed to save uploaded file: %s", e)
            raise HTTPException(status_code=500, detail="Failed to save uploaded file for processing")

        # Hash and cache handling
        file_hash = calculate_file_hash(file_path)
        if use_cache:
            clean_detection_cache()
            with _lock:
                if file_hash in detection_cache:
                    cached = detection_cache[file_hash]
                    logger.info("Returning cached result for file: %s", safe_name)
                    return JSONResponse(
                        content={
                            "status": "success",
                            "filename": file.filename,
                            "cached": True,
                            **cached["result"]
                        }
                    )

        config = DeepfakeConfig(
            confidence_threshold=confidence_threshold,
            max_frames=max_frames,
            sampling_strategy=sampling_strategy
        )

        result = process_video(file_path, config)

        # Update statistics atomically
        with _lock:
            detection_stats["total_videos_processed"] += 1
            detection_stats["total_frames_analyzed"] += result["metadata"]["frames_analyzed"]
            if result["label"] == "fake":
                detection_stats["total_fake_detected"] += 1
            else:
                detection_stats["total_real_detected"] += 1

            # update average processing time
            total_processed = detection_stats["total_videos_processed"]
            current_avg = detection_stats.get("average_processing_time", 0.0)
            new_time = result["metadata"]["processing_time_seconds"]
            detection_stats["average_processing_time"] = ((current_avg * (total_processed - 1) + new_time) / total_processed) if total_processed > 0 else new_time
            detection_stats["last_updated"] = _now_iso()

            detection_history.append({
                "filename": file.filename,
                "file_id": file_id,
                "label": result["label"],
                "probability": result["probability"],
                "timestamp": _now_iso()
            })

            if use_cache:
                detection_cache[file_hash] = {"result": result, "timestamp": datetime.now()}

        logger.info("Deepfake detection complete for %s: %s", safe_name, result["label"])

        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "file_id": file_id,
                "cached": False,
                "deepfake_probability": result["probability"],
                "label": result["label"],
                "statistics": result["statistics"],
                "metadata": result["metadata"],
                "sample_frames": result["frame_results"]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing video: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Cleanup temporary file
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug("Cleaned up temporary file: %s", file_path)
        except Exception as cleanup_error:
            logger.error("Failed to remove temp file %s: %s", file_path, cleanup_error)

# -------------------------------
# EXISTING: Detect DeepFake from live CCTV feed (ENHANCED)
# -------------------------------
@router.get("/cctv")
async def detect_cctv(
    cameras: str = Query(..., description="Comma-separated camera sources"),
    max_frames_per_camera: int = Query(10, ge=1, le=100, description="Frames per camera"),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Detection threshold")
):
    camera_list = [c.strip() for c in cameras.split(",") if c.strip()]
    if not camera_list:
        logger.warning("CCTV detection called with no cameras")
        raise HTTPException(status_code=400, detail="No cameras provided")

    logger.info("Processing CCTV deepfake detection for %d cameras", len(camera_list))
    processor = CCTVProcessor(camera_list)
    summary: Dict[str, Any] = {}
    processing_errors: List[Dict[str, Any]] = []

    try:
        max_total_frames = len(camera_list) * max_frames_per_camera

        for cam_id, frame, detections in processor.run(max_frames=max_total_frames):
            try:
                real_faces = [d for d in detections if not d.get('is_fake')]
                fake_faces = [d for d in detections if d.get('is_fake')]

                high_conf_fake = [d for d in fake_faces if float(d.get('confidence', 0.0)) >= confidence_threshold]
                high_conf_real = [d for d in real_faces if float(d.get('confidence', 0.0)) >= confidence_threshold]

                camera_key = f"camera_{cam_id}"
                if camera_key not in summary:
                    summary[camera_key] = {"total_faces": 0, "real_faces": 0, "fake_faces": 0, "high_confidence_fake": 0, "high_confidence_real": 0, "frames_processed": 0}

                summary[camera_key]["total_faces"] += len(detections)
                summary[camera_key]["real_faces"] += len(real_faces)
                summary[camera_key]["fake_faces"] += len(fake_faces)
                summary[camera_key]["high_confidence_fake"] += len(high_conf_fake)
                summary[camera_key]["high_confidence_real"] += len(high_conf_real)
                summary[camera_key]["frames_processed"] += 1

            except Exception as frame_error:
                logger.error("Error processing frame from camera %s: %s", cam_id, frame_error)
                processing_errors.append({"camera_id": cam_id, "error": str(frame_error)})

    except Exception as e:
        logger.exception("Error processing CCTV: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing CCTV: {str(e)}")
    finally:
        try:
            processor.release()
        except Exception:
            logger.debug("Processor release raised an exception", exc_info=True)

    total_faces = sum(cam.get("total_faces", 0) for cam in summary.values())
    total_fake = sum(cam.get("fake_faces", 0) for cam in summary.values())
    total_real = sum(cam.get("real_faces", 0) for cam in summary.values())

    logger.info("CCTV processing complete: %d faces, %d fake, %d real", total_faces, total_fake, total_real)

    return JSONResponse(content={
        "status": "success",
        "cameras_processed": len(camera_list),
        "configuration": {"max_frames_per_camera": max_frames_per_camera, "confidence_threshold": confidence_threshold},
        "summary": summary,
        "aggregated_stats": {
            "total_faces_detected": total_faces,
            "total_fake_detected": total_fake,
            "total_real_detected": total_real,
            "fake_percentage": round((total_fake / total_faces * 100), 2) if total_faces > 0 else 0
        },
        "errors": processing_errors if processing_errors else None
    })

# -------------------------------
# NEW: Get detection history
# -------------------------------
@router.get("/history")
async def get_detection_history(limit: int = Query(50, ge=1, le=1000, description="Maximum number of records"), label_filter: Optional[str] = Query(None, description="Filter by label (fake/real)")):
    try:
        logger.debug("Fetching detection history (limit=%s, filter=%s)", limit, label_filter)
        with _lock:
            history = list(detection_history)

        if label_filter:
            if label_filter not in ("fake", "real"):
                raise HTTPException(status_code=400, detail="Invalid label filter. Must be 'fake' or 'real'")
            history = [h for h in history if h.get("label") == label_filter]

        history = history[-limit:]
        history.reverse()

        return JSONResponse({"status": "success", "count": len(history), "filter": label_filter, "history": history})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching detection history: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

# -------------------------------
# NEW: Get detection statistics
# -------------------------------
@router.get("/stats")
async def get_detection_stats():
    try:
        logger.debug("Fetching detection statistics")
        with _lock:
            stats_copy = detection_stats.copy()
            total_detected = stats_copy.get("total_fake_detected", 0) + stats_copy.get("total_real_detected", 0)

        fake_percentage = (stats_copy.get("total_fake_detected", 0) / total_detected * 100) if total_detected > 0 else 0
        real_percentage = (stats_copy.get("total_real_detected", 0) / total_detected * 100) if total_detected > 0 else 0

        stats_response = {
            **stats_copy,
            "total_detections": total_detected,
            "fake_percentage": round(fake_percentage, 2),
            "real_percentage": round(real_percentage, 2),
            "cache_size": len(detection_cache),
            "history_size": len(detection_history)
        }

        return JSONResponse({"status": "success", "statistics": stats_response, "timestamp": _now_iso()})
    except Exception as e:
        logger.exception("Error fetching statistics: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# -------------------------------
# NEW: Clear detection history
# -------------------------------
@router.delete("/history")
async def clear_detection_history():
    try:
        logger.info("Clearing detection history and cache")
        with _lock:
            history_count = len(detection_history)
            cache_count = len(detection_cache)
            detection_history.clear()
            detection_cache.clear()
            detection_stats.update({"total_videos_processed": 0, "total_frames_analyzed": 0, "total_fake_detected": 0, "total_real_detected": 0, "average_processing_time": 0.0, "last_updated": _now_iso()})
        logger.info("Cleared %d history records and %d cache entries", history_count, cache_count)
        return JSONResponse({"status": "success", "message": f"Cleared {history_count} history records and {cache_count} cache entries"})
    except Exception as e:
        logger.exception("Error clearing history: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

# -------------------------------
# NEW: Export detection history
# -------------------------------
@router.get("/export")
async def export_detection_history(format: str = Query("json", description="Export format (json/csv)"), label_filter: Optional[str] = Query(None, description="Filter by label")):
    try:
        logger.info("Exporting detection history (format=%s, filter=%s)", format, label_filter)
        with _lock:
            history = list(detection_history)

        if label_filter:
            if label_filter not in ("fake", "real"):
                raise HTTPException(status_code=400, detail="Invalid label filter")
            history = [h for h in history if h.get("label") == label_filter]

        filename = f"deepfake_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if format == "json":
            export_data = {"export_time": _now_iso(), "filter": label_filter, "count": len(history), "statistics": detection_stats, "history": history}
            json_bytes = json.dumps(export_data, indent=2).encode("utf-8")
            file_stream = BytesIO(json_bytes)
            return StreamingResponse(file_stream, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}.json"})
        elif format == "csv":
            import csv
            from io import StringIO
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Filename", "File ID", "Label", "Probability", "Timestamp"])
            for record in history:
                writer.writerow([record.get("filename"), record.get("file_id"), record.get("label"), record.get("probability"), record.get("timestamp")])
            csv_bytes = csv_buffer.getvalue().encode("utf-8")
            file_stream = BytesIO(csv_bytes)
            return StreamingResponse(file_stream, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}.csv"})
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'csv'")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error exporting history: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to export history: {str(e)}")

# -------------------------------
# NEW: Get model information
# -------------------------------
@router.get("/model/info")
async def get_model_info():
    try:
        logger.debug("Fetching model information")
        model_loaded = getattr(detector, "model", None) is not None
        face_detector_loaded = getattr(detector, "face_app", None) is not None

        model_info = {
            "classification_model": {
                "name": "MobileNetV3-Small",
                "status": "loaded" if model_loaded else "not_loaded",
                "weights_loaded": False,
                "description": "Lightweight CNN for real-time deepfake classification",
                "note": "Model weights need to be trained and loaded (see TODO in deepfake_utils.py)" if not model_loaded else None
            },
            "face_detection_model": {
                "name": "RetinaFace (InsightFace)",
                "status": "loaded" if face_detector_loaded else "fallback",
                "fallback": "Haar Cascade" if not face_detector_loaded else None,
                "description": "Fast and accurate face detection for preprocessing"
            },
            "device": str(getattr(detector, "device", "cpu")),
            "ready_for_production": bool(model_loaded and face_detector_loaded),
            "limitations": [
                "DeepFake model weights are not trained yet (giving random predictions)",
                "Need to load actual trained weights from models/deepfake_mobilenet.pth"
            ] if not model_loaded else []
        }

        return JSONResponse({"status": "success", "model_info": model_info, "timestamp": _now_iso()})
    except Exception as e:
        logger.exception("Error fetching model info: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# -------------------------------
# NEW: Health check endpoint
# -------------------------------
@router.get("/health")
async def deepfake_health_check():
    try:
        model_available = getattr(detector, "model", None) is not None
        face_detector_available = getattr(detector, "face_app", None) is not None

        if model_available and face_detector_available:
            health_status = "healthy"
        elif model_available or face_detector_available:
            health_status = "degraded"
        else:
            health_status = "unhealthy"

        return JSONResponse({
            "status": health_status,
            "service": "deepfake_detection",
            "components": {
                "classification_model": "available" if model_available else "unavailable",
                "face_detector": "available" if face_detector_available else "fallback"
            },
            "statistics": {
                "videos_processed": detection_stats["total_videos_processed"],
                "cache_size": len(detection_cache)
            },
            "timestamp": _now_iso()
        })
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        return JSONResponse({"status": "error", "service": "deepfake_detection", "error": str(e)}, status_code=503)

# -------------------------------
# NEW: Batch video processing
# -------------------------------
@router.post("/batch")
async def batch_detect_deepfake(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
    max_frames: int = Query(20, ge=1, le=100)
):
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

        logger.info("Processing batch of %d videos", len(files))
        results: List[Dict[str, Any]] = []
        config = DeepfakeConfig(confidence_threshold=confidence_threshold, max_frames=max_frames)

        for file in files:
            safe_name = _safe_basename(file.filename)
            file_id = str(uuid.uuid4())
            file_path = None
            try:
                if not file.content_type or not file.content_type.startswith("video/"):
                    results.append({"filename": file.filename, "status": "error", "error": "Invalid file type"})
                    continue

                file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")
                with open(file_path, "wb") as buffer:
                    try:
                        shutil.copyfileobj(file.file, buffer)
                    except Exception:
                        content = await file.read()
                        buffer.write(content)

                result = process_video(file_path, config)
                results.append({"filename": file.filename, "status": "success", "label": result["label"], "probability": result["probability"], "confidence_level": result["statistics"]["confidence_level"], "processing_time": result["metadata"]["processing_time_seconds"]})

                # update stats
                with _lock:
                    detection_stats["total_videos_processed"] += 1
                    detection_stats["total_frames_analyzed"] += result["metadata"]["frames_analyzed"]
                    if result["label"] == "fake":
                        detection_stats["total_fake_detected"] += 1
                    else:
                        detection_stats["total_real_detected"] += 1

            except Exception as file_error:
                logger.exception("Error processing %s: %s", file.filename, file_error)
                results.append({"filename": file.filename, "status": "error", "error": str(file_error)})
            finally:
                try:
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as cleanup_error:
                    logger.error("Failed to cleanup %s: %s", file_path, cleanup_error)

        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]
        fake_count = len([r for r in successful if r.get("label") == "fake"])
        real_count = len([r for r in successful if r.get("label") == "real"])

        logger.info("Batch processing complete: %d success, %d failed", len(successful), len(failed))
        return JSONResponse({"status": "completed", "total_files": len(files), "successful": len(successful), "failed": len(failed), "summary": {"fake_detected": fake_count, "real_detected": real_count}, "results": results})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in batch processing: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# -------------------------------
# NEW: Get cache information
# -------------------------------
@router.get("/cache/info")
async def get_cache_info():
    try:
        with _lock:
            if detection_cache:
                ages = [(datetime.now() - entry["timestamp"]).total_seconds() for entry in detection_cache.values()]
                avg_age = (sum(ages) / len(ages) / 60) if ages else 0
                oldest_age = max(ages) / 60 if ages else 0
                newest_age = min(ages) / 60 if ages else 0
            else:
                avg_age = oldest_age = newest_age = 0

            cache_info = {"enabled": True, "size": len(detection_cache), "max_size": "unlimited", "expiry_minutes": CACHE_EXPIRY_MINUTES, "statistics": {"average_age_minutes": round(avg_age, 2), "oldest_entry_minutes": round(oldest_age, 2), "newest_entry_minutes": round(newest_age, 2)}}
        return JSONResponse({"status": "success", "cache_info": cache_info, "timestamp": _now_iso()})
    except Exception as e:
        logger.exception("Error fetching cache info: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get cache info: {str(e)}")

# -------------------------------
# NEW: Clear cache
# -------------------------------
@router.delete("/cache")
async def clear_cache():
    try:
        with _lock:
            cache_size = len(detection_cache)
            detection_cache.clear()
        logger.info("Cleared %d cache entries", cache_size)
        return JSONResponse({"status": "success", "message": f"Cleared {cache_size} cache entries"})
    except Exception as e:
        logger.exception("Error clearing cache: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# -------------------------------
# NEW: Get configuration
# -------------------------------
@router.get("/config")
async def get_detection_config():
    try:
        config = {
            "default_confidence_threshold": 0.5,
            "default_max_frames": 20,
            "default_sampling_strategy": "uniform",
            "supported_sampling_strategies": ["uniform", "random"],
            "cache_enabled": True,
            "cache_expiry_minutes": CACHE_EXPIRY_MINUTES,
            "upload_directory": UPLOAD_DIR,
            "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
            "max_batch_size": 10
        }
        return JSONResponse({"status": "success", "configuration": config})
    except Exception as e:
        logger.exception("Error fetching config: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

# -------------------------------
# NEW: Reset statistics
# -------------------------------
@router.post("/stats/reset")
async def reset_statistics():
    try:
        logger.info("Resetting detection statistics")
        with _lock:
            old_stats = detection_stats.copy()
            detection_stats.update({"total_videos_processed": 0, "total_frames_analyzed": 0, "total_fake_detected": 0, "total_real_detected": 0, "average_processing_time": 0.0, "last_updated": _now_iso()})
        return JSONResponse({"status": "success", "message": "Statistics reset successfully", "previous_stats": old_stats})
    except Exception as e:
        logger.exception("Error resetting statistics: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to reset statistics: {str(e)}")

# -------------------------------
# NEW: Get supported video formats
# -------------------------------
@router.get("/formats")
async def get_supported_formats():
    return JSONResponse({
        "status": "success",
        "supported_formats": [
            {"extension": "mp4", "mime_type": "video/mp4", "description": "MPEG-4 Part 14"},
            {"extension": "avi", "mime_type": "video/x-msvideo", "description": "Audio Video Interleave"},
            {"extension": "mov", "mime_type": "video/quicktime", "description": "QuickTime File Format"},
            {"extension": "mkv", "mime_type": "video/x-matroska", "description": "Matroska Video"},
            {"extension": "webm", "mime_type": "video/webm", "description": "WebM Video"},
            {"extension": "flv", "mime_type": "video/x-flv", "description": "Flash Video"}
        ],
        "note": "All formats supported by OpenCV VideoCapture are accepted"
    })

# -------------------------------
# NEW: Validate video file
# -------------------------------
@router.post("/validate")
async def validate_video_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    safe_name = _safe_basename(file.filename)
    file_path = None
    try:
        logger.info("Validating video file: %s", safe_name)
        file_path = os.path.join(UPLOAD_DIR, f"validate_{file_id}_{safe_name}")
        with open(file_path, "wb") as buffer:
            try:
                shutil.copyfileobj(file.file, buffer)
            except Exception:
                content = await file.read()
                buffer.write(content)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return JSONResponse({"status": "invalid", "filename": file.filename, "valid": False, "error": "Failed to open video file", "suggestion": "Check if the file is corrupted or in an unsupported format"})

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return JSONResponse({"status": "warning", "filename": file.filename, "valid": True, "can_read_frames": False, "warning": "Video file opens but cannot read frames", "metadata": {"frame_count": frame_count, "fps": fps, "resolution": f"{width}x{height}", "duration_seconds": round(duration, 2)}})

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        estimated_time = (frame_count / 20) * 0.1 if frame_count > 0 else 0.1

        logger.info("Video validation successful: %s", file.filename)
        return JSONResponse({
            "status": "valid",
            "filename": file.filename,
            "valid": True,
            "can_read_frames": True,
            "metadata": {"frame_count": frame_count, "fps": round(fps, 2), "resolution": f"{width}x{height}", "duration_seconds": round(duration, 2), "file_size_mb": round(file_size_mb, 2), "estimated_processing_time_seconds": round(estimated_time, 2)},
            "recommendations": {"suggested_max_frames": min(20, frame_count) if frame_count > 0 else 20, "suggested_sampling": "uniform" if duration > 10 else "all"}
        })

    except Exception as e:
        logger.exception("Error validating video: %s", e)
        return JSONResponse({"status": "error", "filename": file.filename, "valid": False, "error": str(e)})
    finally:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            logger.error("Failed to cleanup validation file: %s", cleanup_error)
