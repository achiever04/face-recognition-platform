# backend/app/routes/face.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
import os
from datetime import datetime
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
import logging
import numpy as np
import face_recognition
from io import BytesIO
import json
import tempfile
import threading

from app.services.face_service import face_service
from app.utils.db import create_target_log_files, faces_collection

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["Face Management"])

UPLOAD_DIR = os.getenv("FACE_UPLOAD_DIR", "data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Thread-safety lock for operations touching shared state / disk
_lock = threading.RLock()

# -------------------------------
# Pydantic Models for Request Validation
# -------------------------------
class FaceUpdateRequest(BaseModel):
    """Model for updating face metadata"""
    new_name: Optional[str] = Field(None, min_length=1, max_length=255, description="New name for the face")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class FaceSearchRequest(BaseModel):
    """Model for face search filters"""
    name_pattern: Optional[str] = Field(None, description="Search by name pattern")
    min_quality: Optional[float] = Field(None, ge=0, le=100, description="Minimum quality score")
    limit: Optional[int] = Field(50, ge=1, le=1000, description="Maximum results")


class BatchUploadResponse(BaseModel):
    """Model for batch upload response"""
    total: int
    successful: int
    failed: int
    results: List[dict]


# -------------------------------
# Helper utilities
# -------------------------------
def _now_iso() -> str:
    return datetime.now().isoformat()


def _safe_basename(filename: str) -> str:
    """
    Return a safe basename, stripping any path components.
    Keep only the final component and replace suspicious characters.
    """
    base = os.path.basename(filename or "")
    # Replace path traversal characters, keep filename simple
    safe = base.replace("..", "").replace("/", "_").replace("\\", "_")
    return safe or f"upload_{int(datetime.now().timestamp())}"


def _atomic_write(path: str, data: bytes):
    """
    Write bytes to path atomically using a temp file + os.replace.
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=".tmp_write_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # best-effort cleanup
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def get_directory_size(directory: str) -> float:
    """Calculate total size of directory in MB"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return round(total_size / (1024 * 1024), 2)
    except Exception as e:
        logger.warning(f"Failed to calculate directory size: {e}")
        return 0.0


# -------------------------------
# Upload face (ENHANCED)
# -------------------------------
@router.post("/upload")
async def upload_face(
    file: UploadFile = File(...),
    target_name: Optional[str] = Query(None, description="Target name (defaults to filename)"),
    save_raw: bool = Query(False, description="Save raw image file"),
    override: bool = Query(False, description="Override if face already exists"),
    min_quality: float = Query(50, ge=0, le=100, description="Minimum acceptable quality score")
):
    """
    Upload and encode a face image.

    ENHANCED: Added override option, better quality validation, duplicate detection
    """
    temp_path = None
    saved_raw_path = None

    try:
        # Secure filename and target
        safe_filename = _safe_basename(file.filename)
        target = (target_name or os.path.splitext(safe_filename)[0]).strip()

        logger.info(f"Processing face upload for target: {target} (filename: {safe_filename})")

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            logger.warning("Invalid file type: %s", file.content_type)
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        # Validate size (read file bytes once)
        content = await file.read()
        file_size = len(content)

        max_size = 10 * 1024 * 1024  # 10 MB
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        if file_size > max_size:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f} MB")

        logger.debug(f"File size: {file_size / 1024:.2f} KB")

        # Duplicate check (unless override)
        if not override and target in face_service.get_all_targets():
            logger.warning("Face already exists for target: %s", target)
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "message": f"Face for '{target}' already exists. Use override=true to replace.",
                    "target": target,
                    "existing": True
                }
            )

        # Optionally save raw file atomically
        if save_raw:
            saved_raw_path = os.path.join(UPLOAD_DIR, safe_filename)
            try:
                _atomic_write(saved_raw_path, content)
            except Exception as e:
                logger.warning("Failed to save raw file to disk: %s", e)
                saved_raw_path = None

        # Write a temporary file for processing
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{safe_filename}")
        try:
            _atomic_write(temp_path, content)
        except Exception as e:
            logger.error("Failed to write temp file: %s", e)
            raise HTTPException(status_code=500, detail="Failed to save uploaded file for processing")

        # Use face_service to encode
        encode_result = face_service.encode_face(temp_path, return_locations=True)

        if not encode_result.get("success"):
            logger.error("Face encoding failed: %s", encode_result.get("message"))
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": encode_result.get("message"),
                    "details": "No face detected or encoding failed"
                }
            )

        # Handle multiple faces
        if encode_result.get("face_count", 0) > 1:
            logger.warning("Multiple faces detected in upload for %s: %s", target, encode_result["face_count"])
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Multiple faces detected ({encode_result['face_count']}). Please upload image with single face.",
                    "face_count": encode_result["face_count"],
                    "suggestion": "Crop the image to contain only one face"
                }
            )

        # Extract encoding and location
        encoding = encode_result["encodings"][0]
        face_location = encode_result["locations"][0]

        # Load image for quality assessment
        image = face_recognition.load_image_file(temp_path)
        quality = face_service.assess_face_quality(image, face_location)
        logger.info("Face quality score for %s: %.2f", target, quality.get("score", 0.0))

        # Quality check
        if quality["score"] < min_quality:
            logger.warning("Low quality face for %s: %.2f < %.2f", target, quality["score"], min_quality)
            return JSONResponse(
                status_code=400,
                content={
                    "status": "warning",
                    "message": "Face quality too low for reliable recognition",
                    "quality_score": quality["score"],
                    "minimum_required": min_quality,
                    "issues": quality["issues"],
                    "recommendations": [
                        "Use better lighting",
                        "Ensure face is centered",
                        "Use higher resolution image",
                        "Avoid blurry images"
                    ]
                }
            )

        # If overriding, delete existing first
        overridden = False
        with _lock:
            existing_before = target in face_service.get_all_targets()
            if override and existing_before:
                logger.info("Overriding existing face for: %s", target)
                try:
                    del_result = face_service.delete_face(target)
                    if not del_result.get("success"):
                        logger.warning("Failed to delete existing face during override: %s", del_result.get("message"))
                except Exception:
                    logger.exception("Exception while deleting existing face during override")

            # Store new face
            store_result = face_service.store_face(target, encoding)

            if not store_result.get("success"):
                logger.error("Failed to store face %s: %s", target, store_result.get("message"))
                return JSONResponse(status_code=400, content={"status": "error", "message": store_result.get("message")})

            overridden = override and existing_before

        # Create target log files (best-effort)
        try:
            create_target_log_files(target)
        except Exception as e:
            logger.warning("Failed to create target log files for %s: %s", target, e)

        # Store metadata in DB (best-effort)
        try:
            faces_collection.update_one(
                {"target": target},
                {
                    "$set": {
                        "metadata": {
                            "original_filename": safe_filename,
                            "file_size_bytes": file_size,
                            "quality_score": quality["score"],
                            "uploaded_at": _now_iso(),
                            "face_location": face_location,
                            "image_resolution": f"{image.shape[1]}x{image.shape[0]}"
                        },
                        "updated_at": _now_iso()
                    }
                },
                upsert=True
            )
        except Exception as metadata_error:
            logger.warning("Failed to store metadata for %s: %s", target, metadata_error)

        logger.info("Successfully enrolled face for: %s", target)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Face successfully enrolled for '{target}'",
                "target": target,
                "filename": safe_filename,
                "overridden": overridden,
                "quality": {
                    "score": round(quality["score"], 2),
                    "rating": (
                        "excellent" if quality["score"] >= 80 else
                        "good" if quality["score"] >= 60 else
                        "acceptable"
                    ),
                    "issues": quality["issues"] if quality["issues"] else None
                },
                "metadata": {
                    "file_size_kb": round(file_size / 1024, 2),
                    "resolution": f"{image.shape[1]}x{image.shape[0]}"
                },
                "timestamp": _now_iso()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in face upload: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "detail": str(e)}
        )
    finally:
        # Clean up temp file unless user requested saved raw
        try:
            if temp_path and os.path.exists(temp_path) and (saved_raw_path is None or saved_raw_path != temp_path):
                os.remove(temp_path)
        except Exception:
            logger.debug("Failed to cleanup temp file", exc_info=True)


# -------------------------------
# Batch upload faces
# -------------------------------
@router.post("/upload/batch")
async def batch_upload_faces(
    files: List[UploadFile] = File(...),
    save_raw: bool = Query(False, description="Save raw image files"),
    min_quality: float = Query(50, ge=0, le=100, description="Minimum quality score")
):
    """
    Upload multiple face images in batch.

    Maximum 20 files per batch.
    """
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 files allowed per batch")

        logger.info("Processing batch upload of %d faces", len(files))

        results = []
        successful = 0
        failed = 0

        for file in files:
            safe_filename = _safe_basename(file.filename)
            target = os.path.splitext(safe_filename)[0]

            try:
                if not file.content_type or not file.content_type.startswith("image/"):
                    results.append({"filename": file.filename, "target": target, "status": "error", "error": "Invalid file type"})
                    failed += 1
                    continue

                content = await file.read()
                file_size = len(content)
                if file_size == 0:
                    results.append({"filename": file.filename, "target": target, "status": "error", "error": "File empty"})
                    failed += 1
                    continue

                # Skip duplicates
                if target in face_service.get_all_targets():
                    results.append({"filename": file.filename, "target": target, "status": "skipped", "reason": "Already exists"})
                    failed += 1
                    continue

                temp_path = os.path.join(UPLOAD_DIR, f"batch_{target}_{safe_filename}")
                try:
                    _atomic_write(temp_path, content)
                except Exception:
                    results.append({"filename": file.filename, "target": target, "status": "error", "error": "Failed to write temp file"})
                    failed += 1
                    continue

                try:
                    encode_result = face_service.encode_face(temp_path, return_locations=True)
                    if not encode_result.get("success") or encode_result.get("face_count", 0) == 0:
                        results.append({"filename": file.filename, "target": target, "status": "error", "error": "No face detected"})
                        failed += 1
                        continue
                    if encode_result.get("face_count", 0) > 1:
                        results.append({"filename": file.filename, "target": target, "status": "error", "error": f"Multiple faces detected ({encode_result['face_count']})"})
                        failed += 1
                        continue

                    encoding = encode_result["encodings"][0]
                    face_location = encode_result["locations"][0]
                    image = face_recognition.load_image_file(temp_path)
                    quality = face_service.assess_face_quality(image, face_location)
                    if quality["score"] < min_quality:
                        results.append({"filename": file.filename, "target": target, "status": "error", "error": f"Quality too low ({quality['score']:.1f} < {min_quality})"})
                        failed += 1
                        continue

                    with _lock:
                        store_result = face_service.store_face(target, encoding)

                    if store_result.get("success"):
                        try:
                            create_target_log_files(target)
                        except Exception:
                            logger.debug("Failed to create log files for %s", target)
                        results.append({"filename": file.filename, "target": target, "status": "success", "quality_score": round(quality["score"], 2)})
                        successful += 1
                    else:
                        results.append({"filename": file.filename, "target": target, "status": "error", "error": store_result.get("message")})
                        failed += 1

                finally:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass

            except Exception as file_error:
                logger.exception("Error processing %s: %s", file.filename, file_error)
                results.append({"filename": file.filename, "target": target, "status": "error", "error": str(file_error)})
                failed += 1

        logger.info("Batch upload complete: %d success, %d failed", successful, failed)
        return JSONResponse({"status": "completed", "total": len(files), "successful": successful, "failed": failed, "results": results})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in batch upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


# -------------------------------
# List all enrolled faces
# -------------------------------
@router.get("/list")
async def list_faces(
    include_metadata: bool = Query(False, description="Include face metadata"),
    sort_by: str = Query("name", description="Sort by (name/date/quality)"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Limit results")
):
    """
    Return all enrolled face targets.
    """
    try:
        logger.debug("Listing faces (metadata=%s, sort=%s)", include_metadata, sort_by)
        targets = face_service.get_all_targets()

        if not targets:
            return JSONResponse({"status": "success", "count": 0, "targets": [], "message": "No faces enrolled yet"})

        if include_metadata:
            faces_data: List[Dict[str, Any]] = []
            for t in targets:
                face_data = {"name": t}
                try:
                    db_record = faces_collection.find_one({"target": t})
                    if db_record and "metadata" in db_record:
                        face_data["metadata"] = db_record["metadata"]
                    if db_record and "updated_at" in db_record:
                        face_data["updated_at"] = db_record["updated_at"]
                except Exception as db_error:
                    logger.warning("Failed to get metadata for %s: %s", t, db_error)
                faces_data.append(face_data)

            # Sorting
            if sort_by == "date" and any("updated_at" in f for f in faces_data):
                faces_data.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            elif sort_by == "quality" and any("metadata" in f for f in faces_data):
                faces_data.sort(key=lambda x: x.get("metadata", {}).get("quality_score", 0), reverse=True)
            else:
                faces_data.sort(key=lambda x: x["name"])

            if limit:
                faces_data = faces_data[:limit]

            return JSONResponse({"status": "success", "count": len(faces_data), "total": len(targets), "faces": faces_data})
        else:
            targets_sorted = sorted(targets)
            if limit:
                targets_sorted = targets_sorted[:limit]
            return JSONResponse({"status": "success", "count": len(targets_sorted), "total": len(targets), "targets": targets_sorted})

    except Exception as e:
        logger.exception("Error listing faces: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list faces: {str(e)}")


# -------------------------------
# Get specific face details
# -------------------------------
@router.get("/detail/{target}")
async def get_face_details(target: str):
    """
    Get detailed information about a specific face.
    """
    try:
        logger.debug("Fetching details for target: %s", target)
        if target not in face_service.get_all_targets():
            raise HTTPException(status_code=404, detail=f"Face '{target}' not found")

        db_record = faces_collection.find_one({"target": target})
        if not db_record:
            return JSONResponse({"status": "success", "target": target, "exists": True, "metadata": None, "message": "Face exists but no metadata available"})

        face_details = {
            "target": target,
            "exists": True,
            "enrolled_at": db_record.get("updated_at"),
            "metadata": db_record.get("metadata", {})
        }
        return JSONResponse({"status": "success", **face_details})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching face details: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get face details: {str(e)}")


# -------------------------------
# Delete a face
# -------------------------------
@router.delete("/delete/{target}")
async def delete_face(target: str, delete_logs: bool = Query(False, description="Also delete log files")):
    """
    Remove face from system.
    """
    try:
        logger.info("Deleting face: %s", target)
        result = face_service.delete_face(target)
        if not result.get("success"):
            logger.warning("Face deletion failed for %s: %s", target, result.get("message"))
            raise HTTPException(status_code=404, detail=result.get("message"))

        logs_deleted = False
        if delete_logs:
            try:
                import glob
                log_files = glob.glob(f"logs/{target}.*")
                for log_file in log_files:
                    try:
                        os.remove(log_file)
                    except Exception:
                        pass
                logs_deleted = len(log_files) > 0
                logger.info("Deleted %d log files for %s", len(log_files), target)
            except Exception as log_error:
                logger.warning("Failed to delete logs for %s: %s", target, log_error)

        return JSONResponse({"status": "success", "message": result.get("message"), "target": target, "logs_deleted": logs_deleted})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting face: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {str(e)}")


# -------------------------------
# Update face
# -------------------------------
@router.patch("/update/{target}")
async def update_face(target: str, updates: FaceUpdateRequest):
    """
    Update face metadata or rename face.
    """
    try:
        logger.info("Updating face: %s", target)
        if target not in face_service.get_all_targets():
            raise HTTPException(status_code=404, detail=f"Face '{target}' not found")

        # Handle rename
        if updates.new_name and updates.new_name != target:
            new_name = updates.new_name.strip()
            if new_name in face_service.get_all_targets():
                raise HTTPException(status_code=400, detail=f"Face with name '{new_name}' already exists")

            try:
                from app.state import ENCODINGS
                encoding = ENCODINGS.get(target)
                if not encoding:
                    raise HTTPException(status_code=500, detail="Failed to retrieve face encoding")

                # Perform rename via delete + store (preserve original behavior)
                del_result = face_service.delete_face(target)
                if not del_result.get("success"):
                    raise HTTPException(status_code=500, detail=f"Failed to delete old target during rename: {del_result.get('message')}")

                import numpy as _np
                store_result = face_service.store_face(new_name, _np.array(encoding))
                if not store_result.get("success"):
                    raise HTTPException(status_code=500, detail=f"Failed to store new target during rename: {store_result.get('message')}")

                # Update DB record if exists
                try:
                    faces_collection.update_one({"target": target}, {"$set": {"target": new_name}})
                except Exception:
                    logger.debug("Failed to rename DB record for %s -> %s", target, new_name)

                logger.info("Renamed face from %s to %s", target, new_name)
                target = new_name  # continue to metadata update
            except HTTPException:
                raise
            except Exception as rename_error:
                logger.exception("Rename failed: %s", rename_error)
                raise HTTPException(status_code=500, detail=f"Rename failed: {str(rename_error)}")

        # Update metadata
        metadata_updated = False
        if updates.metadata:
            try:
                # store custom metadata under metadata.custom and update timestamp
                faces_collection.update_one(
                    {"target": target},
                    {"$set": {"metadata.custom": updates.metadata, "metadata.last_updated": _now_iso(), "updated_at": _now_iso()}},
                    upsert=True
                )
                metadata_updated = True
                logger.info("Updated metadata for %s", target)
            except Exception as meta_error:
                logger.warning("Failed to update metadata for %s: %s", target, meta_error)

        return JSONResponse({"status": "success", "message": f"Face '{target}' updated successfully", "target": target, "changes": {"renamed": updates.new_name is not None, "metadata_updated": metadata_updated}})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating face: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to update face: {str(e)}")


# -------------------------------
# Compare two faces
# -------------------------------
@router.post("/compare")
async def compare_faces_endpoint(
    file: UploadFile = File(...),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="Matching threshold"),
    top_k: int = Query(5, ge=1, le=50, description="Return top K matches")
):
    """
    Compare uploaded face against all stored faces.
    """
    temp_path = None
    try:
        safe_filename = _safe_basename(file.filename)
        logger.info("Comparing face from: %s", safe_filename)

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        content = await file.read()
        temp_path = os.path.join(UPLOAD_DIR, f"compare_{safe_filename}")
        try:
            _atomic_write(temp_path, content)
        except Exception:
            logger.exception("Failed to write compare temp file")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file for comparison")

        encode_result = face_service.encode_face(temp_path)
        if not encode_result.get("success"):
            return JSONResponse(status_code=400, content={"status": "error", "message": encode_result.get("message")})

        if encode_result.get("face_count", 0) > 1:
            return JSONResponse(status_code=400, content={"status": "warning", "message": f"Multiple faces detected ({encode_result['face_count']}). Using first face for comparison.", "face_count": encode_result["face_count"]})

        encoding = encode_result["encodings"][0]

        # Temporarily set threshold on service and restore after
        original_tolerance = face_service.tolerance
        face_service.tolerance = threshold
        try:
            comparisons = face_service.compare_faces(encoding, return_distances=True)
        finally:
            face_service.tolerance = original_tolerance

        matches = [c for c in comparisons if c.get("match")]
        matches = matches[:top_k]

        logger.info("Found %d matches (threshold=%s)", len(matches), threshold)

        return JSONResponse({
            "status": "success",
            "filename": safe_filename,
            "threshold": threshold,
            "total_faces_checked": len(comparisons),
            "matches_found": len(matches),
            "top_matches": matches,
            "all_comparisons": comparisons if not matches else None
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in face comparison: %s", e)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


# -------------------------------
# Search faces
# -------------------------------
@router.get("/search")
async def search_faces(query: str = Query(..., min_length=1, description="Search query"), limit: int = Query(50, ge=1, le=1000, description="Maximum results")):
    """
    Search faces by name pattern.
    """
    try:
        logger.debug("Searching faces with query: %s", query)
        targets = face_service.get_all_targets()
        q = query.lower()
        matches = [t for t in targets if q in t.lower()]
        matches = sorted(matches)[:limit]
        return JSONResponse({"status": "success", "query": query, "matches_found": len(matches), "total_faces": len(targets), "matches": matches})
    except Exception as e:
        logger.exception("Error searching faces: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# -------------------------------
# Get face statistics
# -------------------------------
@router.get("/stats")
async def get_face_stats():
    """
    Get face enrollment statistics.
    """
    try:
        logger.debug("Fetching face statistics")
        targets = face_service.get_all_targets()
        quality_distribution = {"excellent": 0, "good": 0, "acceptable": 0}
        total_with_metadata = 0

        for t in targets:
            try:
                db_record = faces_collection.find_one({"target": t})
                if db_record and "metadata" in db_record:
                    total_with_metadata += 1
                    quality = db_record["metadata"].get("quality_score", 0)
                    if quality >= 80:
                        quality_distribution["excellent"] += 1
                    elif quality >= 60:
                        quality_distribution["good"] += 1
                    elif quality >= 50:
                        quality_distribution["acceptable"] += 1
            except Exception:
                continue

        stats = {
            "total_faces": len(targets),
            "faces_with_metadata": total_with_metadata,
            "quality_distribution": quality_distribution,
            "storage_info": {"upload_directory": UPLOAD_DIR, "upload_directory_size_mb": get_directory_size(UPLOAD_DIR)}
        }
        return JSONResponse({"status": "success", "statistics": stats, "timestamp": _now_iso()})
    except Exception as e:
        logger.exception("Error fetching face stats: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# -------------------------------
# Find similar faces
# -------------------------------
@router.get("/similar/{target}")
async def find_similar_faces(target: str, threshold: float = Query(0.5, ge=0.0, le=1.0, description="Similarity threshold"), limit: int = Query(5, ge=1, le=20, description="Maximum similar faces")):
    """
    Find faces similar to a specific target.
    """
    try:
        logger.debug("Finding similar faces to: %s", target)
        if target not in face_service.get_all_targets():
            raise HTTPException(status_code=404, detail=f"Face '{target}' not found")

        from app.state import ENCODINGS
        target_encoding = ENCODINGS.get(target)
        if not target_encoding:
            raise HTTPException(status_code=500, detail="Failed to retrieve face encoding")

        target_enc = np.array(target_encoding)
        original_tolerance = face_service.tolerance
        face_service.tolerance = threshold
        try:
            comparisons = face_service.compare_faces(target_enc, return_distances=True)
        finally:
            face_service.tolerance = original_tolerance

        similar = [c for c in comparisons if c.get("target") != target and c.get("match")]
        similar.sort(key=lambda x: x.get("distance", 1.0))
        similar = similar[:limit]

        logger.info("Found %d similar faces to %s", len(similar), target)
        return JSONResponse({"status": "success", "target": target, "threshold": threshold, "similar_faces_found": len(similar), "similar_faces": similar})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error finding similar faces: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to find similar faces: {str(e)}")


# -------------------------------
# Export face database
# -------------------------------
@router.get("/export")
async def export_face_database(format: str = Query("json", description="Export format (json/csv)"), include_encodings: bool = Query(False, description="Include face encodings (large file)")):
    """
    Export face database to file.
    """
    try:
        logger.info("Exporting face database (format=%s)", format)
        targets = face_service.get_all_targets()
        export_data = []

        for t in targets:
            face_data: Dict[str, Any] = {"name": t}
            try:
                db_record = faces_collection.find_one({"target": t})
                if db_record:
                    if "metadata" in db_record:
                        face_data["metadata"] = db_record["metadata"]
                    if "updated_at" in db_record:
                        face_data["enrolled_at"] = db_record["updated_at"]
                    if include_encodings:
                        from app.state import ENCODINGS
                        enc = ENCODINGS.get(t)
                        if enc:
                            face_data["encoding_length"] = len(enc)
            except Exception as db_error:
                logger.warning("Failed to get data for %s: %s", t, db_error)
            export_data.append(face_data)

        filename = f"face_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if format == "json":
            export_json = {"export_time": _now_iso(), "total_faces": len(export_data), "includes_encodings": include_encodings, "faces": export_data}
            json_bytes = json.dumps(export_json, indent=2).encode("utf-8")
            file_stream = BytesIO(json_bytes)
            return StreamingResponse(file_stream, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}.json"})

        elif format == "csv":
            import csv
            from io import StringIO
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Name", "Enrolled At", "Quality Score", "File Size KB", "Resolution"])
            for face in export_data:
                metadata = face.get("metadata", {})
                file_size_kb = metadata.get("file_size_bytes", 0) / 1024 if metadata.get("file_size_bytes") else "N/A"
                writer.writerow([face["name"], face.get("enrolled_at", "N/A"), metadata.get("quality_score", "N/A"), file_size_kb, metadata.get("image_resolution", "N/A")])
            csv_bytes = csv_buffer.getvalue().encode("utf-8")
            file_stream = BytesIO(csv_bytes)
            return StreamingResponse(file_stream, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}.csv"})
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'csv'")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error exporting database: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# -------------------------------
# Bulk delete faces
# -------------------------------
@router.post("/delete/bulk")
async def bulk_delete_faces(targets: List[str] = Body(..., embed=True, description="List of target names to delete"), delete_logs: bool = Query(False, description="Also delete log files")):
    """
    Delete multiple faces in bulk.
    """
    try:
        if len(targets) == 0:
            raise HTTPException(status_code=400, detail="No targets provided")
        if len(targets) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 faces can be deleted at once")

        logger.info("Bulk deleting %d faces", len(targets))
        results = {"deleted": [], "not_found": [], "errors": []}

        for t in targets:
            try:
                if t not in face_service.get_all_targets():
                    results["not_found"].append(t)
                    continue
                delete_result = face_service.delete_face(t)
                if delete_result.get("success"):
                    results["deleted"].append(t)
                    if delete_logs:
                        try:
                            import glob
                            log_files = glob.glob(f"logs/{t}.*")
                            for log_file in log_files:
                                try:
                                    os.remove(log_file)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                else:
                    results["errors"].append({"target": t, "error": delete_result.get("message")})
            except Exception as delete_error:
                results["errors"].append({"target": t, "error": str(delete_error)})

        logger.info("Bulk delete complete: %d deleted, %d not found, %d errors", len(results["deleted"]), len(results["not_found"]), len(results["errors"]))
        return JSONResponse({"status": "completed", "total_requested": len(targets), "deleted": len(results["deleted"]), "not_found": len(results["not_found"]), "errors": len(results["errors"]), "results": results})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in bulk delete: %s", e)
        raise HTTPException(status_code=500, detail=f"Bulk delete failed: {str(e)}")


# -------------------------------
# Validate face image
# -------------------------------
@router.post("/validate")
async def validate_face_image(file: UploadFile = File(...)):
    """
    Validate a face image without enrolling it.
    """
    temp_path = None
    try:
        safe_filename = _safe_basename(file.filename)
        logger.info("Validating face image: %s", safe_filename)

        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse({"status": "invalid", "valid": False, "error": "Not an image file", "file_type": file.content_type})

        content = await file.read()
        file_size = len(content)
        if file_size == 0:
            return JSONResponse({"status": "invalid", "valid": False, "error": "File is empty"})
        if file_size > 10 * 1024 * 1024:
            return JSONResponse({"status": "invalid", "valid": False, "error": "File too large (max 10MB)", "file_size_mb": round(file_size / (1024 * 1024), 2)})

        temp_path = os.path.join(UPLOAD_DIR, f"validate_{safe_filename}")
        try:
            _atomic_write(temp_path, content)
        except Exception:
            logger.exception("Failed to write validate temp file")
            return JSONResponse({"status": "invalid", "valid": False, "error": "Failed to save uploaded file for validation"})

        encode_result = face_service.encode_face(temp_path, return_locations=True)
        if not encode_result.get("success"):
            return JSONResponse({"status": "invalid", "valid": False, "error": encode_result.get("message"), "suggestion": "Ensure the image contains a clear, front-facing face"})

        face_count = encode_result.get("face_count", 0)
        if face_count == 0:
            return JSONResponse({"status": "invalid", "valid": False, "error": "No faces detected", "suggestions": ["Ensure good lighting", "Face should be clearly visible", "Try a different angle"]})
        if face_count > 1:
            return JSONResponse({"status": "warning", "valid": True, "warning": f"Multiple faces detected ({face_count})", "face_count": face_count, "suggestion": "Crop the image to contain only one face for best results"})

        encoding = encode_result["encodings"][0]
        face_location = encode_result["locations"][0]
        image = face_recognition.load_image_file(temp_path)
        quality = face_service.assess_face_quality(image, face_location)

        if quality["score"] >= 60:
            status = "valid"
            message = "Image is suitable for face enrollment"
        elif quality["score"] >= 50:
            status = "acceptable"
            message = "Image quality is acceptable but could be improved"
        else:
            status = "poor"
            message = "Image quality is too low for reliable recognition"

        logger.info("Validation complete: %s (quality: %.2f)", status, quality["score"])

        recommendations = [
            ("Use better lighting" if quality["score"] < 70 else None),
            ("Center the face in the frame" if quality["position_score"] < 70 else None),
            ("Move closer to camera" if quality["size_score"] < 70 else None),
            ("Ensure face is not tilted" if quality["aspect_score"] < 80 else None)
        ]
        recommendations = [r for r in recommendations if r is not None]

        return JSONResponse({
            "status": status,
            "valid": quality["score"] >= 50,
            "message": message,
            "quality": {
                "overall_score": round(quality["score"], 2),
                "size_score": round(quality["size_score"], 2),
                "position_score": round(quality["position_score"], 2),
                "aspect_score": round(quality["aspect_score"], 2),
                "rating": ("excellent" if quality["score"] >= 80 else "good" if quality["score"] >= 60 else "acceptable" if quality["score"] >= 50 else "poor")
            },
            "issues": quality["issues"],
            "metadata": {"face_count": face_count, "file_size_kb": round(file_size / 1024, 2), "resolution": f"{image.shape[1]}x{image.shape[0]}"},
            "recommendations": recommendations
        })

    except Exception as e:
        logger.exception("Error validating image: %s", e)
        return JSONResponse({"status": "error", "valid": False, "error": str(e)})
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


# -------------------------------
# Health check
# -------------------------------
@router.get("/health")
async def face_health_check():
    """
    Health check for face management service.
    """
    try:
        targets = face_service.get_all_targets()
        db_accessible = True
        try:
            faces_collection.find_one()
        except Exception as db_error:
            logger.error("Database check failed: %s", db_error)
            db_accessible = False

        upload_dir_writable = os.access(UPLOAD_DIR, os.W_OK)
        if db_accessible and upload_dir_writable:
            health_status = "healthy"
        elif db_accessible or upload_dir_writable:
            health_status = "degraded"
        else:
            health_status = "unhealthy"

        return JSONResponse({
            "status": health_status,
            "service": "face_management",
            "components": {
                "face_service": "operational",
                "database": "accessible" if db_accessible else "unavailable",
                "upload_directory": "writable" if upload_dir_writable else "read-only"
            },
            "statistics": {"total_faces": len(targets), "upload_directory": UPLOAD_DIR},
            "timestamp": _now_iso()
        })
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        return JSONResponse({"status": "error", "service": "face_management", "error": str(e)}, status_code=503)


# -------------------------------
# Clear all faces (with confirmation)
# -------------------------------
@router.delete("/clear")
async def clear_all_faces(confirmation: str = Query(..., description="Must be 'CONFIRM_DELETE_ALL'")):
    """
    Clear all enrolled faces from the system.
    """
    try:
        if confirmation != "CONFIRM_DELETE_ALL":
            raise HTTPException(status_code=400, detail="Invalid confirmation. Must provide confirmation=CONFIRM_DELETE_ALL")

        logger.warning("CLEARING ALL FACES - confirmed")
        targets = face_service.get_all_targets()
        total = len(targets)
        deleted = 0
        errors = []

        for t in targets:
            try:
                result = face_service.delete_face(t)
                if result.get("success"):
                    deleted += 1
                else:
                    errors.append({"target": t, "error": result.get("message")})
            except Exception as delete_error:
                errors.append({"target": t, "error": str(delete_error)})

        logger.warning("Cleared %d faces, %d errors", deleted, len(errors))
        return JSONResponse({"status": "completed", "message": f"Cleared {deleted} of {total} faces", "deleted": deleted, "errors": len(errors), "error_details": errors if errors else None})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error clearing faces: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to clear faces: {str(e)}")
