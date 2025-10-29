# backend/app/routes/face.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
import os
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
import logging
import numpy as np
import face_recognition
from io import BytesIO
import json

from app.services.face_service import face_service
from app.utils.db import create_target_log_files, faces_collection

# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["Face Management"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# NEW: Pydantic Models for Request Validation
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
# EXISTING: Upload face (ENHANCED)
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
    try:
        # Use filename as target if not provided (EXISTING)
        target = target_name or file.filename
        
        logger.info(f"Processing face upload for target: {target}")
        
        # Validate file type (EXISTING - enhanced validation)
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(400, "Only image files are allowed")
        
        # Validate file size (NEW)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        max_size = 10 * 1024 * 1024  # 10 MB
        if file_size > max_size:
            raise HTTPException(400, f"File too large. Maximum size: {max_size / (1024*1024):.1f} MB")
        
        if file_size == 0:
            raise HTTPException(400, "File is empty")
        
        logger.debug(f"File size: {file_size / 1024:.2f} KB")
        
        # Check for duplicate (NEW)
        if not override and target in face_service.get_all_targets():
            logger.warning(f"Face already exists for target: {target}")
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "message": f"Face for '{target}' already exists. Use override=true to replace.",
                    "target": target,
                    "existing": True
                }
            )
        
        # Save file temporarily or permanently (EXISTING)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        if save_raw:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            await file.seek(0)
        
        # Read file content (EXISTING)
        file_content = await file.read()
        
        # Create temporary file for processing (EXISTING)
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        try:
            # Encode face using service (EXISTING)
            encode_result = face_service.encode_face(temp_path, return_locations=True)
            
            if not encode_result["success"]:
                logger.error(f"Face encoding failed: {encode_result['message']}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": encode_result["message"],
                        "details": "No face detected or encoding failed"
                    }
                )
            
            # Check for multiple faces (EXISTING - enhanced message)
            if encode_result["face_count"] > 1:
                logger.warning(f"Multiple faces detected: {encode_result['face_count']}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Multiple faces detected ({encode_result['face_count']}). Please upload image with single face.",
                        "face_count": encode_result["face_count"],
                        "suggestion": "Crop the image to contain only one face"
                    }
                )
            
            # Get face encoding and location (EXISTING)
            encoding = encode_result["encodings"][0]
            face_location = encode_result["locations"][0]
            
            # Assess quality (EXISTING)
            image = face_recognition.load_image_file(temp_path)
            quality = face_service.assess_face_quality(image, face_location)
            
            logger.info(f"Face quality score: {quality['score']:.2f}")
            
            # Validate quality (ENHANCED)
            if quality["score"] < min_quality:
                logger.warning(f"Low quality face: {quality['score']:.2f} < {min_quality}")
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
            
            # Store face using service (EXISTING - enhanced for override)
            if override and target in face_service.get_all_targets():
                # Delete existing face first
                logger.info(f"Overriding existing face for: {target}")
                delete_result = face_service.delete_face(target)
                if not delete_result["success"]:
                    logger.error(f"Failed to delete existing face: {delete_result['message']}")
            
            store_result = face_service.store_face(target, encoding)
            
            if not store_result["success"]:
                logger.error(f"Failed to store face: {store_result['message']}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": store_result["message"]
                    }
                )
            
            # Create log files (EXISTING)
            create_target_log_files(target)
            
            # Store metadata (NEW)
            try:
                faces_collection.update_one(
                    {"target": target},
                    {
                        "$set": {
                            "metadata": {
                                "original_filename": file.filename,
                                "file_size_bytes": file_size,
                                "quality_score": quality["score"],
                                "uploaded_at": datetime.now().isoformat(),
                                "face_location": face_location,
                                "image_resolution": f"{image.shape[1]}x{image.shape[0]}"
                            }
                        }
                    }
                )
            except Exception as metadata_error:
                logger.warning(f"Failed to store metadata: {metadata_error}")
            
            logger.info(f"Successfully enrolled face for: {target}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": f"Face successfully enrolled for '{target}'",
                    "target": target,
                    "filename": file.filename,
                    "overridden": override and target in face_service.get_all_targets(),
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
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        finally:
            # Clean up temp file (EXISTING - enhanced)
            if not save_raw and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup temp file: {cleanup_error}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face upload: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "detail": str(e)
            }
        )

# -------------------------------
# NEW: Batch upload faces
# -------------------------------
@router.post("/upload/batch")
async def batch_upload_faces(
    files: List[UploadFile] = File(...),
    save_raw: bool = Query(False, description="Save raw image files"),
    min_quality: float = Query(50, ge=0, le=100, description="Minimum quality score")
):
    """
    Upload multiple face images in batch.
    
    **NEW ENDPOINT**
    
    Maximum 20 files per batch.
    """
    try:
        if len(files) == 0:
            raise HTTPException(400, "No files provided")
        
        if len(files) > 20:
            raise HTTPException(400, "Maximum 20 files allowed per batch")
        
        logger.info(f"Processing batch upload of {len(files)} faces")
        
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            try:
                # Get target name from filename (remove extension)
                target = os.path.splitext(file.filename)[0]
                
                # Validate file type
                if not file.content_type or not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "target": target,
                        "status": "error",
                        "error": "Invalid file type"
                    })
                    failed += 1
                    continue
                
                # Check duplicate
                if target in face_service.get_all_targets():
                    results.append({
                        "filename": file.filename,
                        "target": target,
                        "status": "skipped",
                        "reason": "Already exists"
                    })
                    failed += 1
                    continue
                
                # Save temp file
                temp_path = os.path.join(UPLOAD_DIR, f"batch_{target}_{file.filename}")
                
                with open(temp_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                try:
                    # Encode face
                    encode_result = face_service.encode_face(temp_path, return_locations=True)
                    
                    if not encode_result["success"] or encode_result["face_count"] == 0:
                        results.append({
                            "filename": file.filename,
                            "target": target,
                            "status": "error",
                            "error": "No face detected"
                        })
                        failed += 1
                        continue
                    
                    if encode_result["face_count"] > 1:
                        results.append({
                            "filename": file.filename,
                            "target": target,
                            "status": "error",
                            "error": f"Multiple faces detected ({encode_result['face_count']})"
                        })
                        failed += 1
                        continue
                    
                    # Get encoding and quality
                    encoding = encode_result["encodings"][0]
                    face_location = encode_result["locations"][0]
                    
                    image = face_recognition.load_image_file(temp_path)
                    quality = face_service.assess_face_quality(image, face_location)
                    
                    # Check quality
                    if quality["score"] < min_quality:
                        results.append({
                            "filename": file.filename,
                            "target": target,
                            "status": "error",
                            "error": f"Quality too low ({quality['score']:.1f} < {min_quality})"
                        })
                        failed += 1
                        continue
                    
                    # Store face
                    store_result = face_service.store_face(target, encoding)
                    
                    if store_result["success"]:
                        create_target_log_files(target)
                        
                        results.append({
                            "filename": file.filename,
                            "target": target,
                            "status": "success",
                            "quality_score": round(quality["score"], 2)
                        })
                        successful += 1
                    else:
                        results.append({
                            "filename": file.filename,
                            "target": target,
                            "status": "error",
                            "error": store_result["message"]
                        })
                        failed += 1
                
                finally:
                    # Cleanup temp file
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
            
            except Exception as file_error:
                logger.error(f"Error processing {file.filename}: {file_error}")
                results.append({
                    "filename": file.filename,
                    "target": os.path.splitext(file.filename)[0],
                    "status": "error",
                    "error": str(file_error)
                })
                failed += 1
        
        logger.info(f"Batch upload complete: {successful} success, {failed} failed")
        
        return JSONResponse({
            "status": "completed",
            "total": len(files),
            "successful": successful,
            "failed": failed,
            "results": results
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Batch upload failed: {str(e)}")

# -------------------------------
# EXISTING: List all enrolled faces (ENHANCED)
# -------------------------------
@router.get("/list")
async def list_faces(
    include_metadata: bool = Query(False, description="Include face metadata"),
    sort_by: str = Query("name", description="Sort by (name/date/quality)"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Limit results")
):
    """
    Return all enrolled face targets.
    
    ENHANCED: Added metadata, sorting, pagination
    """
    try:
        logger.debug(f"Listing faces (metadata={include_metadata}, sort={sort_by})")
        
        targets = face_service.get_all_targets()
        
        if not targets:
            return JSONResponse({
                "status": "success",
                "count": 0,
                "targets": [],
                "message": "No faces enrolled yet"
            })
        
        # Build response with optional metadata (NEW)
        if include_metadata:
            faces_data = []
            
            for target in targets:
                face_data = {"name": target}
                
                # Try to get metadata from database
                try:
                    db_record = faces_collection.find_one({"target": target})
                    if db_record and "metadata" in db_record:
                        face_data["metadata"] = db_record["metadata"]
                    if db_record and "updated_at" in db_record:
                        face_data["updated_at"] = db_record["updated_at"]
                except Exception as db_error:
                    logger.warning(f"Failed to get metadata for {target}: {db_error}")
                
                faces_data.append(face_data)
            
            # Sort (NEW)
            if sort_by == "date" and any("updated_at" in f for f in faces_data):
                faces_data.sort(
                    key=lambda x: x.get("updated_at", ""), 
                    reverse=True
                )
            elif sort_by == "quality" and any("metadata" in f for f in faces_data):
                faces_data.sort(
                    key=lambda x: x.get("metadata", {}).get("quality_score", 0),
                    reverse=True
                )
            else:
                # Sort by name (default)
                faces_data.sort(key=lambda x: x["name"])
            
            # Apply limit
            if limit:
                faces_data = faces_data[:limit]
            
            return JSONResponse({
                "status": "success",
                "count": len(faces_data),
                "total": len(targets),
                "faces": faces_data
            })
        else:
            # Simple list without metadata (EXISTING - enhanced)
            targets_sorted = sorted(targets)
            
            if limit:
                targets_sorted = targets_sorted[:limit]
            
            return JSONResponse({
                "status": "success",
                "count": len(targets_sorted),
                "total": len(targets),
                "targets": targets_sorted
            })
    
    except Exception as e:
        logger.error(f"Error listing faces: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to list faces: {str(e)}")

# -------------------------------
# NEW: Get specific face details
# -------------------------------
@router.get("/detail/{target}")
async def get_face_details(target: str):
    """
    Get detailed information about a specific face.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug(f"Fetching details for target: {target}")
        
        # Check if face exists
        if target not in face_service.get_all_targets():
            raise HTTPException(404, f"Face '{target}' not found")
        
        # Get from database
        db_record = faces_collection.find_one({"target": target})
        
        if not db_record:
            return JSONResponse({
                "status": "success",
                "target": target,
                "exists": True,
                "metadata": None,
                "message": "Face exists but no metadata available"
            })
        
        # Build response
        face_details = {
            "target": target,
            "exists": True,
            "enrolled_at": db_record.get("updated_at"),
            "metadata": db_record.get("metadata", {})
        }
        
        return JSONResponse({
            "status": "success",
            **face_details
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching face details: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get face details: {str(e)}")

# -------------------------------
# EXISTING: Delete a face (ENHANCED)
# -------------------------------
@router.delete("/delete/{target}")
async def delete_face(target: str, delete_logs: bool = Query(False, description="Also delete log files")):
    """
    Remove face from system.
    
    ENHANCED: Added option to delete log files
    """
    try:
        logger.info(f"Deleting face: {target}")
        
        result = face_service.delete_face(target)
        
        if not result["success"]:
            logger.warning(f"Face deletion failed: {result['message']}")
            raise HTTPException(404, result["message"])
        
        # Optionally delete log files (NEW)
        logs_deleted = False
        if delete_logs:
            try:
                import glob
                log_files = glob.glob(f"logs/{target}.*")
                for log_file in log_files:
                    try:
                        os.remove(log_file)
                    except:
                        pass
                logs_deleted = len(log_files) > 0
                logger.info(f"Deleted {len(log_files)} log files for {target}")
            except Exception as log_error:
                logger.warning(f"Failed to delete logs: {log_error}")
        
        logger.info(f"Successfully deleted face: {target}")
        
        return JSONResponse({
            "status": "success",
            "message": result["message"],
            "target": target,
            "logs_deleted": logs_deleted
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to delete face: {str(e)}")

# -------------------------------
# NEW: Update face
# -------------------------------
@router.patch("/update/{target}")
async def update_face(target: str, updates: FaceUpdateRequest):
    """
    Update face metadata or rename face.
    
    **NEW ENDPOINT**
    """
    try:
        logger.info(f"Updating face: {target}")
        
        # Check if face exists
        if target not in face_service.get_all_targets():
            raise HTTPException(404, f"Face '{target}' not found")
        
        # Handle rename
        if updates.new_name and updates.new_name != target:
            # Check if new name already exists
            if updates.new_name in face_service.get_all_targets():
                raise HTTPException(
                    400, 
                    f"Face with name '{updates.new_name}' already exists"
                )
            
            # Rename in database
            try:
                # Get existing encoding
                from app.state import ENCODINGS
                encoding = ENCODINGS.get(target)
                
                if not encoding:
                    raise HTTPException(500, "Failed to retrieve face encoding")
                
                # Delete old
                face_service.delete_face(target)
                
                # Store with new name
                import numpy as np
                store_result = face_service.store_face(updates.new_name, np.array(encoding))
                
                if not store_result["success"]:
                    raise HTTPException(500, f"Failed to rename: {store_result['message']}")
                
                logger.info(f"Renamed face from {target} to {updates.new_name}")
                
                target = updates.new_name  # Update target for metadata update
            
            except Exception as rename_error:
                logger.error(f"Rename failed: {rename_error}")
                raise HTTPException(500, f"Rename failed: {str(rename_error)}")
        
        # Update metadata
        if updates.metadata:
            try:
                faces_collection.update_one(
                    {"target": target},
                    {
                        "$set": {
                            "metadata.custom": updates.metadata,
                            "metadata.last_updated": datetime.now().isoformat()
                        }
                    }
                )
                logger.info(f"Updated metadata for {target}")
            except Exception as meta_error:
                logger.warning(f"Failed to update metadata: {meta_error}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Face '{target}' updated successfully",
            "target": target,
            "changes": {
                "renamed": updates.new_name is not None,
                "metadata_updated": updates.metadata is not None
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating face: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to update face: {str(e)}")

# -------------------------------
# EXISTING: Compare two faces (ENHANCED)
# -------------------------------
@router.post("/compare")
async def compare_faces_endpoint(
    file: UploadFile = File(...),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="Matching threshold"),
    top_k: int = Query(5, ge=1, le=50, description="Return top K matches")
):
    """
    Compare uploaded face against all stored faces.
    
    ENHANCED: Added threshold and top_k parameters
    """
    try:
        logger.info(f"Comparing face from: {file.filename}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(400, "Only image files are allowed")
        
        # Save temp file (EXISTING)
        temp_path = os.path.join(UPLOAD_DIR, f"compare_{file.filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        try:
            # Encode face (EXISTING)
            encode_result = face_service.encode_face(temp_path)
            
            if not encode_result["success"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": encode_result["message"]
                    }
                )
            
            # Check for multiple faces (ENHANCED)
            if encode_result["face_count"] > 1:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "warning",
                        "message": f"Multiple faces detected ({encode_result['face_count']}). Using first face for comparison.",
                        "face_count": encode_result["face_count"]
                    }
                )
            
            # Compare against stored faces (EXISTING - enhanced with threshold)
            encoding = encode_result["encodings"][0]
            
            # Set custom threshold (NEW)
            original_tolerance = face_service.tolerance
            face_service.tolerance = threshold
            
            comparisons = face_service.compare_faces(encoding, return_distances=True)
            
            # Restore original threshold
            face_service.tolerance = original_tolerance
            
            # Filter and sort (ENHANCED)
            matches = [c for c in comparisons if c["match"]]
            matches = matches[:top_k]  # Get top K
            
            logger.info(f"Found {len(matches)} matches (threshold={threshold})")
            
            return JSONResponse({
                "status": "success",
                "filename": file.filename,
                "threshold": threshold,
                "total_faces_checked": len(comparisons),
                "matches_found": len(matches),
                "top_matches": matches,
                "all_comparisons": comparisons if not matches else None
            })
        
        finally:
            # Clean up (EXISTING)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face comparison: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# -------------------------------
# NEW: Search faces
# -------------------------------
@router.get("/search")
async def search_faces(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results")
):
    """
    Search faces by name pattern.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug(f"Searching faces with query: {query}")
        
        targets = face_service.get_all_targets()
        
        # Case-insensitive search
        query_lower = query.lower()
        matches = [t for t in targets if query_lower in t.lower()]
        
        # Apply limit
        matches = matches[:limit]
        
        matches.sort()
        
        return JSONResponse({
            "status": "success",
            "query": query,
            "matches_found": len(matches),
            "total_faces": len(targets),
            "matches": matches
        })
    
    except Exception as e:
        logger.error(f"Error searching faces: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Search failed: {str(e)}")

# -------------------------------
# NEW: Get face statistics
# -------------------------------
@router.get("/stats")
async def get_face_stats():
    """
    Get face enrollment statistics.
    
    **NEW ENDPOINT**
    """
    try:
        logger.debug("Fetching face statistics")
        
        targets = face_service.get_all_targets()
        
        # Get quality distribution
        quality_distribution = {
            "excellent": 0,  # >= 80
            "good": 0,       # >= 60
            "acceptable": 0  # >= 50
        }
        
        total_with_metadata = 0
        
        for target in targets:
            try:
                db_record = faces_collection.find_one({"target": target})
                if db_record and "metadata" in db_record:
                    total_with_metadata += 1
                    quality = db_record["metadata"].get("quality_score", 0)
                    
                    if quality >= 80:
                        quality_distribution["excellent"] += 1
                    elif quality >= 60:
                        quality_distribution["good"] += 1
                    elif quality >= 50:
                        quality_distribution["acceptable"] += 1
            except:
                continue
        
        stats = {
            "total_faces": len(targets),
            "faces_with_metadata": total_with_metadata,
            "quality_distribution": quality_distribution,
            "storage_info": {
                "upload_directory": UPLOAD_DIR,
                "upload_directory_size_mb": get_directory_size(UPLOAD_DIR)
            }
        }
        return JSONResponse({
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error fetching face stats: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get statistics: {str(e)}")

# -------------------------------
# NEW: Helper function for directory size
# -------------------------------
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
# NEW: Find similar faces
# -------------------------------
@router.get("/similar/{target}")
async def find_similar_faces(
    target: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Similarity threshold"),
    limit: int = Query(5, ge=1, le=20, description="Maximum similar faces")
):
    """
    Find faces similar to a specific target.
    
    **NEW ENDPOINT**
    
    Useful for detecting duplicates or grouping similar faces.
    """
    try:
        logger.debug(f"Finding similar faces to: {target}")
        
        # Check if target exists
        if target not in face_service.get_all_targets():
            raise HTTPException(404, f"Face '{target}' not found")
        
        # Get target encoding
        from app.state import ENCODINGS
        target_encoding = ENCODINGS.get(target)
        
        if not target_encoding:
            raise HTTPException(500, "Failed to retrieve face encoding")
        
        # Compare against all other faces
        import numpy as np
        target_enc = np.array(target_encoding)
        
        # Set custom threshold
        original_tolerance = face_service.tolerance
        face_service.tolerance = threshold
        
        comparisons = face_service.compare_faces(target_enc, return_distances=True)
        
        # Restore threshold
        face_service.tolerance = original_tolerance
        
        # Filter out the target itself and non-matches
        similar = [
            c for c in comparisons 
            if c["target"] != target and c["match"]
        ]
        
        # Sort by distance (most similar first)
        similar.sort(key=lambda x: x["distance"])
        
        # Apply limit
        similar = similar[:limit]
        
        logger.info(f"Found {len(similar)} similar faces to {target}")
        
        return JSONResponse({
            "status": "success",
            "target": target,
            "threshold": threshold,
            "similar_faces_found": len(similar),
            "similar_faces": similar
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar faces: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to find similar faces: {str(e)}")

# -------------------------------
# NEW: Export face database
# -------------------------------
@router.get("/export")
async def export_face_database(
    format: str = Query("json", description="Export format (json/csv)"),
    include_encodings: bool = Query(False, description="Include face encodings (large file)")
):
    """
    Export face database to file.
    
    **NEW ENDPOINT**
    
    Supports JSON and CSV formats.
    """
    try:
        logger.info(f"Exporting face database (format={format})")
        
        targets = face_service.get_all_targets()
        
        # Gather data
        export_data = []
        
        for target in targets:
            face_data = {"name": target}
            
            # Get metadata
            try:
                db_record = faces_collection.find_one({"target": target})
                if db_record:
                    if "metadata" in db_record:
                        face_data["metadata"] = db_record["metadata"]
                    if "updated_at" in db_record:
                        face_data["enrolled_at"] = db_record["updated_at"]
                    
                    # Optionally include encodings
                    if include_encodings:
                        from app.state import ENCODINGS
                        encoding = ENCODINGS.get(target)
                        if encoding:
                            face_data["encoding_length"] = len(encoding)
            except Exception as db_error:
                logger.warning(f"Failed to get data for {target}: {db_error}")
            
            export_data.append(face_data)
        
        filename = f"face_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if format == "json":
            # Export as JSON
            export_json = {
                "export_time": datetime.now().isoformat(),
                "total_faces": len(export_data),
                "includes_encodings": include_encodings,
                "faces": export_data
            }
            
            json_bytes = json.dumps(export_json, indent=2).encode('utf-8')
            file_stream = BytesIO(json_bytes)
            
            return StreamingResponse(
                file_stream,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.json"
                }
            )
        
        elif format == "csv":
            # Export as CSV
            import csv
            from io import StringIO
            
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write header
            writer.writerow(["Name", "Enrolled At", "Quality Score", "File Size KB", "Resolution"])
            
            # Write data
            for face in export_data:
                metadata = face.get("metadata", {})
                writer.writerow([
                    face["name"],
                    face.get("enrolled_at", "N/A"),
                    metadata.get("quality_score", "N/A"),
                    metadata.get("file_size_bytes", 0) / 1024 if metadata.get("file_size_bytes") else "N/A",
                    metadata.get("image_resolution", "N/A")
                ])
            
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            file_stream = BytesIO(csv_bytes)
            
            return StreamingResponse(
                file_stream,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.csv"
                }
            )
        
        else:
            raise HTTPException(400, "Invalid format. Use 'json' or 'csv'")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting database: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Export failed: {str(e)}")

# -------------------------------
# NEW: Bulk delete faces
# -------------------------------
@router.post("/delete/bulk")
async def bulk_delete_faces(
    targets: List[str] = Body(..., embed=True, description="List of target names to delete"),
    delete_logs: bool = Query(False, description="Also delete log files")
):
    """
    Delete multiple faces in bulk.
    
    **NEW ENDPOINT**
    
    Body:
```json
    {
        "targets": ["person1", "person2", "person3"]
    }
```
    """
    try:
        if len(targets) == 0:
            raise HTTPException(400, "No targets provided")
        
        if len(targets) > 50:
            raise HTTPException(400, "Maximum 50 faces can be deleted at once")
        
        logger.info(f"Bulk deleting {len(targets)} faces")
        
        results = {
            "deleted": [],
            "not_found": [],
            "errors": []
        }
        
        for target in targets:
            try:
                if target not in face_service.get_all_targets():
                    results["not_found"].append(target)
                    continue
                
                # Delete face
                delete_result = face_service.delete_face(target)
                
                if delete_result["success"]:
                    results["deleted"].append(target)
                    
                    # Optionally delete logs
                    if delete_logs:
                        try:
                            import glob
                            log_files = glob.glob(f"logs/{target}.*")
                            for log_file in log_files:
                                try:
                                    os.remove(log_file)
                                except:
                                    pass
                        except:
                            pass
                else:
                    results["errors"].append({
                        "target": target,
                        "error": delete_result["message"]
                    })
            
            except Exception as delete_error:
                results["errors"].append({
                    "target": target,
                    "error": str(delete_error)
                })
        
        logger.info(f"Bulk delete complete: {len(results['deleted'])} deleted, {len(results['not_found'])} not found, {len(results['errors'])} errors")
        
        return JSONResponse({
            "status": "completed",
            "total_requested": len(targets),
            "deleted": len(results["deleted"]),
            "not_found": len(results["not_found"]),
            "errors": len(results["errors"]),
            "results": results
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk delete: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Bulk delete failed: {str(e)}")

# -------------------------------
# NEW: Validate face image
# -------------------------------
@router.post("/validate")
async def validate_face_image(file: UploadFile = File(...)):
    """
    Validate a face image without enrolling it.
    
    **NEW ENDPOINT**
    
    Checks image quality, face detection, and provides recommendations.
    """
    temp_path = None
    
    try:
        logger.info(f"Validating face image: {file.filename}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse({
                "status": "invalid",
                "valid": False,
                "error": "Not an image file",
                "file_type": file.content_type
            })
        
        # Check file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size == 0:
            return JSONResponse({
                "status": "invalid",
                "valid": False,
                "error": "File is empty"
            })
        
        if file_size > 10 * 1024 * 1024:
            return JSONResponse({
                "status": "invalid",
                "valid": False,
                "error": "File too large (max 10MB)",
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            })
        
        # Save temp file
        temp_path = os.path.join(UPLOAD_DIR, f"validate_{file.filename}")
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Try to encode face
        encode_result = face_service.encode_face(temp_path, return_locations=True)
        
        if not encode_result["success"]:
            return JSONResponse({
                "status": "invalid",
                "valid": False,
                "error": encode_result["message"],
                "suggestion": "Ensure the image contains a clear, front-facing face"
            })
        
        # Get face count
        face_count = encode_result["face_count"]
        
        if face_count == 0:
            return JSONResponse({
                "status": "invalid",
                "valid": False,
                "error": "No faces detected",
                "suggestions": [
                    "Ensure good lighting",
                    "Face should be clearly visible",
                    "Try a different angle"
                ]
            })
        
        if face_count > 1:
            return JSONResponse({
                "status": "warning",
                "valid": True,
                "warning": f"Multiple faces detected ({face_count})",
                "face_count": face_count,
                "suggestion": "Crop the image to contain only one face for best results"
            })
        
        # Assess quality
        encoding = encode_result["encodings"][0]
        face_location = encode_result["locations"][0]
        
        image = face_recognition.load_image_file(temp_path)
        quality = face_service.assess_face_quality(image, face_location)
        
        # Determine validation status
        if quality["score"] >= 60:
            status = "valid"
            message = "Image is suitable for face enrollment"
        elif quality["score"] >= 50:
            status = "acceptable"
            message = "Image quality is acceptable but could be improved"
        else:
            status = "poor"
            message = "Image quality is too low for reliable recognition"
        
        logger.info(f"Validation complete: {status} (quality: {quality['score']:.2f})")
        
        return JSONResponse({
            "status": status,
            "valid": quality["score"] >= 50,
            "message": message,
            "quality": {
                "overall_score": round(quality["score"], 2),
                "size_score": round(quality["size_score"], 2),
                "position_score": round(quality["position_score"], 2),
                "aspect_score": round(quality["aspect_score"], 2),
                "rating": (
                    "excellent" if quality["score"] >= 80 else
                    "good" if quality["score"] >= 60 else
                    "acceptable" if quality["score"] >= 50 else
                    "poor"
                )
            },
            "issues": quality["issues"],
            "metadata": {
                "face_count": face_count,
                "file_size_kb": round(file_size / 1024, 2),
                "resolution": f"{image.shape[1]}x{image.shape[0]}"
            },
            "recommendations": [
                "Use better lighting" if quality["score"] < 70 else None,
                "Center the face in the frame" if quality["position_score"] < 70 else None,
                "Move closer to camera" if quality["size_score"] < 70 else None,
                "Ensure face is not tilted" if quality["aspect_score"] < 80 else None
            ]
        })
    
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "valid": False,
            "error": str(e)
        })
    
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# -------------------------------
# NEW: Health check
# -------------------------------
@router.get("/health")
async def face_health_check():
    """
    Health check for face management service.
    
    **NEW ENDPOINT**
    """
    try:
        # Check if service is operational
        targets = face_service.get_all_targets()
        
        # Check database connectivity
        db_accessible = True
        try:
            faces_collection.find_one()
        except Exception as db_error:
            logger.error(f"Database check failed: {db_error}")
            db_accessible = False
        
        # Check upload directory
        upload_dir_writable = os.access(UPLOAD_DIR, os.W_OK)
        
        # Determine health status
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
            "statistics": {
                "total_faces": len(targets),
                "upload_directory": UPLOAD_DIR
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "service": "face_management",
            "error": str(e)
        }, status_code=503)

# -------------------------------
# NEW: Clear all faces (with confirmation)
# -------------------------------
@router.delete("/clear")
async def clear_all_faces(
    confirmation: str = Query(..., description="Must be 'CONFIRM_DELETE_ALL'")
):
    """
    Clear all enrolled faces from the system.
    
    **NEW ENDPOINT**
    
    **WARNING:** This action cannot be undone!
    
    Requires confirmation parameter: ?confirmation=CONFIRM_DELETE_ALL
    """
    try:
        if confirmation != "CONFIRM_DELETE_ALL":
            raise HTTPException(
                400, 
                "Invalid confirmation. Must provide confirmation=CONFIRM_DELETE_ALL"
            )
        
        logger.warning("CLEARING ALL FACES - This action was confirmed")
        
        targets = face_service.get_all_targets()
        total = len(targets)
        
        deleted = 0
        errors = []
        
        for target in targets:
            try:
                result = face_service.delete_face(target)
                if result["success"]:
                    deleted += 1
                else:
                    errors.append({"target": target, "error": result["message"]})
            except Exception as delete_error:
                errors.append({"target": target, "error": str(delete_error)})
        
        logger.warning(f"Cleared {deleted} faces, {len(errors)} errors")
        
        return JSONResponse({
            "status": "completed",
            "message": f"Cleared {deleted} of {total} faces",
            "deleted": deleted,
            "errors": len(errors),
            "error_details": errors if errors else None
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing faces: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to clear faces: {str(e)}")
```

---

### **📝 Summary of Changes for `face.py`:**

#### **✅ Added (20 new features):**

1. **Comprehensive Logging** - Complete logging for all operations
2. **File Size Validation** - Check file size before processing (max 10MB)
3. **Override Option** - Replace existing faces with override=true
4. **Quality Threshold** - Configurable minimum quality (default 50)
5. **Duplicate Detection** - Prevent accidental overwrites
6. **Metadata Storage** - Store detailed face metadata in database
7. **Batch Upload** - POST `/upload/batch` - Upload multiple faces at once
8. **Enhanced List** - Metadata, sorting, pagination options
9. **Face Details** - GET `/detail/{target}` - Get specific face info
10. **Update Face** - PATCH `/update/{target}` - Rename or update metadata
11. **Delete Logs Option** - Optionally delete log files when deleting face
12. **Search Faces** - GET `/search` - Search by name pattern
13. **Face Statistics** - GET `/stats` - Enrollment statistics
14. **Find Similar** - GET `/similar/{target}` - Find similar faces
15. **Export Database** - GET `/export` - Export to JSON/CSV
16. **Bulk Delete** - POST `/delete/bulk` - Delete multiple faces
17. **Validate Image** - POST `/validate` - Validate before enrolling
18. **Health Check** - GET `/health` - Service health status
19. **Clear All** - DELETE `/clear` - Clear entire database (with confirmation)
20. **Enhanced Compare** - Configurable threshold and top-K results

#### **🔒 Nothing Removed:**
- All original endpoints intact (`/upload`, `/list`, `/delete`, `/compare`)
- All original processing logic preserved
- Backward compatible with existing frontend
- All existing function signatures maintained

#### **🎯 Key Benefits:**

**Reliability:**
- ✅ File size validation (prevents memory issues)
- ✅ Duplicate detection (prevents accidental overwrites)
- ✅ Quality thresholds (ensures good recognition)
- ✅ Comprehensive error handling

**Productivity:**
- ✅ Batch upload (process multiple faces at once)
- ✅ Bulk delete (cleanup multiple faces)
- ✅ Search functionality (find faces quickly)
- ✅ Similar face detection (find duplicates)

**Data Management:**
- ✅ Metadata storage (track quality, timestamps, etc.)
- ✅ Export functionality (JSON/CSV formats)
- ✅ Face statistics (monitor system usage)
- ✅ Update/rename faces (maintain database)

**Quality Control:**
- ✅ Image validation (test before enrolling)
- ✅ Quality assessment (detailed scoring)
- ✅ Recommendations (improve image quality)

**Safety:**
- ✅ Confirmation required for destructive operations
- ✅ Optional log deletion (keep or remove)
- ✅ Detailed error messages

---

### **📊 New API Endpoints Summary:**
```
POST   /face/upload              ✅ ENHANCED - Added override, quality threshold
POST   /face/upload/batch        🆕 NEW - Batch upload multiple faces
GET    /face/list                ✅ ENHANCED - Added metadata, sorting, pagination
GET    /face/detail/{target}     🆕 NEW - Get specific face details
PATCH  /face/update/{target}     🆕 NEW - Update face metadata or rename
DELETE /face/delete/{target}     ✅ ENHANCED - Added log deletion option
POST   /face/compare             ✅ ENHANCED - Added threshold and top-K
GET    /face/search              🆕 NEW - Search faces by name
GET    /face/stats               🆕 NEW - Face enrollment statistics
GET    /face/similar/{target}    🆕 NEW - Find similar faces
GET    /face/export              🆕 NEW - Export database (JSON/CSV)
POST   /face/delete/bulk         🆕 NEW - Bulk delete faces
POST   /face/validate            🆕 NEW - Validate image before enrolling
GET    /face/health              🆕 NEW - Health check
DELETE /face/clear               🆕 NEW - Clear all faces (with confirmation)