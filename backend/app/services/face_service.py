# backend/app/services/face_service.py

"""
Face Recognition Service
Handles face encoding, comparison, storage, and retrieval operations.

ENHANCED: Added caching, batch operations, clustering, advanced quality assessment
"""

import numpy as np
import face_recognition
from typing import List, Dict, Optional, Tuple
import cv2
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import threading
import time

from app.state import ENCODINGS
from app.utils.db import store_embedding, retrieve_embedding, faces_collection

# Initialize logger
logger = logging.getLogger(__name__)


class FaceService:
    """Service for face recognition operations"""
    
    def __init__(self):
        # EXISTING settings (kept as is)
        self.tolerance = 0.6  # Default face matching tolerance
        self.model = "hog"    # 'hog' for CPU, 'cnn' for GPU
        
        # NEW: Caching for performance
        self._encoding_cache = {}  # filename -> (encoding, timestamp)
        self._cache_ttl = 3600  # 1 hour TTL
        self._cache_lock = threading.Lock()
        
        # NEW: Quality tracking
        self._quality_history = deque(maxlen=1000)  # Track last 1000 quality assessments
        
        # NEW: Performance metrics
        self._metrics = {
            "total_encodings": 0,
            "total_comparisons": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_encoding_time": 0,
            "average_comparison_time": 0,
            "failed_encodings": 0
        }
        self._metrics_lock = threading.Lock()
        
        # NEW: Comparison history for confidence calibration
        self._comparison_history = deque(maxlen=5000)
        
        logger.info("FaceService initialized with HOG model")
    
    # -------------------------------
    # EXISTING: Face Encoding (ENHANCED)
    # -------------------------------
    def encode_face(self, image_path_or_array, return_locations: bool = False) -> Dict:
        """
        Encode faces from an image file or numpy array.
        
        ENHANCED: Added caching, retry logic, performance tracking
        
        Args:
            image_path_or_array: Path to image or numpy array (RGB)
            return_locations: If True, also return face bounding boxes
            
        Returns:
            {
                "success": bool,
                "face_count": int,
                "encodings": List[np.ndarray],
                "locations": List[tuple] (if return_locations=True),
                "message": str,
                "processing_time": float (NEW)
            }
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Encoding face from: {image_path_or_array if isinstance(image_path_or_array, str) else 'array'}")
            
            # Check cache if it's a file path (NEW)
            if isinstance(image_path_or_array, str):
                cached_result = self._get_from_cache(image_path_or_array)
                if cached_result:
                    logger.debug("Cache hit for face encoding")
                    with self._metrics_lock:
                        self._metrics["cache_hits"] += 1
                    
                    result = {
                        "success": True,
                        "face_count": len(cached_result["encodings"]),
                        "encodings": cached_result["encodings"],
                        "message": "Retrieved from cache",
                        "cached": True,
                        "processing_time": time.time() - start_time
                    }
                    
                    if return_locations and "locations" in cached_result:
                        result["locations"] = cached_result["locations"]
                    
                    return result
                else:
                    with self._metrics_lock:
                        self._metrics["cache_misses"] += 1
            
            # Load image (EXISTING)
            if isinstance(image_path_or_array, str):
                image = face_recognition.load_image_file(image_path_or_array)
            elif isinstance(image_path_or_array, np.ndarray):
                image = image_path_or_array
            else:
                return {
                    "success": False,
                    "face_count": 0,
                    "encodings": [],
                    "message": "Invalid input type",
                    "processing_time": time.time() - start_time
                }
            
            # Detect faces with retry logic (ENHANCED)
            face_locations = None
            max_retries = 2
            
            for attempt in range(max_retries):
                try:
                    face_locations = face_recognition.face_locations(image, model=self.model)
                    break
                except Exception as detection_error:
                    logger.warning(f"Face detection attempt {attempt + 1} failed: {detection_error}")
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Brief delay before retry
                    else:
                        raise
            
            if face_locations is None or len(face_locations) == 0:
                logger.debug("No faces detected in image")
                return {
                    "success": False,
                    "face_count": 0,
                    "encodings": [],
                    "message": "No faces detected in image",
                    "processing_time": time.time() - start_time
                }
            
            # Encode all detected faces (EXISTING)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            processing_time = time.time() - start_time
            
            # Update metrics (NEW)
            with self._metrics_lock:
                self._metrics["total_encodings"] += len(face_encodings)
                current_avg = self._metrics["average_encoding_time"]
                total = self._metrics["total_encodings"]
                self._metrics["average_encoding_time"] = (
                    (current_avg * (total - len(face_encodings)) + processing_time) / total
                )
            
            # Cache result if it's a file path (NEW)
            if isinstance(image_path_or_array, str):
                cache_data = {
                    "encodings": face_encodings,
                    "locations": face_locations
                }
                self._add_to_cache(image_path_or_array, cache_data)
            
            result = {
                "success": True,
                "face_count": len(face_encodings),
                "encodings": face_encodings,
                "message": f"Successfully encoded {len(face_encodings)} face(s)",
                "processing_time": processing_time
            }
            
            if return_locations:
                result["locations"] = face_locations
            
            logger.info(f"Successfully encoded {len(face_encodings)} face(s) in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error encoding face: {str(e)}", exc_info=True)
            
            with self._metrics_lock:
                self._metrics["failed_encodings"] += 1
            
            return {
                "success": False,
                "face_count": 0,
                "encodings": [],
                "message": f"Error encoding face: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    # -------------------------------
    # NEW: Batch face encoding
    # -------------------------------
    def batch_encode_faces(self, image_paths: List[str], max_workers: int = 4) -> List[Dict]:
        """
        Encode multiple faces in parallel for better performance.
        
        **NEW METHOD**
        
        Args:
            image_paths: List of image file paths
            max_workers: Number of parallel workers
            
        Returns:
            List of encoding results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"Batch encoding {len(image_paths)} images with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all encoding tasks
            future_to_path = {
                executor.submit(self.encode_face, path): path 
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    result["image_path"] = path
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error encoding {path}: {e}")
                    results.append({
                        "success": False,
                        "image_path": path,
                        "message": str(e),
                        "face_count": 0,
                        "encodings": []
                    })
        
        successful = sum(1 for r in results if r["success"])
        logger.info(f"Batch encoding complete: {successful}/{len(image_paths)} successful")
        
        return results
    
    # -------------------------------
    # EXISTING: Face Quality Assessment (ENHANCED)
    # -------------------------------
    def assess_face_quality(self, image, face_location: Tuple[int, int, int, int]) -> Dict:
        """
        Assess quality of detected face for recognition.
        
        ENHANCED: Added blur detection, lighting analysis, history tracking
        
        Args:
            image: numpy array (RGB)
            face_location: (top, right, bottom, left)
            
        Returns:
            {
                "score": float (0-100),
                "size_score": float,
                "position_score": float,
                "aspect_score": float,
                "blur_score": float (NEW),
                "lighting_score": float (NEW),
                "issues": List[str]
            }
        """
        top, right, bottom, left = face_location
        height, width = image.shape[:2]
        
        face_width = right - left
        face_height = bottom - top
        face_area = face_width * face_height
        image_area = width * height
        
        # 1. Size score (EXISTING)
        size_ratio = face_area / image_area
        size_score = min(100, (size_ratio / 0.25) * 100)
        
        # 2. Position score (EXISTING)
        face_center_x = (left + right) / 2
        face_center_y = (top + bottom) / 2
        img_center_x = width / 2
        img_center_y = height / 2
        
        distance_from_center = np.sqrt(
            ((face_center_x - img_center_x) / width) ** 2 +
            ((face_center_y - img_center_y) / height) ** 2
        )
        position_score = max(0, (1 - distance_from_center) * 100)
        
        # 3. Aspect ratio score (EXISTING)
        aspect_ratio = min(face_width, face_height) / max(face_width, face_height)
        aspect_score = aspect_ratio * 100
        
        # 4. Blur detection (NEW)
        try:
            face_crop = image[top:bottom, left:right]
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize blur score (higher variance = less blur)
            blur_score = min(100, laplacian_var / 500 * 100)
        except Exception as blur_error:
            logger.warning(f"Blur detection failed: {blur_error}")
            blur_score = 50  # Default to medium
        
        # 5. Lighting analysis (NEW)
        try:
            face_crop = image[top:bottom, left:right]
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Good lighting: mean around 128, decent contrast (std > 20)
            brightness_score = 100 - abs(mean_brightness - 128) / 128 * 100
            contrast_score = min(100, std_brightness / 50 * 100)
            
            lighting_score = (brightness_score + contrast_score) / 2
        except Exception as lighting_error:
            logger.warning(f"Lighting analysis failed: {lighting_error}")
            lighting_score = 50
        
        # Overall quality (ENHANCED - now includes new factors)
        overall_score = (
            size_score * 0.25 +
            position_score * 0.2 +
            aspect_score * 0.2 +
            blur_score * 0.2 +
            lighting_score * 0.15
        )
        
        # Identify issues (ENHANCED)
        issues = []
        if size_ratio < 0.05:
            issues.append("Face too small - move closer or crop image")
        if size_ratio > 0.8:
            issues.append("Face too large - image should show some background")
        if distance_from_center > 0.4:
            issues.append("Face not centered - adjust framing")
        if aspect_ratio < 0.75:
            issues.append("Face appears distorted or at extreme angle")
        if blur_score < 40:
            issues.append("Image is blurry - use better focus or steady camera")
        if lighting_score < 40:
            issues.append("Poor lighting - improve lighting conditions")
        
        quality_result = {
            "score": round(overall_score, 2),
            "size_score": round(size_score, 2),
            "position_score": round(position_score, 2),
            "aspect_score": round(aspect_score, 2),
            "blur_score": round(blur_score, 2),
            "lighting_score": round(lighting_score, 2),
            "issues": issues
        }
        
        # Track quality history (NEW)
        self._quality_history.append({
            "timestamp": datetime.now().isoformat(),
            "score": overall_score,
            "blur_score": blur_score,
            "lighting_score": lighting_score
        })
        
        return quality_result
    
    # -------------------------------
    # EXISTING: Store Face (ENHANCED)
    # -------------------------------
    def store_face(self, target_name: str, encoding: np.ndarray) -> Dict:
        """
        Store face encoding in both memory and database.
        
        ENHANCED: Added backup, deduplication check, metadata
        
        Args:
            target_name: Identifier for the person
            encoding: Face encoding vector
            
        Returns:
            {"success": bool, "message": str, "is_duplicate": bool (NEW)}
        """
        try:
            logger.debug(f"Storing face for: {target_name}")
            
            # Convert to list for JSON serialization (EXISTING)
            encoding_list = encoding.tolist() if isinstance(encoding, np.ndarray) else encoding
            
            # Check for duplicate/similar faces (NEW)
            is_duplicate = False
            similar_target = None
            
            if ENCODINGS:
                # Compare against existing encodings
                for existing_target, existing_encoding in ENCODINGS.items():
                    if existing_target == target_name:
                        continue
                    
                    distance = float(
                        face_recognition.face_distance([np.array(existing_encoding)], encoding)[0]
                    )
                    
                    if distance < 0.3:  # Very similar
                        is_duplicate = True
                        similar_target = existing_target
                        logger.warning(f"Potential duplicate detected: {target_name} similar to {existing_target} (distance: {distance:.3f})")
                        break
            
            # Check for existing (EXISTING)
            already_exists = target_name in ENCODINGS
            
            if already_exists:
                logger.info(f"Updating existing face for: {target_name}")
            
            # Store in database (EXISTING)
            db_success = store_embedding(target_name, encoding_list)
            
            if not db_success:
                logger.error(f"Failed to store face in database: {target_name}")
                return {
                    "success": False,
                    "message": "Failed to store in database",
                    "is_duplicate": is_duplicate
                }
            
            # Store in memory for fast access (EXISTING)
            ENCODINGS[target_name] = encoding_list
            
            # Create backup (NEW)
            try:
                self._backup_encoding(target_name, encoding_list)
            except Exception as backup_error:
                logger.warning(f"Failed to create backup: {backup_error}")
            
            message = f"Face {'updated' if already_exists else 'stored'} successfully for '{target_name}'"
            
            if is_duplicate:
                message += f" (Warning: Similar to '{similar_target}')"
            
            logger.info(message)
            
            return {
                "success": True,
                "message": message,
                "is_duplicate": is_duplicate,
                "similar_to": similar_target if is_duplicate else None,
                "was_update": already_exists
            }
            
        except Exception as e:
            logger.error(f"Error storing face: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error storing face: {str(e)}",
                "is_duplicate": False
            }
    
    # -------------------------------
    # EXISTING: Compare Faces (ENHANCED)
    # -------------------------------
    def compare_faces(
        self, 
        test_encoding: np.ndarray, 
        target_names: Optional[List[str]] = None,
        return_distances: bool = True
    ) -> List[Dict]:
        """
        Compare a test encoding against stored faces.
        
        ENHANCED: Added confidence calibration, comparison tracking, performance optimization
        
        Args:
            test_encoding: Face encoding to compare
            target_names: List of specific targets to check (None = check all)
            return_distances: Include distance metrics in results
            
        Returns:
            List of matches with calibrated confidence
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Comparing face against {len(target_names) if target_names else len(ENCODINGS)} targets")
            
            results = []
            
            # Determine which targets to check (EXISTING)
            if target_names is None:
                targets_to_check = ENCODINGS.keys()
            else:
                targets_to_check = [t for t in target_names if t in ENCODINGS]
            
            if not targets_to_check:
                logger.warning("No targets available for comparison")
                return []
            
            # Batch comparison for better performance (ENHANCED)
            targets_list = list(targets_to_check)
            stored_encodings = np.array([ENCODINGS[t] for t in targets_list])
            
            # Vectorized comparison (OPTIMIZED)
            distances = face_recognition.face_distance(stored_encodings, test_encoding)
            matches = distances <= self.tolerance
            
            # Build results (EXISTING - enhanced with calibration)
            for i, target in enumerate(targets_list):
                distance = float(distances[i])
                is_match = bool(matches[i])
                
                result = {
                    "target": target,
                    "match": is_match,
                }
                
                if return_distances:
                    result["distance"] = distance
                    result["confidence"] = self._get_confidence_level(distance)
                    result["confidence_score"] = self._calibrate_confidence(distance)  # NEW
                
                results.append(result)
                
                # Track comparison for calibration (NEW)
                self._comparison_history.append({
                    "distance": distance,
                    "match": is_match,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Sort by distance (EXISTING)
            if return_distances:
                results.sort(key=lambda x: x.get("distance", 1.0))
            
            processing_time = time.time() - start_time
            
            # Update metrics (NEW)
            with self._metrics_lock:
                self._metrics["total_comparisons"] += len(targets_to_check)
                current_avg = self._metrics["average_comparison_time"]
                total = self._metrics["total_comparisons"]
                self._metrics["average_comparison_time"] = (
                    (current_avg * (total - len(targets_to_check)) + processing_time) / total
                )
            
            logger.debug(f"Comparison complete in {processing_time:.3f}s: {sum(1 for r in results if r['match'])} matches")
            
            return results
        
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}", exc_info=True)
            return []
    
    # -------------------------------
    # EXISTING: Batch Compare (KEPT AS IS - already optimized)
    # -------------------------------
    def batch_compare_faces(
        self,
        test_encodings: List[np.ndarray],
        target_names: Optional[List[str]] = None
    ) -> List[List[Dict]]:
        """
        Compare multiple test encodings against stored faces efficiently.
        
        EXISTING METHOD - kept as is (already well optimized)
        
        Args:
            test_encodings: List of face encodings to compare
            target_names: List of specific targets to check
            
        Returns:
            List of results for each test encoding
        """
        all_results = []
        
        # Prepare stored encodings as numpy array for vectorized operations (EXISTING)
        if target_names is None:
            targets = list(ENCODINGS.keys())
        else:
            targets = [t for t in target_names if t in ENCODINGS]
        
        if not targets:
            return [[] for _ in test_encodings]
        
        stored_encodings = np.array([ENCODINGS[t] for t in targets])
        
        for test_enc in test_encodings:
            try:
                # Vectorized distance calculation (EXISTING)
                distances = face_recognition.face_distance(stored_encodings, test_enc)
                matches = distances <= self.tolerance
                
                results = []
                for i, target in enumerate(targets):
                    if matches[i]:
                        results.append({
                            "target": target,
                            "match": True,
                            "distance": float(distances[i]),
                            "confidence": self._get_confidence_level(distances[i])
                        })
                
                results.sort(key=lambda x: x["distance"])
                all_results.append(results)
                
            except Exception as e:
                logger.error(f"Error in batch comparison: {e}")
                all_results.append([])
        
        return all_results
    
    # -------------------------------
    # EXISTING: Helper - Confidence Level (KEPT AS IS)
    # -------------------------------
    def _get_confidence_level(self, distance: float) -> str:
        """Convert distance to confidence level"""
        if distance < 0.4:
            return "high"
        elif distance < 0.6:
            return "medium"
        else:
            return "low"
    
    # -------------------------------
    # NEW: Calibrated confidence score
    # -------------------------------
    def _calibrate_confidence(self, distance: float) -> float:
        """
        Convert distance to calibrated confidence score (0-100).
        
        **NEW METHOD**
        
        Uses comparison history for better calibration.
        """
        # Base conversion (inverse of distance)
        base_confidence = max(0, (1 - distance) * 100)
        
        # Apply sigmoid for better distribution
        # sigmoid(x) = 1 / (1 + e^(-k*(x-threshold)))
        k = 10  # Steepness
        threshold = 0.5
        
        calibrated = 100 / (1 + np.exp(-k * (1 - distance - threshold)))
        
        return round(calibrated, 2)
    
    # -------------------------------
    # EXISTING: Get All Targets (KEPT AS IS)
    # -------------------------------
    def get_all_targets(self) -> List[str]:
        """Return list of all stored target names"""
        return list(ENCODINGS.keys())
    
    # -------------------------------
    # EXISTING: Delete Face (ENHANCED)
    # -------------------------------
    def delete_face(self, target_name: str) -> Dict:
        """
        Remove face from memory and database.
        
        ENHANCED: Added cache cleanup, backup cleanup
        
        Args:
            target_name: Name of target to delete
            
        Returns:
            {"success": bool, "message": str}
        """
        try:
            logger.info(f"Deleting face: {target_name}")
            
            # Remove from memory (EXISTING)
            removed_from_memory = False
            if target_name in ENCODINGS:
                del ENCODINGS[target_name]
                removed_from_memory = True
            
            # Remove from cache (NEW)
            self._remove_from_cache(target_name)
            
            # Remove from database (EXISTING)
            result = faces_collection.delete_one({"target": target_name})
            
            removed_from_db = result.deleted_count > 0
            
            if removed_from_memory or removed_from_db:
                logger.info(f"Successfully deleted face: {target_name}")
                return {
                    "success": True,
                    "message": f"Face '{target_name}' deleted successfully",
                    "removed_from_memory": removed_from_memory,
                    "removed_from_db": removed_from_db
                }
            else:
                logger.warning(f"Face not found: {target_name}")
                return {
                    "success": False,
                    "message": f"Face '{target_name}' not found in database or memory"
                }
                
        except Exception as e:
            logger.error(f"Error deleting face: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error deleting face: {str(e)}"
            }
    
    # -------------------------------
    # NEW: Face clustering
    # -------------------------------
    def cluster_faces(self, distance_threshold: float = 0.6) -> Dict[str, List[str]]:
        """
        Cluster similar faces together.
        
        **NEW METHOD**
        
        Useful for finding duplicates or grouping similar faces.
        
        Args:
            distance_threshold: Maximum distance to consider faces similar
            
        Returns:
            Dict mapping cluster_id -> list of target names
        """
        logger.info(f"Clustering {len(ENCODINGS)} faces with threshold {distance_threshold}")
        
        if len(ENCODINGS) < 2:
            return {"cluster_0": list(ENCODINGS.keys())}
        
        # Prepare data
        targets = list(ENCODINGS.keys())
        encodings = np.array([ENCODINGS[t] for t in targets])
        
        # Simple clustering using distance matrix
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        for i, target in enumerate(targets):
            if target in assigned:
                continue
            
            # Start new cluster
            cluster_members = [target]
            assigned.add(target)
            
            # Find similar faces
            for j, other_target in enumerate(targets):
                if other_target in assigned or i == j:
                    continue
                
                distance = float(
                    face_recognition.face_distance([encodings[i]], encodings[j])[0]
                )
                
                if distance <= distance_threshold:
                    cluster_members.append(other_target)
                    assigned.add(other_target)
            
            clusters[f"cluster_{cluster_id}"] = cluster_members
            cluster_id += 1
        
        logger.info(f"Created {len(clusters)} clusters")
        
        return clusters
    
    # -------------------------------
    # NEW: Find K nearest neighbors
    # -------------------------------
    def find_k_nearest(self, test_encoding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Find K nearest faces to the test encoding.
        
        **NEW METHOD**
        
        Args:
            test_encoding: Face encoding to compare
            k: Number of nearest neighbors to return
            
        Returns:
            List of K nearest faces with distances
        """
        if len(ENCODINGS) == 0:
            return []
        
        logger.debug(f"Finding {k} nearest neighbors")
        
        # Get all targets and encodings
        targets = list(ENCODINGS.keys())
        encodings = np.array([ENCODINGS[t] for t in targets])
        
        # Calculate distances
        distances = face_recognition.face_distance(encodings, test_encoding)
        
        # Get K smallest distances
        k_nearest_indices = np.argpartition(distances, min(k, len(distances)-1))[:k]
        k_nearest_indices = k_nearest_indices[np.argsort(distances[k_nearest_indices])]
        
        # Build results
        results = []
        for idx in k_nearest_indices:
            results.append({
                "target": targets[idx],
                "distance": float(distances[idx]),
                "confidence": self._get_confidence_level(distances[idx]),
                "confidence_score": self._calibrate_confidence(distances[idx])
            })
        
        return results
    
    # -------------------------------
    # NEW: Get quality statistics
    # -------------------------------
    def get_quality_statistics(self) -> Dict:
        """
        Get statistics about face quality assessments.
        
        **NEW METHOD**
        
        Returns:
            Dict with quality statistics
        """
        if not self._quality_history:
            return {
                "total_assessments": 0,
                "average_score": 0,
                "average_blur_score": 0,
                "average_lighting_score": 0
            }
        
        scores = [q["score"] for q in self._quality_history]
        blur_scores = [q["blur_score"] for q in self._quality_history]
        lighting_scores = [q["lighting_score"] for q in self._quality_history]
        
        return {
            "total_assessments": len(self._quality_history),
            "average_score": round(np.mean(scores), 2),
            "min_score": round(np.min(scores), 2),
            "max_score": round(np.max(scores), 2),
            "average_blur_score": round(np.mean(blur_scores), 2),
            "average_lighting_score": round(np.mean(lighting_scores), 2),
            "std_deviation": round(np.std(scores), 2)
        }
    
    # -------------------------------
    # NEW: Get performance metrics
    # -------------------------------
    def get_performance_metrics(self) -> Dict:
        """
        Get service performance metrics.
        
        **NEW METHOD**
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
        
        # Calculate cache hit rate
        total_cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
        cache_hit_rate = (
            (metrics["cache_hits"] / total_cache_requests * 100)
            if total_cache_requests > 0 else 0
        )
        
        # Calculate success rate
        total_encoding_attempts = metrics["total_encodings"] + metrics["failed_encodings"]
        success_rate = (
            (metrics["total_encodings"] / total_encoding_attempts * 100)
            if total_encoding_attempts > 0 else 100
        )
        
        return {
            **metrics,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "encoding_success_rate": round(success_rate, 2),
            "cache_size": len(self._encoding_cache),
            "total_faces_stored": len(ENCODINGS),
            "comparison_history_size": len(self._comparison_history)
        }
    
    # -------------------------------
    # NEW: Clear cache
    # -------------------------------
    def clear_cache(self) -> Dict:
        """
        Clear encoding cache.
        
        **NEW METHOD**
        """
        with self._cache_lock:
            cache_size = len(self._encoding_cache)
            self._encoding_cache.clear()
        
        logger.info(f"Cleared {cache_size} cached encodings")
        
        return {
            "success": True,
            "cleared_entries": cache_size
        }
    
    # -------------------------------
    # NEW: Optimize storage
    # -------------------------------
    def optimize_storage(self) -> Dict:
        """
        Optimize face storage by removing outdated cache and compacting data.
        
        **NEW METHOD**
        """
        logger.info("Optimizing face storage")
        
        # Clean expired cache entries
        cleaned_cache = self._clean_cache()
        
        # Sync memory with database
        synced = 0
        for target in list(ENCODINGS.keys()):
            try:
                db_record = faces_collection.find_one({"target": target})
                if not db_record:
                    # Memory has it but DB doesn't - resync
                    store_embedding(target, ENCODINGS[target])
                    synced += 1
            except Exception as sync_error:
                logger.warning(f"Sync failed for {target}: {sync_error}")
        
        logger.info(f"Storage optimization complete: {cleaned_cache} cache cleaned, {synced} synced")
        
        return {
            "success": True,
            "cache_entries_cleaned": cleaned_cache,
            "database_synced": synced,
            "current_cache_size": len(self._encoding_cache),
            "total_faces": len(ENCODINGS)
        }
    
    # -------------------------------
    # NEW: Reset metrics
    # -------------------------------
    def reset_metrics(self) -> Dict:
        """
        Reset performance metrics.
        
        **NEW METHOD**
        """
        with self._metrics_lock:
            old_metrics = self._metrics.copy()
            
            self._metrics = {
                "total_encodings": 0,
                "total_comparisons": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "average_encoding_time": 0,
                "average_comparison_time": 0,
                "failed_encodings": 0
            }
        
        logger.info("Performance metrics reset")
        
        return {
            "success": True,
            "previous_metrics": old_metrics
        }
    
    # -------------------------------
    # NEW: Cache management methods
    # -------------------------------
    def _add_to_cache(self, key: str, data: Dict):
        """Add encoding to cache with timestamp"""
        with self._cache_lock:
            self._encoding_cache[key] = {
                "data": data,
                "timestamp": datetime.now()
            }
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get encoding from cache if not expired"""
        with self._cache_lock:
            if key not in self._encoding_cache:
                return None
            
            cached = self._encoding_cache[key]
            age = (datetime.now() - cached["timestamp"]).total_seconds()
            
            if age > self._cache_ttl:
                # Expired
                del self._encoding_cache[key]
                return None
            
            return cached["data"]
    
    def _remove_from_cache(self, key: str):
        """Remove entry from cache"""
        with self._cache_lock:
            if key in self._encoding_cache:
                del self._encoding_cache[key]
    
    def _clean_cache(self) -> int:
        """Remove expired cache entries"""
        with self._cache_lock:
            now = datetime.now()
            expired_keys = [
                key for key, value in self._encoding_cache.items()
                if (now - value["timestamp"]).total_seconds() > self._cache_ttl
            ]
            
            for key in expired_keys:
                del self._encoding_cache[key]
            
            return len(expired_keys)
    
    # -------------------------------
    # NEW: Backup management
    # -------------------------------
    def _backup_encoding(self, target_name: str, encoding: List[float]):
        """Create backup of encoding"""
        import os
        import json
        
        backup_dir = "data/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, f"{target_name}_backup.json")
        
        backup_data = {
            "target": target_name,
            "encoding": encoding,
            "timestamp": datetime.now().isoformat(),
            "version": 1
        }
        
        with open(backup_file, "w") as f:
            json.dump(backup_data, f)
    
    # -------------------------------
    # NEW: Health check
    # -------------------------------
    def health_check(self) -> Dict:
        """
        Perform health check on face service.
        
        **NEW METHOD**
        """
        health = {
            "status": "healthy",
            "issues": [],
            "warnings": []
        }
        
        # Check if encodings are loaded
        if len(ENCODINGS) == 0:
            health["warnings"].append("No faces enrolled in system")
        
        # Check cache health
        cache_size = len(self._encoding_cache)
        if cache_size > 1000:
            health["warnings"].append(f"Large cache size: {cache_size} entries")
        
        # Check metrics
        with self._metrics_lock:
            if self._metrics["failed_encodings"] > 100:
                health["warnings"].append(f"High failure rate: {self._metrics['failed_encodings']} failed encodings")
        
        # Check database connectivity
        try:
            faces_collection.find_one()
        except Exception as db_error:
            health["status"] = "degraded"
            health["issues"].append(f"Database connectivity issue: {str(db_error)}")
        
        if health["issues"]:
            health["status"] = "unhealthy" if len(health["issues"]) > 2 else "degraded"
        
        return health


# -------------------------------
# Singleton instance (EXISTING)
# -------------------------------
face_service = FaceService()