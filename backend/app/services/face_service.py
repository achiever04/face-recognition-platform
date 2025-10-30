# backend/app/services/face_service.py

"""
Face Recognition Service
Handles face encoding, comparison, storage, and retrieval operations.

ENHANCED: Added caching, batch operations, clustering, advanced quality assessment,
robust logging, atomic backups, safer metric calculations, optional dependency handling.
"""

from __future__ import annotations

import os
import io
import json
import time
import threading
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import cv2

# face_recognition is optional — wrap import to avoid hard crash on machines without it.
try:
    import face_recognition  # type: ignore
except Exception:
    face_recognition = None  # We will handle absence gracefully

# App-specific imports (keep same names & behavior)
from app.state import ENCODINGS
from app.utils.db import store_embedding, retrieve_embedding, faces_collection
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)

# Constants (tweakable via env if desired)
DEFAULT_TOLERANCE = float(os.getenv("FACE_TOLERANCE", "0.6"))
DEFAULT_MODEL = os.getenv("FACE_MODEL", "hog")  # 'hog' or 'cnn' (gpu)
CACHE_TTL_SECONDS = int(os.getenv("FACE_CACHE_TTL", "3600"))  # 1 hour default
BATCH_WORKERS = int(os.getenv("FACE_BATCH_WORKERS", "4"))
BACKUP_DIR = Path(os.getenv("FACE_BACKUP_DIR", "data/backups"))
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


class FaceService:
    """Service for face recognition operations."""

    def __init__(self):
        # Matching params (kept from original)
        self.tolerance = DEFAULT_TOLERANCE
        self.model = DEFAULT_MODEL

        # Caching
        self._encoding_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"data":..., "timestamp": datetime}
        self._cache_ttl = CACHE_TTL_SECONDS
        self._cache_lock = threading.RLock()

        # Quality tracking
        self._quality_history = deque(maxlen=1000)

        # Performance metrics (improved math)
        self._metrics_lock = threading.RLock()
        self._metrics = {
            "total_encodings": 0,
            "total_comparisons": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cumulative_encoding_time": 0.0,  # store cumulative to compute average safely
            "cumulative_comparison_time": 0.0,
            "failed_encodings": 0
        }

        # Comparison history for calibration
        self._comparison_history = deque(maxlen=5000)

        logger.info("FaceService initialized (model=%s, tolerance=%.3f)", self.model, self.tolerance)

    # -------------------------------
    # EXISTING: Face Encoding (ENHANCED)
    # -------------------------------
    def encode_face(self, image_path_or_array, return_locations: bool = False) -> Dict[str, Any]:
        """
        Encode faces from an image file path or numpy array (RGB expected).
        Preserves original API but improves caching, retry logic, and metrics.

        Returns dictionary similar to original implementation.
        """
        start_time = time.time()

        # If face_recognition is not available, return a clear error (instead of crashing)
        if face_recognition is None:
            msg = "face_recognition library not installed; encoding unavailable"
            logger.error(msg)
            with self._metrics_lock:
                self._metrics["failed_encodings"] += 1
            return {
                "success": False,
                "face_count": 0,
                "encodings": [],
                "message": msg,
                "processing_time": time.time() - start_time
            }

        try:
            input_desc = image_path_or_array if isinstance(image_path_or_array, str) else "array"
            logger.debug("encode_face called for %s", input_desc)

            # 1) Try cache if a file path
            if isinstance(image_path_or_array, str):
                cached = self._get_from_cache(image_path_or_array)
                if cached:
                    logger.debug("Cache hit for encoding: %s", image_path_or_array)
                    with self._metrics_lock:
                        self._metrics["cache_hits"] += 1
                    result = {
                        "success": True,
                        "face_count": len(cached.get("encodings", [])),
                        "encodings": cached.get("encodings", []),
                        "message": "Retrieved from cache",
                        "cached": True,
                        "processing_time": time.time() - start_time
                    }
                    if return_locations:
                        result["locations"] = cached.get("locations", [])
                    return result
                else:
                    with self._metrics_lock:
                        self._metrics["cache_misses"] += 1

            # 2) Load image
            if isinstance(image_path_or_array, str):
                # face_recognition.load_image_file uses PIL or numpy internally
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

            # 3) Detect locations with retry
            face_locations = None
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    face_locations = face_recognition.face_locations(image, model=self.model)
                    break
                except Exception as detection_error:
                    logger.warning("Face detection attempt %d failed: %s", attempt + 1, detection_error)
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        raise

            if not face_locations:
                logger.debug("No faces detected in the input")
                with self._metrics_lock:
                    self._metrics["failed_encodings"] += 1
                return {
                    "success": False,
                    "face_count": 0,
                    "encodings": [],
                    "message": "No faces detected in image",
                    "processing_time": time.time() - start_time
                }

            # 4) Encode faces
            encode_start = time.time()
            face_encodings = face_recognition.face_encodings(image, face_locations)
            encode_time = time.time() - encode_start

            # 5) Update metrics (safe math)
            with self._metrics_lock:
                self._metrics["total_encodings"] += len(face_encodings)
                self._metrics["cumulative_encoding_time"] += encode_time

            # 6) Cache results if source was path
            if isinstance(image_path_or_array, str):
                cache_payload = {"encodings": face_encodings, "locations": face_locations}
                try:
                    self._add_to_cache(image_path_or_array, cache_payload)
                except Exception:
                    logger.debug("Failed to add encoding to cache (non-fatal)")

            processing_time = time.time() - start_time
            logger.info("Encoded %d face(s) in %.3fs (io+proc=%.3fs)", len(face_encodings), processing_time, encode_time)

            result = {
                "success": True,
                "face_count": len(face_encodings),
                "encodings": face_encodings,
                "message": f"Successfully encoded {len(face_encodings)} face(s)",
                "processing_time": processing_time
            }
            if return_locations:
                result["locations"] = face_locations
            return result

        except Exception as e:
            logger.exception("Error encoding face: %s", e)
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
    # NEW: Batch face encoding (keeps original interface but improved)
    # -------------------------------
    def batch_encode_faces(self, image_paths: List[str], max_workers: int = BATCH_WORKERS) -> List[Dict[str, Any]]:
        logger.info("Batch encoding %d images with %d workers", len(image_paths), max_workers)
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.encode_face, path): path for path in image_paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    r = future.result()
                    r["image_path"] = path
                    results.append(r)
                except Exception as e:
                    logger.exception("Batch encode failed for %s: %s", path, e)
                    results.append({
                        "success": False,
                        "image_path": path,
                        "message": str(e),
                        "face_count": 0,
                        "encodings": []
                    })
        successful = sum(1 for r in results if r.get("success"))
        logger.info("Batch encoding finished: %d/%d successful", successful, len(image_paths))
        return results

    # -------------------------------
    # EXISTING: Face Quality Assessment (ENHANCED)
    # -------------------------------
    def assess_face_quality(self, image, face_location: Tuple[int, int, int, int]) -> Dict[str, Any]:
        top, right, bottom, left = face_location
        height, width = image.shape[:2]

        face_width = max(1, right - left)
        face_height = max(1, bottom - top)
        face_area = float(face_width * face_height)
        image_area = float(width * height)

        # Size score
        size_ratio = face_area / image_area if image_area > 0 else 0.0
        size_score = min(100.0, (size_ratio / 0.25) * 100.0)

        # Position score
        face_center_x = (left + right) / 2.0
        face_center_y = (top + bottom) / 2.0
        img_center_x = width / 2.0
        img_center_y = height / 2.0
        distance_from_center = np.sqrt(((face_center_x - img_center_x) / width) ** 2 + ((face_center_y - img_center_y) / height) ** 2) if width and height else 0.0
        position_score = max(0.0, (1.0 - distance_from_center) * 100.0)

        # Aspect score
        aspect_ratio = min(face_width, face_height) / max(face_width, face_height)
        aspect_score = aspect_ratio * 100.0

        # Blur detection
        try:
            face_crop = image[top:bottom, left:right]
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(100.0, (laplacian_var / 500.0) * 100.0)
        except Exception as be:
            logger.debug("Blur detection error: %s", be)
            blur_score = 50.0

        # Lighting analysis
        try:
            face_crop = image[top:bottom, left:right]
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            mean_brightness = float(np.mean(gray))
            std_brightness = float(np.std(gray))
            brightness_score = 100.0 - abs(mean_brightness - 128.0) / 128.0 * 100.0
            contrast_score = min(100.0, (std_brightness / 50.0) * 100.0)
            lighting_score = (brightness_score + contrast_score) / 2.0
        except Exception as le:
            logger.debug("Lighting analysis error: %s", le)
            lighting_score = 50.0

        overall_score = (
            size_score * 0.25 +
            position_score * 0.2 +
            aspect_score * 0.2 +
            blur_score * 0.2 +
            lighting_score * 0.15
        )

        issues: List[str] = []
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

        # Track quality history
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
    def store_face(self, target_name: str, encoding: np.ndarray) -> Dict[str, Any]:
        logger.debug("Storing face: %s", target_name)
        try:
            encoding_list = encoding.tolist() if isinstance(encoding, np.ndarray) else encoding

            # Duplicate detection (safe if face_recognition available)
            is_duplicate = False
            similar_target = None
            if ENCODINGS and face_recognition is not None:
                for existing_target, existing_encoding in ENCODINGS.items():
                    if existing_target == target_name:
                        continue
                    try:
                        distance = float(face_recognition.face_distance([np.array(existing_encoding)], encoding)[0])
                    except Exception:
                        continue
                    if distance < 0.3:
                        is_duplicate = True
                        similar_target = existing_target
                        logger.warning("Potential duplicate: %s ~ %s (distance=%.3f)", target_name, existing_target, distance)
                        break

            already_exists = target_name in ENCODINGS

            db_success = store_embedding(target_name, encoding_list)
            if not db_success:
                logger.error("Failed to store embedding in DB for %s", target_name)
                return {"success": False, "message": "Failed to store in database", "is_duplicate": is_duplicate}

            # Update in-memory encodings
            ENCODINGS[target_name] = encoding_list

            # Backup atomically
            try:
                self._backup_encoding_atomic(target_name, encoding_list)
            except Exception as be:
                logger.warning("Backup failed for %s: %s", target_name, be)

            message = f"Face {'updated' if already_exists else 'stored'} successfully for '{target_name}'"
            if is_duplicate:
                message += f" (Warning: Similar to '{similar_target}')"
            logger.info(message)

            return {"success": True, "message": message, "is_duplicate": is_duplicate, "similar_to": similar_target if is_duplicate else None, "was_update": already_exists}
        except Exception as e:
            logger.exception("Error storing face %s: %s", target_name, e)
            return {"success": False, "message": f"Error storing face: {str(e)}", "is_duplicate": False}

    # -------------------------------
    # EXISTING: Compare Faces (ENHANCED)
    # -------------------------------
    def compare_faces(self, test_encoding: np.ndarray, target_names: Optional[List[str]] = None, return_distances: bool = True) -> List[Dict[str, Any]]:
        start_time = time.time()

        if face_recognition is None:
            logger.error("face_recognition not available; compare_faces cannot run")
            return []

        try:
            targets_to_check = list(ENCODINGS.keys()) if target_names is None else [t for t in target_names if t in ENCODINGS]
            if not targets_to_check:
                logger.warning("No targets available for comparison")
                return []

            # Vectorized comparison
            stored_encodings = np.array([ENCODINGS[t] for t in targets_to_check])
            distances = face_recognition.face_distance(stored_encodings, test_encoding)
            matches = distances <= self.tolerance

            results: List[Dict[str, Any]] = []
            for i, target in enumerate(targets_to_check):
                distance = float(distances[i])
                is_match = bool(matches[i])
                item = {"target": target, "match": is_match}
                if return_distances:
                    item["distance"] = distance
                    item["confidence"] = self._get_confidence_level(distance)
                    item["confidence_score"] = self._calibrate_confidence(distance)
                results.append(item)

                # track
                self._comparison_history.append({
                    "distance": distance,
                    "match": is_match,
                    "timestamp": datetime.now().isoformat()
                })

            if return_distances:
                results.sort(key=lambda x: x.get("distance", 1.0))

            processing_time = time.time() - start_time
            with self._metrics_lock:
                self._metrics["total_comparisons"] += len(targets_to_check)
                self._metrics["cumulative_comparison_time"] += processing_time

            logger.debug("Comparison completed in %.3fs; matches=%d", processing_time, sum(1 for r in results if r.get("match")))
            return results
        except Exception as e:
            logger.exception("Error comparing faces: %s", e)
            return []

    # -------------------------------
    # EXISTING: Batch Compare (KEPT)
    # -------------------------------
    def batch_compare_faces(self, test_encodings: List[np.ndarray], target_names: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        if face_recognition is None:
            logger.error("face_recognition not available; batch_compare_faces cannot run")
            return [[] for _ in test_encodings]

        if target_names is None:
            targets = list(ENCODINGS.keys())
        else:
            targets = [t for t in target_names if t in ENCODINGS]

        if not targets:
            return [[] for _ in test_encodings]

        stored_encodings = np.array([ENCODINGS[t] for t in targets])
        all_results: List[List[Dict[str, Any]]] = []
        for test_enc in test_encodings:
            try:
                distances = face_recognition.face_distance(stored_encodings, test_enc)
                matches = distances <= self.tolerance
                results: List[Dict[str, Any]] = []
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
                logger.exception("Error in batch comparison: %s", e)
                all_results.append([])
        return all_results

    # -------------------------------
    # EXISTING: Confidence Level (KEPT)
    # -------------------------------
    def _get_confidence_level(self, distance: float) -> str:
        if distance < 0.4:
            return "high"
        elif distance < 0.6:
            return "medium"
        else:
            return "low"

    # -------------------------------
    # NEW: Calibrated confidence score (improved)
    # -------------------------------
    def _calibrate_confidence(self, distance: float) -> float:
        """
        Convert distance to a calibrated confidence score (0-100).
        Uses a smooth sigmoid-like mapping for better distribution.
        """
        x = max(0.0, min(1.0, 1.0 - distance))  # 0..1 where 1 is perfect match
        # parameters tuned for reasonable mapping
        k = 12.0
        calibrated = 100.0 / (1.0 + np.exp(-k * (x - 0.5)))
        return round(float(calibrated), 2)

    # -------------------------------
    # EXISTING: Get All Targets (KEPT)
    # -------------------------------
    def get_all_targets(self) -> List[str]:
        return list(ENCODINGS.keys())

    # -------------------------------
    # EXISTING: Delete Face (ENHANCED)
    # -------------------------------
    def delete_face(self, target_name: str) -> Dict[str, Any]:
        logger.info("Deleting face: %s", target_name)
        try:
            removed_from_memory = False
            if target_name in ENCODINGS:
                del ENCODINGS[target_name]
                removed_from_memory = True

            self._remove_from_cache(target_name)

            # delete from MongoDB
            result = faces_collection.delete_one({"target": target_name})
            removed_from_db = result.deleted_count > 0

            # remove backup file if present
            backup_path = BACKUP_DIR / f"{target_name}_backup.json"
            try:
                if backup_path.exists():
                    backup_path.unlink()
            except Exception:
                logger.debug("Failed to remove backup for %s (non-fatal)", target_name)

            if removed_from_memory or removed_from_db:
                logger.info("Deleted face %s (memory=%s, db=%s)", target_name, removed_from_memory, removed_from_db)
                return {"success": True, "message": f"Face '{target_name}' deleted successfully", "removed_from_memory": removed_from_memory, "removed_from_db": removed_from_db}
            else:
                logger.warning("Face %s not found", target_name)
                return {"success": False, "message": f"Face '{target_name}' not found in database or memory"}
        except Exception as e:
            logger.exception("Error deleting face %s: %s", target_name, e)
            return {"success": False, "message": f"Error deleting face: {str(e)}"}

    # -------------------------------
    # NEW: Face clustering (keeps algorithm style)
    # -------------------------------
    def cluster_faces(self, distance_threshold: float = 0.6) -> Dict[str, List[str]]:
        logger.info("Clustering %d faces with threshold %.3f", len(ENCODINGS), distance_threshold)
        if len(ENCODINGS) < 2:
            return {"cluster_0": list(ENCODINGS.keys())}

        targets = list(ENCODINGS.keys())
        encodings = np.array([ENCODINGS[t] for t in targets])
        clusters: Dict[str, List[str]] = {}
        cluster_id = 0
        assigned = set()

        for i, target in enumerate(targets):
            if target in assigned:
                continue
            members = [target]
            assigned.add(target)
            for j, other in enumerate(targets):
                if other in assigned or i == j:
                    continue
                try:
                    if face_recognition is None:
                        # fallback to euclidean distance
                        dist = float(np.linalg.norm(encodings[i] - encodings[j]))
                    else:
                        dist = float(face_recognition.face_distance([encodings[i]], encodings[j])[0])
                    if dist <= distance_threshold:
                        members.append(other)
                        assigned.add(other)
                except Exception:
                    continue
            clusters[f"cluster_{cluster_id}"] = members
            cluster_id += 1
        logger.info("Clustering produced %d clusters", len(clusters))
        return clusters

    # -------------------------------
    # NEW: Find K nearest neighbors
    # -------------------------------
    def find_k_nearest(self, test_encoding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if len(ENCODINGS) == 0:
            return []
        logger.debug("Finding %d nearest neighbors", k)
        targets = list(ENCODINGS.keys())
        encodings = np.array([ENCODINGS[t] for t in targets])
        if face_recognition is None:
            distances = np.linalg.norm(encodings - test_encoding, axis=1)
        else:
            distances = face_recognition.face_distance(encodings, test_encoding)
        # handle k > n
        k = min(k, len(distances))
        idx = np.argpartition(distances, k - 1)[:k]
        idx = idx[np.argsort(distances[idx])]
        results = []
        for i in idx:
            results.append({
                "target": targets[int(i)],
                "distance": float(distances[int(i)]),
                "confidence": self._get_confidence_level(float(distances[int(i)])),
                "confidence_score": self._calibrate_confidence(float(distances[int(i)]))
            })
        return results

    # -------------------------------
    # NEW: Get quality statistics
    # -------------------------------
    def get_quality_statistics(self) -> Dict[str, Any]:
        if not self._quality_history:
            return {"total_assessments": 0, "average_score": 0, "average_blur_score": 0, "average_lighting_score": 0}
        scores = [q["score"] for q in self._quality_history]
        blur = [q["blur_score"] for q in self._quality_history]
        light = [q["lighting_score"] for q in self._quality_history]
        return {
            "total_assessments": len(self._quality_history),
            "average_score": round(float(np.mean(scores)), 2),
            "min_score": round(float(np.min(scores)), 2),
            "max_score": round(float(np.max(scores)), 2),
            "average_blur_score": round(float(np.mean(blur)), 2),
            "average_lighting_score": round(float(np.mean(light)), 2),
            "std_deviation": round(float(np.std(scores)), 2)
        }

    # -------------------------------
    # NEW: Performance metrics (fixed math)
    # -------------------------------
    def get_performance_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)  # shallow copy
        total_enc = m.get("total_encodings", 0)
        avg_enc_time = (m["cumulative_encoding_time"] / total_enc) if total_enc > 0 else 0.0
        total_cmp = m.get("total_comparisons", 0)
        avg_cmp_time = (m["cumulative_comparison_time"] / total_cmp) if total_cmp > 0 else 0.0
        total_cache_requests = m.get("cache_hits", 0) + m.get("cache_misses", 0)
        cache_hit_rate = (m["cache_hits"] / total_cache_requests * 100.0) if total_cache_requests > 0 else 0.0
        total_encoding_attempts = total_enc + m.get("failed_encodings", 0)
        success_rate = (total_enc / total_encoding_attempts * 100.0) if total_encoding_attempts > 0 else 100.0
        return {
            **m,
            "average_encoding_time": round(avg_enc_time, 4),
            "average_comparison_time": round(avg_cmp_time, 4),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "encoding_success_rate": round(success_rate, 2),
            "cache_size": len(self._encoding_cache),
            "total_faces_stored": len(ENCODINGS),
            "comparison_history_size": len(self._comparison_history)
        }

    # -------------------------------
    # NEW: Clear cache
    # -------------------------------
    def clear_cache(self) -> Dict[str, Any]:
        with self._cache_lock:
            size = len(self._encoding_cache)
            self._encoding_cache.clear()
        logger.info("Cache cleared (%d entries)", size)
        return {"success": True, "cleared_entries": size}

    # -------------------------------
    # NEW: Optimize storage
    # -------------------------------
    def optimize_storage(self) -> Dict[str, Any]:
        logger.info("Optimizing storage")
        cleaned = self._clean_cache()
        synced = 0
        for target in list(ENCODINGS.keys()):
            try:
                rec = faces_collection.find_one({"target": target})
                if not rec:
                    store_embedding(target, ENCODINGS[target])
                    synced += 1
            except Exception as e:
                logger.debug("Sync failed for %s: %s", target, e)
        logger.info("Optimization done: cleaned=%d, synced=%d", cleaned, synced)
        return {"success": True, "cache_entries_cleaned": cleaned, "database_synced": synced, "current_cache_size": len(self._encoding_cache), "total_faces": len(ENCODINGS)}

    # -------------------------------
    # NEW: Reset metrics
    # -------------------------------
    def reset_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            old = dict(self._metrics)
            self._metrics = {"total_encodings": 0, "total_comparisons": 0, "cache_hits": 0, "cache_misses": 0, "cumulative_encoding_time": 0.0, "cumulative_comparison_time": 0.0, "failed_encodings": 0}
        logger.info("Metrics reset")
        return {"success": True, "previous_metrics": old}

    # -------------------------------
    # CACHE helpers (same semantics but robust)
    # -------------------------------
    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        with self._cache_lock:
            self._encoding_cache[key] = {"data": data, "timestamp": datetime.now()}

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        with self._cache_lock:
            entry = self._encoding_cache.get(key)
            if not entry:
                return None
            age = (datetime.now() - entry["timestamp"]).total_seconds()
            if age > self._cache_ttl:
                # expired
                del self._encoding_cache[key]
                return None
            return entry["data"]

    def _remove_from_cache(self, key: str):
        with self._cache_lock:
            if key in self._encoding_cache:
                del self._encoding_cache[key]

    def _clean_cache(self) -> int:
        with self._cache_lock:
            now = datetime.now()
            expired = [k for k, v in self._encoding_cache.items() if (now - v["timestamp"]).total_seconds() > self._cache_ttl]
            for k in expired:
                del self._encoding_cache[k]
            return len(expired)

    # -------------------------------
    # BACKUP helpers (atomic)
    # -------------------------------
    def _backup_encoding_atomic(self, target_name: str, encoding: List[float]):
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_file = BACKUP_DIR / f"{target_name}_backup.json"
        temp_file = BACKUP_DIR / f"{target_name}_backup.json.tmp"
        payload = {"target": target_name, "encoding": encoding, "timestamp": datetime.now().isoformat(), "version": 1}
        # atomic write: write to temp then rename
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temp_file.replace(backup_file)

    # -------------------------------
    # HEALTH CHECK
    # -------------------------------
    def health_check(self) -> Dict[str, Any]:
        health = {"status": "healthy", "issues": [], "warnings": []}
        if len(ENCODINGS) == 0:
            health["warnings"].append("No faces enrolled in system")
        cache_size = len(self._encoding_cache)
        if cache_size > 1000:
            health["warnings"].append(f"Large cache size: {cache_size}")
        with self._metrics_lock:
            if self._metrics.get("failed_encodings", 0) > 100:
                health["warnings"].append(f"High failure rate: {self._metrics['failed_encodings']}")
        try:
            faces_collection.find_one()
        except Exception as db_err:
            health["status"] = "degraded"
            health["issues"].append(f"Database connectivity issue: {db_err}")
        if len(health["issues"]) > 2:
            health["status"] = "unhealthy"
        return health


# -------------------------------
# Singleton instance (EXISTING)
# -------------------------------
face_service = FaceService()
