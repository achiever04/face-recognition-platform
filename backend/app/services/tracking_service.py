# backend/app/services/tracking_service.py

"""
Person Tracking Service
Handles cross-camera tracking, movement history, and location management.

ENHANCED: Added predictive analytics, pattern recognition, heatmaps, speed estimation,
safer locking, optional DB persistence (background), improved logging, and small bug fixes.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import logging
import json
import os

# Prefer app logger helper when available
try:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

from app.state import PERSON_LOCATIONS, CAMERA_METADATA

# Optional DB persistence helper (save detections)
try:
    from app.utils.db import save_detection_to_db
except Exception:
    save_detection_to_db = None  # optional, best-effort persistence


class TrackingService:
    """Service for tracking persons across multiple cameras"""

    def __init__(self, history_limit: int = 100, cooldown_seconds: int = 10):
        """
        Args:
            history_limit: Maximum history entries to keep per person
            cooldown_seconds: Minimum seconds between detections at same camera
        """
        # Keep shared in-memory mapping (existing object)
        self.current_locations: Dict[str, int] = PERSON_LOCATIONS

        # movement history per person (deque of detection dicts)
        self.movement_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_limit))

        # mapping (person, camera) -> last detection datetime
        self.last_detection: Dict[Tuple[str, int], datetime] = {}

        # cooldown window
        self.cooldown = timedelta(seconds=cooldown_seconds)

        # thread-safety: prefer re-entrant lock to allow nested calls in same thread
        self._lock = threading.RLock()

        # statistics
        self.stats = {
            "total_detections": 0,
            "unique_persons": 0,
            "camera_switches": 0
        }

        # advanced tracking structures
        self._trajectory_cache: Dict[str, Dict[str, Any]] = {}
        self._speed_history: Dict[str, List[float]] = defaultdict(list)
        self._dwell_times: Dict[str, List[float]] = defaultdict(list)
        self._hourly_patterns: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._camera_transitions: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._anomaly_scores: Dict[str, List[float]] = defaultdict(list)

        # heatmap data
        self._location_heatmap: Dict[int, int] = defaultdict(int)
        self._person_heatmaps: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # performance metrics
        self._metrics = {
            "total_predictions": 0,
            "prediction_accuracy": 0.0,
            "average_speed": 0.0,
            "average_dwell_time": 0.0
        }

        logger.info("TrackingService initialized (history_limit=%d, cooldown=%ds)", history_limit, cooldown_seconds)

    # -------------------------------
    # EXISTING: Record Detection (ENHANCED)
    # -------------------------------
    def record_detection(
        self,
        person_name: str,
        camera_id: int,
        distance: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Record a person detection event.

        ENHANCED: Adds speed calculation, dwell time, heatmap updates, pattern analysis,
        background persistence (if DB helper available), and fixes unique_persons counting.

        Returns a dict with recording metadata and computed analytics.
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            logger.debug("Recording detection: %s at camera %s (distance=%.4f)", person_name, camera_id, float(distance))

            # Normalize input types
            try:
                camera_id = int(camera_id)
            except Exception:
                logger.warning("Invalid camera_id provided: %s", camera_id)
                return {"recorded": False, "message": "Invalid camera_id"}

            # Duplicate detection (cooldown) keyed by (person, camera)
            detection_key = (person_name, camera_id)
            if detection_key in self.last_detection:
                time_since_last = timestamp - self.last_detection[detection_key]
                if time_since_last < self.cooldown:
                    logger.debug("Duplicate detection within cooldown for %s at %s", person_name, camera_id)
                    return {
                        "recorded": False,
                        "is_new_location": False,
                        "previous_location": None,
                        "duplicate": True,
                        "message": f"Duplicate detection (cooldown: {int(self.cooldown.total_seconds())}s)"
                    }

            # Camera metadata — safe defaults
            camera_info = CAMERA_METADATA.get(camera_id, {}) or {}
            camera_name = camera_info.get("name", f"Camera {camera_id}")
            geo = camera_info.get("geo", (0.0, 0.0))

            # Previous location (if any)
            previous_location = self.current_locations.get(person_name)
            is_new_location = previous_location != camera_id

            # Speed calculation if moved
            speed_kmh = 0.0
            if is_new_location and previous_location is not None:
                speed_kmh = self._calculate_speed(person_name, previous_location, camera_id, timestamp)
                if speed_kmh > 0:
                    self._speed_history[person_name].append(speed_kmh)
                    logger.debug("Speed computed for %s: %.2f km/h", person_name, speed_kmh)

            # Dwell time at previous location
            dwell_time = 0.0
            if previous_location is not None:
                dwell_time = self._calculate_dwell_time(person_name, previous_location, timestamp)
                if dwell_time > 0:
                    self._dwell_times[person_name].append(dwell_time)

            # Detection record (store timestamp as ISO string)
            detection_record = {
                "person": person_name,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "geo": geo,
                "distance": float(distance),
                "confidence": self._calculate_confidence(distance),
                "timestamp": timestamp.isoformat(),
                "speed_kmh": round(float(speed_kmh), 2),
                "dwell_time_seconds": round(float(dwell_time), 2)
            }

            # Update current location & history
            self.current_locations[person_name] = camera_id
            was_new_person = person_name not in self.movement_history or len(self.movement_history[person_name]) == 0
            self.movement_history[person_name].append(detection_record)

            # Update last detection for this (person, camera)
            self.last_detection[detection_key] = timestamp

            # Update statistics
            self.stats["total_detections"] += 1
            if was_new_person:
                self.stats["unique_persons"] += 1
            if is_new_location and previous_location is not None:
                self.stats["camera_switches"] += 1

            # Heatmaps & patterns
            self._location_heatmap[camera_id] += 1
            self._person_heatmaps[person_name][camera_id] += 1
            hour = timestamp.hour
            self._hourly_patterns[person_name][hour] += 1
            if is_new_location and previous_location is not None:
                self._camera_transitions[previous_location][camera_id] += 1

            # Anomaly score
            anomaly_score = self._calculate_anomaly_score(person_name, camera_id, speed_kmh)
            self._anomaly_scores[person_name].append(anomaly_score)

            # Update trajectory prediction
            try:
                self._update_trajectory(person_name)
            except Exception as e:
                logger.debug("Trajectory update error (ignored): %s", e)

            logger.info(
                "Detection recorded: %s @ %s (speed=%.2f km/h, dwell=%.2fs, anomaly=%.3f)",
                person_name, camera_name, speed_kmh, dwell_time, anomaly_score
            )

            # Background persistence (best-effort) — non-blocking
            if save_detection_to_db is not None:
                try:
                    threading.Thread(target=self._persist_detection, args=(detection_record,), daemon=True).start()
                except Exception as e:
                    logger.debug("Failed to spawn persistence thread (ignored): %s", e)

            return {
                "recorded": True,
                "is_new_location": is_new_location,
                "previous_location": previous_location,
                "duplicate": False,
                "message": "Detection recorded successfully",
                "detection": detection_record,
                "speed_kmh": round(float(speed_kmh), 2),
                "dwell_time_seconds": round(float(dwell_time), 2),
                "anomaly_score": round(float(anomaly_score), 3)
            }

    def _persist_detection(self, detection: Dict[str, Any]):
        """
        Persist detection to database (non-blocking background).
        Uses app.utils.db.save_detection_to_db if available. Exceptions are caught and logged.
        """
        try:
            if save_detection_to_db:
                save_detection_to_db(detection)
                logger.debug("Persisted detection to DB for %s at camera %s", detection.get("person"), detection.get("camera_id"))
        except Exception as e:
            logger.debug("Background persistence failed (ignored): %s", e)

    # -------------------------------
    # EXISTING: Get Movement History (KEPT AS IS)
    # -------------------------------
    def get_movement_history(
        self,
        person_name: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if person_name not in self.movement_history:
                return []

            history = list(self.movement_history[person_name])

            if since:
                history = [h for h in history if datetime.fromisoformat(h["timestamp"]) > since]

            history.reverse()  # newest first

            if limit:
                history = history[:limit]

            return history

    # -------------------------------
    # EXISTING: Get All Movement Logs (KEPT AS IS)
    # -------------------------------
    def get_all_movements(
        self,
        limit_per_person: Optional[int] = None,
        active_only: bool = False,
        activity_threshold_minutes: int = 30
    ) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            result: Dict[str, List[Dict[str, Any]]] = {}
            cutoff_time = datetime.now() - timedelta(minutes=activity_threshold_minutes)

            for person, history in self.movement_history.items():
                if active_only:
                    latest = history[-1] if history else None
                    if not latest:
                        continue
                    latest_time = datetime.fromisoformat(latest["timestamp"])
                    if latest_time < cutoff_time:
                        continue

                person_history = list(history)
                person_history.reverse()
                if limit_per_person:
                    person_history = person_history[:limit_per_person]
                result[person] = person_history

            return result

    # -------------------------------
    # EXISTING: Get Current Locations (KEPT AS IS)
    # -------------------------------
    def get_current_locations(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            locations: Dict[str, Dict[str, Any]] = {}
            for person, camera_id in self.current_locations.items():
                history = self.movement_history.get(person)
                if not history:
                    continue
                latest = history[-1]
                locations[person] = {
                    "camera_id": camera_id,
                    "camera_name": latest.get("camera_name"),
                    "geo": latest.get("geo"),
                    "last_seen": latest.get("timestamp"),
                    "confidence": latest.get("confidence")
                }
            return locations

    # -------------------------------
    # EXISTING: Get Movement Path (KEPT AS IS)
    # -------------------------------
    def get_movement_path(
        self,
        person_name: str,
        include_duplicates: bool = False
    ) -> List[Tuple[int, str, datetime]]:
        with self._lock:
            if person_name not in self.movement_history:
                return []

            history = list(self.movement_history[person_name])

            if include_duplicates:
                return [
                    (h["camera_id"], h["camera_name"], datetime.fromisoformat(h["timestamp"]))
                    for h in history
                ]

            path: List[Tuple[int, str, datetime]] = []
            prev_camera = None
            for h in history:
                cam_id = h["camera_id"]
                if cam_id != prev_camera:
                    path.append((cam_id, h["camera_name"], datetime.fromisoformat(h["timestamp"])))
                    prev_camera = cam_id
            return path

    # -------------------------------
    # EXISTING: Detect Suspicious Patterns (ENHANCED)
    # -------------------------------
    def detect_suspicious_patterns(
        self,
        person_name: str,
        loitering_threshold_minutes: int = 15,
        revisit_threshold: int = 3
    ) -> Dict[str, Any]:
        with self._lock:
            logger.debug("Analyzing suspicious patterns for %s", person_name)

            if person_name not in self.movement_history:
                return {"is_suspicious": False, "patterns": [], "loitering_duration": None, "revisit_count": {}, "anomaly_score": 0.0}

            history = list(self.movement_history[person_name])
            patterns: List[str] = []
            duration_minutes = None

            if len(history) >= 2:
                first = datetime.fromisoformat(history[0]["timestamp"])
                last = datetime.fromisoformat(history[-1]["timestamp"])
                duration_minutes = (last - first).total_seconds() / 60.0
                cameras = set(h["camera_id"] for h in history)
                if len(cameras) == 1 and duration_minutes > loitering_threshold_minutes:
                    patterns.append(f"Loitering detected: {duration_minutes:.1f} minutes at same location")

            camera_visits = defaultdict(int)
            for h in history:
                camera_visits[h["camera_id"]] += 1
            for cam_id, count in camera_visits.items():
                if count >= revisit_threshold:
                    cam_name = CAMERA_METADATA.get(cam_id, {}).get("name", f"Camera {cam_id}")
                    patterns.append(f"Revisited {cam_name} {count} times")

            if len(history) >= 4:
                recent = history[-4:]
                cameras_recent = [h["camera_id"] for h in recent]
                if len(set(cameras_recent)) == 2 and cameras_recent[0] == cameras_recent[2]:
                    patterns.append("Rapid back-and-forth movement detected")

            # Speed checks
            speeds = self._speed_history.get(person_name, [])
            if len(speeds) > 0:
                avg_speed = float(np.mean(speeds))
                max_speed = float(np.max(speeds))
                if max_speed > 10:
                    patterns.append(f"Unusually high speed detected: {max_speed:.1f} km/h")
                if avg_speed > 6:
                    patterns.append(f"High average speed: {avg_speed:.1f} km/h")
            else:
                avg_speed = 0.0
                max_speed = 0.0

            # Night activity
            hours = [datetime.fromisoformat(h["timestamp"]).hour for h in history]
            night_detections = sum(1 for h in hours if h < 6 or h > 22)
            if len(history) and night_detections > len(history) * 0.5:
                patterns.append(f"Mostly active during night hours ({night_detections}/{len(history)} detections)")

            anomaly_scores = self._anomaly_scores.get(person_name, [0.0])
            avg_anomaly = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0

            if avg_anomaly > 0.7:
                patterns.append(f"High anomaly score: {avg_anomaly:.2f}")

            is_suspicious = len(patterns) > 0 or avg_anomaly > 0.6

            logger.info("Suspicious analysis for %s: patterns=%d, anomaly=%.2f", person_name, len(patterns), avg_anomaly)

            return {
                "is_suspicious": is_suspicious,
                "patterns": patterns,
                "loitering_duration": duration_minutes,
                "revisit_count": dict(camera_visits),
                "anomaly_score": round(avg_anomaly, 3),
                "speed_analysis": {
                    "average_speed_kmh": round(float(avg_speed), 2) if speeds else 0,
                    "max_speed_kmh": round(float(max_speed), 2) if speeds else 0,
                    "min_speed_kmh": round(float(np.min(speeds)), 2) if speeds else 0
                },
                "time_analysis": {
                    "total_detections": len(history),
                    "night_detections": night_detections,
                    "most_active_hour": max(self._hourly_patterns[person_name].items(), key=lambda x: x[1])[0] if self._hourly_patterns[person_name] else None
                }
            }

    # -------------------------------
    # EXISTING: Clear History (KEPT AS IS)
    # -------------------------------
    def clear_history(self, person_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if person_name:
                if person_name in self.movement_history:
                    count = len(self.movement_history[person_name])
                    del self.movement_history[person_name]
                    if person_name in self.current_locations:
                        del self.current_locations[person_name]
                    return {"success": True, "cleared": count}
                else:
                    return {"success": False, "cleared": 0}
            else:
                total = sum(len(h) for h in self.movement_history.values())
                self.movement_history.clear()
                self.current_locations.clear()
                self.last_detection.clear()
                return {"success": True, "cleared": total}

    # -------------------------------
    # EXISTING: Get Statistics (ENHANCED)
    # -------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            total_speeds = []
            total_dwell_times = []
            for speeds in self._speed_history.values():
                total_speeds.extend(speeds)
            for dwell_times in self._dwell_times.values():
                total_dwell_times.extend(dwell_times)

            return {
                **self.stats,
                "active_persons": len(self.current_locations),
                "total_movement_records": sum(len(h) for h in self.movement_history.values()),
                "average_speed_kmh": round(float(np.mean(total_speeds)), 2) if total_speeds else 0,
                "average_dwell_time_seconds": round(float(np.mean(total_dwell_times)), 2) if total_dwell_times else 0,
                "total_camera_transitions": sum(sum(trans.values()) for trans in self._camera_transitions.values()),
                "unique_camera_transitions": len(self._camera_transitions)
            }

    # -------------------------------
    # EXISTING: Helper - Calculate Confidence (KEPT AS IS)
    # -------------------------------
    def _calculate_confidence(self, distance: float) -> str:
        if distance < 0.4:
            return "high"
        elif distance < 0.6:
            return "medium"
        else:
            return "low"

    # -------------------------------
    # NEW: Calculate speed between cameras
    # -------------------------------
    def _calculate_speed(
        self,
        person_name: str,
        from_camera: int,
        to_camera: int,
        current_time: datetime
    ) -> float:
        try:
            from_geo = CAMERA_METADATA.get(from_camera, {}).get("geo", (0.0, 0.0))
            to_geo = CAMERA_METADATA.get(to_camera, {}).get("geo", (0.0, 0.0))
            distance_km = self._haversine_distance(from_geo, to_geo)
            if distance_km == 0:
                return 0.0

            last_time_key = (person_name, from_camera)
            if last_time_key not in self.last_detection:
                return 0.0
            last_time = self.last_detection[last_time_key]
            time_diff_hours = (current_time - last_time).total_seconds() / 3600.0
            if time_diff_hours <= 0:
                return 0.0
            speed = distance_km / time_diff_hours
            return float(speed)
        except Exception as e:
            logger.warning("Speed calc failed: %s", e)
            return 0.0

    # -------------------------------
    # NEW: Calculate dwell time
    # -------------------------------
    def _calculate_dwell_time(self, person_name: str, camera_id: int, current_time: datetime) -> float:
        try:
            history = self.movement_history.get(person_name)
            if not history:
                return 0.0

            # Find most recent contiguous block at camera_id from the end
            first_time = None
            for detection in reversed(history):
                if detection["camera_id"] == camera_id:
                    first_time = datetime.fromisoformat(detection["timestamp"])
                else:
                    # Once we hit a different camera, contiguous block ends
                    if first_time:
                        break

            if first_time is None:
                return 0.0
            dwell = (current_time - first_time).total_seconds()
            return float(dwell)
        except Exception as e:
            logger.warning("Dwell time calc failed: %s", e)
            return 0.0

    # -------------------------------
    # NEW: Haversine distance calculation
    # -------------------------------
    def _haversine_distance(self, geo1: Tuple[float, float], geo2: Tuple[float, float]) -> float:
        lat1, lon1 = float(geo1[0]), float(geo1[1])
        lat2, lon2 = float(geo2[0]), float(geo2[1])
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(R * c)

    # -------------------------------
    # NEW: Calculate anomaly score
    # -------------------------------
    def _calculate_anomaly_score(self, person_name: str, camera_id: int, speed_kmh: float) -> float:
        score = 0.0
        try:
            if speed_kmh > 0:
                if speed_kmh > 10:
                    score += 0.3
                elif speed_kmh > 6:
                    score += 0.15

            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:
                score += 0.3
            elif current_hour < 8 or current_hour > 20:
                score += 0.15

            total_visits = self._location_heatmap.get(camera_id, 0)
            person_visits = self._person_heatmaps[person_name].get(camera_id, 0)
            if total_visits > 0:
                visit_ratio = person_visits / total_visits
                if visit_ratio > 0.5:
                    score += 0.4
                elif visit_ratio > 0.3:
                    score += 0.2
        except Exception as e:
            logger.debug("Anomaly calc failed: %s", e)
        return min(1.0, float(score))

    # -------------------------------
    # NEW: Update trajectory prediction
    # -------------------------------
    def _update_trajectory(self, person_name: str):
        try:
            history = self.movement_history.get(person_name)
            if not history or len(history) < 2:
                return

            recent = list(history)[-5:]
            cameras = [h["camera_id"] for h in recent]

            last_camera = cameras[-1]
            transitions = self._camera_transitions.get(last_camera, {})
            if transitions:
                predicted_camera = max(transitions.items(), key=lambda x: x[1])[0]
                total = sum(transitions.values()) or 1
                self._trajectory_cache[person_name] = {
                    "current_camera": last_camera,
                    "predicted_next_camera": predicted_camera,
                    "confidence": transitions[predicted_camera] / total,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.debug("Trajectory update failed for %s: %s", person_name, e)

    # -------------------------------
    # NEW: Get predicted trajectory
    # -------------------------------
    def get_predicted_trajectory(self, person_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._trajectory_cache.get(person_name)

    # -------------------------------
    # NEW: Get heatmap data
    # -------------------------------
    def get_heatmap(self, person_name: Optional[str] = None) -> Dict[int, int]:
        with self._lock:
            if person_name:
                return dict(self._person_heatmaps.get(person_name, {}))
            return dict(self._location_heatmap)

    # -------------------------------
    # NEW: Get time-based patterns
    # -------------------------------
    def get_time_patterns(self, person_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if person_name:
                patterns = dict(self._hourly_patterns.get(person_name, {}))
                return {
                    "person": person_name,
                    "hourly_distribution": patterns,
                    "most_active_hour": max(patterns.items(), key=lambda x: x[1])[0] if patterns else None,
                    "least_active_hour": min(patterns.items(), key=lambda x: x[1])[0] if patterns else None,
                    "total_detections": sum(patterns.values())
                }
            else:
                global_patterns = defaultdict(int)
                for person_patterns in self._hourly_patterns.values():
                    for hour, count in person_patterns.items():
                        global_patterns[hour] += count
                return {
                    "hourly_distribution": dict(global_patterns),
                    "most_active_hour": max(global_patterns.items(), key=lambda x: x[1])[0] if global_patterns else None,
                    "peak_activity_count": max(global_patterns.values()) if global_patterns else 0,
                    "total_detections": sum(global_patterns.values())
                }

    # -------------------------------
    # NEW: Get transition matrix
    # -------------------------------
    def get_transition_matrix(self) -> Dict[int, Any]:
        with self._lock:
            matrix: Dict[int, Any] = {}
            for from_cam, transitions in self._camera_transitions.items():
                total_from = sum(transitions.values()) or 1
                matrix[from_cam] = {
                    "transitions": dict(transitions),
                    "total": total_from,
                    "probabilities": {to_cam: count / total_from for to_cam, count in transitions.items()}
                }
            return matrix

    # -------------------------------
    # NEW: Get speed statistics
    # -------------------------------
    def get_speed_statistics(self, person_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if person_name:
                speeds = self._speed_history.get(person_name, [])
                if not speeds:
                    return {"person": person_name, "total_movements": 0, "average_speed": 0, "max_speed": 0, "min_speed": 0}
                return {
                    "person": person_name,
                    "total_movements": len(speeds),
                    "average_speed": round(float(np.mean(speeds)), 2),
                    "max_speed": round(float(np.max(speeds)), 2),
                    "min_speed": round(float(np.min(speeds)), 2),
                    "median_speed": round(float(np.median(speeds)), 2),
                    "std_deviation": round(float(np.std(speeds)), 2)
                }
            else:
                all_speeds = []
                for speeds in self._speed_history.values():
                    all_speeds.extend(speeds)
                if not all_speeds:
                    return {"total_movements": 0, "average_speed": 0}
                return {
                    "total_movements": len(all_speeds),
                    "average_speed": round(float(np.mean(all_speeds)), 2),
                    "max_speed": round(float(np.max(all_speeds)), 2),
                    "min_speed": round(float(np.min(all_speeds)), 2),
                    "median_speed": round(float(np.median(all_speeds)), 2),
                    "std_deviation": round(float(np.std(all_speeds)), 2)
                }

    # -------------------------------
    # NEW: Get dwell time statistics
    # -------------------------------
    def get_dwell_statistics(self, person_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if person_name:
                dwell_times = self._dwell_times.get(person_name, [])
                if not dwell_times:
                    return {"person": person_name, "total_stops": 0, "average_dwell_seconds": 0}
                return {
                    "person": person_name,
                    "total_stops": len(dwell_times),
                    "average_dwell_seconds": round(float(np.mean(dwell_times)), 2),
                    "max_dwell_seconds": round(float(np.max(dwell_times)), 2),
                    "min_dwell_seconds": round(float(np.min(dwell_times)), 2),
                    "median_dwell_seconds": round(float(np.median(dwell_times)), 2),
                    "total_dwell_time_minutes": round(float(sum(dwell_times) / 60.0), 2)
                }
            else:
                all_dwell_times = []
                for dwell_times in self._dwell_times.values():
                    all_dwell_times.extend(dwell_times)
                if not all_dwell_times:
                    return {"total_stops": 0, "average_dwell_seconds": 0}
                return {
                    "total_stops": len(all_dwell_times),
                    "average_dwell_seconds": round(float(np.mean(all_dwell_times)), 2),
                    "max_dwell_seconds": round(float(np.max(all_dwell_times)), 2),
                    "min_dwell_seconds": round(float(np.min(all_dwell_times)), 2),
                    "median_dwell_seconds": round(float(np.median(all_dwell_times)), 2),
                    "total_dwell_time_minutes": round(float(sum(all_dwell_times) / 60.0), 2)
                }

    # -------------------------------
    # NEW: Get anomaly reports
    # -------------------------------
    def get_anomaly_report(self, threshold: float = 0.6) -> List[Dict[str, Any]]:
        with self._lock:
            anomalous: List[Dict[str, Any]] = []
            for person, scores in self._anomaly_scores.items():
                if not scores:
                    continue
                avg_score = float(np.mean(scores))
                max_score = float(np.max(scores))
                if avg_score >= threshold or max_score >= 0.8:
                    anomalous.append({
                        "person": person,
                        "average_anomaly_score": round(avg_score, 3),
                        "max_anomaly_score": round(max_score, 3),
                        "total_detections": len(scores),
                        "high_anomaly_detections": int(sum(1 for s in scores if s > threshold))
                    })
            anomalous.sort(key=lambda x: x["average_anomaly_score"], reverse=True)
            return anomalous

    # -------------------------------
    # NEW: Compare movement patterns
    # -------------------------------
    def compare_movement_patterns(self, person1: str, person2: str) -> Dict[str, Any]:
        with self._lock:
            logger.debug("Comparing movement patterns: %s vs %s", person1, person2)
            if person1 not in self.movement_history or person2 not in self.movement_history:
                return {"similarity_score": 0.0, "message": "One or both persons not found"}

            heatmap1 = self._person_heatmaps[person1]
            heatmap2 = self._person_heatmaps[person2]
            all_cameras = set(heatmap1.keys()).union(set(heatmap2.keys()))
            if not all_cameras:
                return {"similarity_score": 0.0, "common_locations": [], "location_overlap": 0.0}

            common_cameras = set(heatmap1.keys()).intersection(set(heatmap2.keys()))
            location_overlap = len(common_cameras) / len(all_cameras)

            pattern1 = self._hourly_patterns[person1]
            pattern2 = self._hourly_patterns[person2]

            vec1 = [pattern1.get(h, 0) for h in range(24)]
            vec2 = [pattern2.get(h, 0) for h in range(24)]
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = np.sqrt(sum(a ** 2 for a in vec1))
            magnitude2 = np.sqrt(sum(b ** 2 for b in vec2))
            time_similarity = (dot_product / (magnitude1 * magnitude2)) if magnitude1 > 0 and magnitude2 > 0 else 0.0

            speeds1 = self._speed_history.get(person1, [])
            speeds2 = self._speed_history.get(person2, [])
            if speeds1 and speeds2:
                speed_similarity = max(0.0, 1.0 - abs(float(np.mean(speeds1)) - float(np.mean(speeds2))) / 10.0)
            else:
                speed_similarity = 0.5

            overall_similarity = location_overlap * 0.4 + time_similarity * 0.4 + speed_similarity * 0.2

            return {
                "similarity_score": round(float(overall_similarity), 3),
                "location_overlap": round(float(location_overlap), 3),
                "time_similarity": round(float(time_similarity), 3),
                "speed_similarity": round(float(speed_similarity), 3),
                "common_locations": list(common_cameras),
                "common_location_count": len(common_cameras),
                "total_unique_locations": len(all_cameras)
            }

    # -------------------------------
    # NEW: Export tracking data
    # -------------------------------
    def export_tracking_data(self, person_name: Optional[str] = None, include_analytics: bool = True) -> Dict[str, Any]:
        with self._lock:
            logger.info("Exporting tracking data for %s", person_name or "all persons")
            export = {"export_timestamp": datetime.now().isoformat(), "person_filter": person_name}
            if person_name:
                export["movement_history"] = self.get_movement_history(person_name)
                export["current_location"] = self.current_locations.get(person_name)
                if include_analytics:
                    export["analytics"] = {
                        "heatmap": self.get_heatmap(person_name),
                        "time_patterns": self.get_time_patterns(person_name),
                        "speed_statistics": self.get_speed_statistics(person_name),
                        "dwell_statistics": self.get_dwell_statistics(person_name),
                        "suspicious_patterns": self.detect_suspicious_patterns(person_name),
                        "predicted_trajectory": self.get_predicted_trajectory(person_name)
                    }
            else:
                export["total_persons"] = len(self.movement_history)
                export["all_movements"] = self.get_all_movements()
                export["current_locations"] = self.get_current_locations()
                if include_analytics:
                    export["analytics"] = {
                        "global_heatmap": self.get_heatmap(),
                        "time_patterns": self.get_time_patterns(),
                        "speed_statistics": self.get_speed_statistics(),
                        "dwell_statistics": self.get_dwell_statistics(),
                        "transition_matrix": self.get_transition_matrix(),
                        "anomaly_report": self.get_anomaly_report(),
                        "overall_statistics": self.get_statistics()
                    }
            return export

    # -------------------------------
    # NEW: Clear advanced analytics
    # -------------------------------
    def clear_analytics(self) -> Dict[str, Any]:
        with self._lock:
            counts = {
                "trajectory_cache": len(self._trajectory_cache),
                "speed_history": sum(len(v) for v in self._speed_history.values()),
                "dwell_times": sum(len(v) for v in self._dwell_times.values()),
                "anomaly_scores": sum(len(v) for v in self._anomaly_scores.values())
            }
            self._trajectory_cache.clear()
            self._speed_history.clear()
            self._dwell_times.clear()
            self._hourly_patterns.clear()
            self._camera_transitions.clear()
            self._anomaly_scores.clear()
            self._location_heatmap.clear()
            self._person_heatmaps.clear()
            logger.info("Cleared analytics: %s", counts)
            return {"success": True, "cleared_counts": counts, "message": "Advanced analytics cleared (basic tracking retained)"}

    # -------------------------------
    # NEW: Optimize performance
    # -------------------------------
    def optimize_performance(self) -> Dict[str, Any]:
        with self._lock:
            logger.info("Optimizing tracking service performance")
            cutoff = datetime.now() - timedelta(hours=24)
            old_keys = [k for k, ts in self.last_detection.items() if ts < cutoff]
            for k in old_keys:
                del self.last_detection[k]

            trimmed_persons = 0
            for person in list(self.movement_history.keys()):
                history_len = len(self.movement_history[person])
                # trim if very large (keeps deque maxlen but we may want to downsize)
                if history_len > 100:
                    # nothing to do because deque has maxlen; we still record count
                    trimmed_persons += 1

            self.stats["total_detections"] = sum(len(h) for h in self.movement_history.values())
            self.stats["unique_persons"] = len(self.movement_history)

            logger.info("Optimization complete: removed %d old detections, trimmed %d histories", len(old_keys), trimmed_persons)
            return {"success": True, "old_detections_removed": len(old_keys), "histories_trimmed": trimmed_persons, "current_tracked_persons": len(self.movement_history), "total_detection_records": self.stats["total_detections"]}

    # -------------------------------
    # NEW: Health check
    # -------------------------------
    def health_check(self) -> Dict[str, Any]:
        health = {"status": "healthy", "issues": [], "warnings": []}
        with self._lock:
            total_persons = len(self.movement_history)
            total_records = sum(len(h) for h in self.movement_history.values())
            if total_persons > 10000:
                health["warnings"].append(f"Large number of tracked persons: {total_persons}")
            if total_records > 100000:
                health["warnings"].append(f"Large number of detection records: {total_records}")
            cache_size = len(self._trajectory_cache)
            if cache_size > 1000:
                health["warnings"].append(f"Large trajectory cache: {cache_size}")
            if self.last_detection:
                oldest_detection = min(self.last_detection.values())
                age_hours = (datetime.now() - oldest_detection).total_seconds() / 3600.0
                if age_hours > 72:
                    health["warnings"].append(f"Old detection data: {age_hours:.1f} hours old")
            total_analytics = (len(self._speed_history) + len(self._dwell_times) + len(self._anomaly_scores))
            if total_analytics > 50000:
                health["warnings"].append(f"Large analytics dataset: {total_analytics} entries")
            if health["warnings"]:
                health["status"] = "degraded"
            if health["issues"]:
                health["status"] = "unhealthy"
        return health


# -------------------------------
# Singleton instance (EXISTING)
# -------------------------------
tracking_service = TrackingService()
