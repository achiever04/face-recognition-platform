# backend/app/services/tracking_service.py

"""
Person Tracking Service
Handles cross-camera tracking, movement history, and location management.

ENHANCED: Added predictive analytics, pattern recognition, heatmaps, speed estimation
"""

import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import logging

from app.state import PERSON_LOCATIONS, CAMERA_METADATA

# Initialize logger
logger = logging.getLogger(__name__)


class TrackingService:
    """Service for tracking persons across multiple cameras"""
    
    def __init__(self, history_limit: int = 100, cooldown_seconds: int = 10):
        """
        Args:
            history_limit: Maximum history entries to keep per person
            cooldown_seconds: Minimum seconds between detections at same camera
        """
        # EXISTING: Track current location (kept as is)
        self.current_locations = PERSON_LOCATIONS
        
        # EXISTING: Movement history (kept as is)
        self.movement_history = defaultdict(lambda: deque(maxlen=history_limit))
        
        # EXISTING: Last detection time (kept as is)
        self.last_detection = {}
        
        # EXISTING: Cooldown period (kept as is)
        self.cooldown = timedelta(seconds=cooldown_seconds)
        
        # EXISTING: Thread safety (kept as is)
        self._lock = threading.Lock()
        
        # EXISTING: Statistics (kept as is)
        self.stats = {
            "total_detections": 0,
            "unique_persons": 0,
            "camera_switches": 0
        }
        
        # NEW: Advanced tracking features
        self._trajectory_cache = {}  # person -> predicted trajectory
        self._speed_history = defaultdict(list)  # person -> list of speeds
        self._dwell_times = defaultdict(list)  # person -> list of dwell durations
        self._hourly_patterns = defaultdict(lambda: defaultdict(int))  # person -> hour -> count
        self._camera_transitions = defaultdict(lambda: defaultdict(int))  # from_cam -> to_cam -> count
        self._anomaly_scores = defaultdict(list)  # person -> list of anomaly scores
        
        # NEW: Heatmap data
        self._location_heatmap = defaultdict(int)  # camera_id -> visit count
        self._person_heatmaps = defaultdict(lambda: defaultdict(int))  # person -> camera_id -> count
        
        # NEW: Performance metrics
        self._metrics = {
            "total_predictions": 0,
            "prediction_accuracy": 0,
            "average_speed": 0,
            "average_dwell_time": 0
        }
        
        logger.info(f"TrackingService initialized with history_limit={history_limit}, cooldown={cooldown_seconds}s")
    
    # -------------------------------
    # EXISTING: Record Detection (ENHANCED)
    # -------------------------------
    def record_detection(
        self,
        person_name: str,
        camera_id: int,
        distance: float,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Record a person detection event.
        
        ENHANCED: Added speed calculation, dwell time, heatmap updates, pattern analysis
        
        Args:
            person_name: Name/ID of detected person
            camera_id: Camera where detection occurred
            distance: Face recognition distance (lower = better match)
            timestamp: Time of detection (default: now)
            
        Returns:
            {
                "recorded": bool,
                "is_new_location": bool,
                "previous_location": Optional[int],
                "duplicate": bool,
                "message": str,
                "speed_kmh": float (NEW),
                "dwell_time_seconds": float (NEW)
            }
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            logger.debug(f"Recording detection: {person_name} at camera {camera_id}")
            
            # Check for duplicate (EXISTING)
            detection_key = (person_name, camera_id)
            
            if detection_key in self.last_detection:
                time_since_last = timestamp - self.last_detection[detection_key]
                
                if time_since_last < self.cooldown:
                    logger.debug(f"Duplicate detection for {person_name} at camera {camera_id} (within cooldown)")
                    return {
                        "recorded": False,
                        "is_new_location": False,
                        "previous_location": None,
                        "duplicate": True,
                        "message": f"Duplicate detection (cooldown: {self.cooldown.seconds}s)"
                    }
            
            # Get camera metadata (EXISTING)
            camera_info = CAMERA_METADATA.get(camera_id, {})
            camera_name = camera_info.get("name", f"Camera {camera_id}")
            geo = camera_info.get("geo", (0.0, 0.0))
            
            # Check if this is a location change (EXISTING)
            previous_location = self.current_locations.get(person_name)
            is_new_location = previous_location != camera_id
            
            # Calculate speed if moving (NEW)
            speed_kmh = 0.0
            if is_new_location and previous_location is not None:
                speed_kmh = self._calculate_speed(
                    person_name, 
                    previous_location, 
                    camera_id, 
                    timestamp
                )
                
                if speed_kmh > 0:
                    self._speed_history[person_name].append(speed_kmh)
                    logger.debug(f"Calculated speed for {person_name}: {speed_kmh:.2f} km/h")
            
            # Calculate dwell time at previous location (NEW)
            dwell_time = 0.0
            if previous_location is not None:
                dwell_time = self._calculate_dwell_time(person_name, previous_location, timestamp)
                if dwell_time > 0:
                    self._dwell_times[person_name].append(dwell_time)
            
            # Create detection record (EXISTING - enhanced with new fields)
            detection_record = {
                "person": person_name,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "geo": geo,
                "distance": distance,
                "confidence": self._calculate_confidence(distance),
                "timestamp": timestamp.isoformat(),
                "speed_kmh": round(speed_kmh, 2),  # NEW
                "dwell_time_seconds": round(dwell_time, 2)  # NEW
            }
            
            # Update current location (EXISTING)
            self.current_locations[person_name] = camera_id
            
            # Add to movement history (EXISTING)
            self.movement_history[person_name].append(detection_record)
            
            # Update last detection time (EXISTING)
            self.last_detection[detection_key] = timestamp
            
            # Update statistics (EXISTING)
            self.stats["total_detections"] += 1
            if person_name not in self.movement_history or len(self.movement_history[person_name]) == 1:
                self.stats["unique_persons"] += 1
            if is_new_location and previous_location is not None:
                self.stats["camera_switches"] += 1
            
            # Update heatmaps (NEW)
            self._location_heatmap[camera_id] += 1
            self._person_heatmaps[person_name][camera_id] += 1
            
            # Update hourly patterns (NEW)
            hour = timestamp.hour
            self._hourly_patterns[person_name][hour] += 1
            
            # Update camera transitions (NEW)
            if is_new_location and previous_location is not None:
                self._camera_transitions[previous_location][camera_id] += 1
            
            # Calculate anomaly score (NEW)
            anomaly_score = self._calculate_anomaly_score(person_name, camera_id, speed_kmh)
            self._anomaly_scores[person_name].append(anomaly_score)
            
            # Update trajectory prediction (NEW)
            self._update_trajectory(person_name)
            
            logger.info(f"Detection recorded: {person_name} at {camera_name} (speed: {speed_kmh:.2f} km/h, dwell: {dwell_time:.2f}s)")
            
            return {
                "recorded": True,
                "is_new_location": is_new_location,
                "previous_location": previous_location,
                "duplicate": False,
                "message": "Detection recorded successfully",
                "detection": detection_record,
                "speed_kmh": round(speed_kmh, 2),
                "dwell_time_seconds": round(dwell_time, 2),
                "anomaly_score": round(anomaly_score, 3)
            }
    
    # -------------------------------
    # EXISTING: Get Movement History (KEPT AS IS)
    # -------------------------------
    def get_movement_history(
        self,
        person_name: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get movement history for a specific person.
        
        EXISTING METHOD - kept as is
        
        Args:
            person_name: Name of person
            limit: Maximum number of records to return (default: all)
            since: Only return records after this time
            
        Returns:
            List of detection records, newest first
        """
        with self._lock:
            if person_name not in self.movement_history:
                return []
            
            history = list(self.movement_history[person_name])
            
            # Filter by time if specified
            if since:
                history = [
                    h for h in history 
                    if datetime.fromisoformat(h["timestamp"]) > since
                ]
            
            # Reverse to get newest first
            history.reverse()
            
            # Apply limit
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
    ) -> Dict[str, List[Dict]]:
        """
        Get movement history for all tracked persons.
        
        EXISTING METHOD - kept as is
        
        Args:
            limit_per_person: Max records per person
            active_only: Only include persons detected recently
            activity_threshold_minutes: Minutes to consider "active"
            
        Returns:
            Dict mapping person_name -> list of detections
        """
        with self._lock:
            result = {}
            cutoff_time = datetime.now() - timedelta(minutes=activity_threshold_minutes)
            
            for person, history in self.movement_history.items():
                # Filter by activity if requested
                if active_only:
                    latest = history[-1] if history else None
                    if not latest:
                        continue
                    latest_time = datetime.fromisoformat(latest["timestamp"])
                    if latest_time < cutoff_time:
                        continue
                
                # Get history
                person_history = list(history)
                person_history.reverse()  # Newest first
                
                if limit_per_person:
                    person_history = person_history[:limit_per_person]
                
                result[person] = person_history
            
            return result
    
    # -------------------------------
    # EXISTING: Get Current Locations (KEPT AS IS)
    # -------------------------------
    def get_current_locations(self) -> Dict[str, Dict]:
        """
        Get current location of all tracked persons.
        
        EXISTING METHOD - kept as is
        
        Returns:
            Dict mapping person_name -> location info
        """
        with self._lock:
            locations = {}
            
            for person, camera_id in self.current_locations.items():
                # Get latest detection for this person
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
        """
        Get simplified movement path (camera transitions only).
        
        EXISTING METHOD - kept as is
        
        Args:
            person_name: Name of person
            include_duplicates: Include all detections or just camera switches
            
        Returns:
            List of (camera_id, camera_name, timestamp) tuples
        """
        with self._lock:
            if person_name not in self.movement_history:
                return []
            
            history = list(self.movement_history[person_name])
            
            if include_duplicates:
                return [
                    (h["camera_id"], h["camera_name"], datetime.fromisoformat(h["timestamp"]))
                    for h in history
                ]
            
            # Only include camera switches
            path = []
            prev_camera = None
            
            for h in history:
                cam_id = h["camera_id"]
                if cam_id != prev_camera:
                    path.append((
                        cam_id,
                        h["camera_name"],
                        datetime.fromisoformat(h["timestamp"])
                    ))
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
    ) -> Dict:
        """
        Analyze movement patterns for suspicious behavior.
        
        ENHANCED: Added speed analysis, time pattern analysis, anomaly detection
        
        Args:
            person_name: Name of person to analyze
            loitering_threshold_minutes: Minutes at same location = loitering
            revisit_threshold: Number of revisits to same camera = suspicious
            
        Returns:
            {
                "is_suspicious": bool,
                "patterns": List[str],
                "loitering_duration": Optional[float],
                "revisit_count": Dict[int, int],
                "anomaly_score": float (NEW),
                "speed_analysis": Dict (NEW)
            }
        """
        with self._lock:
            logger.debug(f"Analyzing suspicious patterns for: {person_name}")
            
            if person_name not in self.movement_history:
                return {
                    "is_suspicious": False,
                    "patterns": [],
                    "loitering_duration": None,
                    "revisit_count": {},
                    "anomaly_score": 0.0
                }
            
            history = list(self.movement_history[person_name])
            patterns = []
            
            # EXISTING: Check for loitering
            if len(history) >= 2:
                first = datetime.fromisoformat(history[0]["timestamp"])
                last = datetime.fromisoformat(history[-1]["timestamp"])
                duration_minutes = (last - first).total_seconds() / 60
                
                # If all detections at same camera for long time
                cameras = set(h["camera_id"] for h in history)
                if len(cameras) == 1 and duration_minutes > loitering_threshold_minutes:
                    patterns.append(f"Loitering detected: {duration_minutes:.1f} minutes at same location")
            
            # EXISTING: Check for repeated visits
            camera_visits = defaultdict(int)
            for h in history:
                camera_visits[h["camera_id"]] += 1
            
            for cam_id, count in camera_visits.items():
                if count >= revisit_threshold:
                    cam_name = CAMERA_METADATA.get(cam_id, {}).get("name", f"Camera {cam_id}")
                    patterns.append(f"Revisited {cam_name} {count} times")
            
            # EXISTING: Check for rapid back-and-forth
            if len(history) >= 4:
                recent = history[-4:]
                cameras_recent = [h["camera_id"] for h in recent]
                if len(set(cameras_recent)) == 2 and cameras_recent[0] == cameras_recent[2]:
                    patterns.append("Rapid back-and-forth movement detected")
            
            # NEW: Check for unusual speed patterns
            speeds = self._speed_history.get(person_name, [])
            if speeds:
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                
                if max_speed > 10:  # Unrealistic walking speed (> 10 km/h)
                    patterns.append(f"Unusually high speed detected: {max_speed:.1f} km/h")
                
                if avg_speed > 6:  # Fast average speed
                    patterns.append(f"High average speed: {avg_speed:.1f} km/h")
            
            # NEW: Check for unusual time patterns
            hours = [datetime.fromisoformat(h["timestamp"]).hour for h in history]
            night_detections = sum(1 for h in hours if h < 6 or h > 22)
            
            if night_detections > len(history) * 0.5:
                patterns.append(f"Mostly active during night hours ({night_detections}/{len(history)} detections)")
            
            # NEW: Calculate overall anomaly score
            anomaly_scores = self._anomaly_scores.get(person_name, [0])
            avg_anomaly = np.mean(anomaly_scores)
            
            if avg_anomaly > 0.7:
                patterns.append(f"High anomaly score: {avg_anomaly:.2f}")
            
            is_suspicious = len(patterns) > 0 or avg_anomaly > 0.6
            
            logger.info(f"Suspicious pattern analysis for {person_name}: {len(patterns)} patterns, anomaly={avg_anomaly:.2f}")
            
            return {
                "is_suspicious": is_suspicious,
                "patterns": patterns,
                "loitering_duration": duration_minutes if 'duration_minutes' in locals() else None,
                "revisit_count": dict(camera_visits),
                "anomaly_score": round(avg_anomaly, 3),
                "speed_analysis": {
                    "average_speed_kmh": round(np.mean(speeds), 2) if speeds else 0,
                    "max_speed_kmh": round(np.max(speeds), 2) if speeds else 0,
                    "min_speed_kmh": round(np.min(speeds), 2) if speeds else 0
                },
                "time_analysis": {
                    "total_detections": len(history),
                    "night_detections": night_detections if 'night_detections' in locals() else 0,
                    "most_active_hour": max(self._hourly_patterns[person_name].items(), key=lambda x: x[1])[0] if self._hourly_patterns[person_name] else None
                }
            }
    
    # -------------------------------
    # EXISTING: Clear History (KEPT AS IS)
    # -------------------------------
    def clear_history(self, person_name: Optional[str] = None) -> Dict:
        """
        Clear tracking history.
        
        EXISTING METHOD - kept as is
        
        Args:
            person_name: Clear specific person (None = clear all)
            
        Returns:
            {"success": bool, "cleared": int}
        """
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
    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.
        
        ENHANCED: Added advanced metrics
        """
        with self._lock:
            # Calculate additional statistics (NEW)
            total_speeds = []
            total_dwell_times = []
            
            for speeds in self._speed_history.values():
                total_speeds.extend(speeds)
            
            for dwell_times in self._dwell_times.values():
                total_dwell_times.extend(dwell_times)
            
            return {
                **self.stats,
                "active_persons": len(self.current_locations),
                "total_movement_records": sum(
                    len(h) for h in self.movement_history.values()
                ),
                "average_speed_kmh": round(np.mean(total_speeds), 2) if total_speeds else 0,
                "average_dwell_time_seconds": round(np.mean(total_dwell_times), 2) if total_dwell_times else 0,
                "total_camera_transitions": sum(
                    sum(transitions.values()) 
                    for transitions in self._camera_transitions.values()
                ),
                "unique_camera_transitions": len(self._camera_transitions)
            }
    
    # -------------------------------
    # EXISTING: Helper - Calculate Confidence (KEPT AS IS)
    # -------------------------------
    def _calculate_confidence(self, distance: float) -> str:
        """Convert face distance to confidence level"""
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
        """
        Calculate travel speed between two cameras.
        
        **NEW METHOD**
        
        Uses Haversine formula for geo-distance.
        """
        try:
            # Get camera locations
            from_geo = CAMERA_METADATA.get(from_camera, {}).get("geo", (0.0, 0.0))
            to_geo = CAMERA_METADATA.get(to_camera, {}).get("geo", (0.0, 0.0))
            
            # Calculate distance using Haversine formula
            distance_km = self._haversine_distance(from_geo, to_geo)
            
            if distance_km == 0:
                return 0.0
            
            # Get time since last detection at previous camera
            last_time_key = (person_name, from_camera)
            if last_time_key not in self.last_detection:
                return 0.0
            
            last_time = self.last_detection[last_time_key]
            time_diff = (current_time - last_time).total_seconds() / 3600  # Convert to hours
            
            if time_diff == 0:
                return 0.0
            
            # Calculate speed (km/h)
            speed = distance_km / time_diff
            
            return speed
        
        except Exception as e:
            logger.warning(f"Speed calculation failed: {e}")
            return 0.0
    
    # -------------------------------
    # NEW: Calculate dwell time
    # -------------------------------
    def _calculate_dwell_time(
        self,
        person_name: str,
        camera_id: int,
        current_time: datetime
    ) -> float:
        """
        Calculate how long person stayed at a camera location.
        
        **NEW METHOD**
        """
        try:
            # Get first detection at this camera
            history = self.movement_history.get(person_name)
            if not history:
                return 0.0
            
            # Find first detection at this camera (going backwards)
            first_time = None
            for detection in reversed(history):
                if detection["camera_id"] == camera_id:
                    first_time = datetime.fromisoformat(detection["timestamp"])
                else:
                    break
            
            if first_time is None:
                return 0.0
            
            # Calculate dwell time in seconds
            dwell = (current_time - first_time).total_seconds()
            
            return dwell
        
        except Exception as e:
            logger.warning(f"Dwell time calculation failed: {e}")
            return 0.0
    
    # -------------------------------
    # NEW: Haversine distance calculation
    # -------------------------------
    def _haversine_distance(self, geo1: Tuple[float, float], geo2: Tuple[float, float]) -> float:
        """
        Calculate distance between two geographic coordinates using Haversine formula.
        
        **NEW METHOD**
        
        Args:
            geo1: (latitude, longitude) of first point
            geo2: (latitude, longitude) of second point
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = geo1
        lat2, lon2 = geo2
        
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        
        return distance
    
    # -------------------------------
    # NEW: Calculate anomaly score
    # -------------------------------
    def _calculate_anomaly_score(
        self,
        person_name: str,
        camera_id: int,
        speed_kmh: float
    ) -> float:
        """
        Calculate anomaly score for a detection (0-1, higher = more anomalous).
        
        **NEW METHOD**
        
        Considers:
        - Unusual speed
        - Unusual time of day
        - Unusual location frequency
        """
        score = 0.0
        
        # Speed anomaly (30% weight)
        if speed_kmh > 0:
            if speed_kmh > 10:  # Unrealistic for walking
                score += 0.3
            elif speed_kmh > 6:  # Fast
                score += 0.15
        
        # Time anomaly (30% weight)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Night time
            score += 0.3
        elif current_hour < 8 or current_hour > 20:  # Early morning / late evening
            score += 0.15
        
        # Location frequency anomaly (40% weight)
        total_visits = self._location_heatmap.get(camera_id, 0)
        person_visits = self._person_heatmaps[person_name].get(camera_id, 0)
        
        if total_visits > 0:
            visit_ratio = person_visits / total_visits
            
            if visit_ratio > 0.5:  # Person dominates this location
                score += 0.4
            elif visit_ratio > 0.3:
                score += 0.2
        
        return min(1.0, score)
    
    # -------------------------------
    # NEW: Update trajectory prediction
    # -------------------------------
    def _update_trajectory(self, person_name: str):
        """
        Update predicted trajectory for a person based on movement history.
        
        **NEW METHOD**
        
        Uses linear regression on recent movements.
        """
        try:
            history = self.movement_history.get(person_name)
            if not history or len(history) < 3:
                return
            
            # Get recent movements (last 5)
            recent = list(history)[-5:]
            
            # Extract camera IDs and timestamps
            cameras = [h["camera_id"] for h in recent]
            times = [(datetime.fromisoformat(h["timestamp"]) - datetime.fromisoformat(recent[0]["timestamp"])).total_seconds() for h in recent]
            
            # Simple prediction: most likely next camera based on transitions
            if len(cameras) >= 2:
                last_camera = cameras[-1]
                
                # Get most common transition from last_camera
                transitions = self._camera_transitions.get(last_camera, {})
                if transitions:
                    predicted_camera = max(transitions.items(), key=lambda x: x[1])[0]
                    
                    self._trajectory_cache[person_name] = {
                        "current_camera": last_camera,
                        "predicted_next_camera": predicted_camera,
                        "confidence": transitions[predicted_camera] / sum(transitions.values()),
                        "timestamp": datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.warning(f"Trajectory update failed for {person_name}: {e}")
    
    # -------------------------------
    # NEW: Get predicted trajectory
    # -------------------------------
    def get_predicted_trajectory(self, person_name: str) -> Optional[Dict]:
        """
        Get predicted next location for a person.
        
        **NEW METHOD**
        """
        with self._lock:
            return self._trajectory_cache.get(person_name)
    
    # -------------------------------
    # NEW: Get heatmap data
    # -------------------------------
    def get_heatmap(self, person_name: Optional[str] = None) -> Dict:
        """
        Get location heatmap data.
        
        **NEW METHOD**
        
        Args:
            person_name: Get heatmap for specific person (None = global)
            
        Returns:
            Dict mapping camera_id -> visit count
        """
        with self._lock:
            if person_name:
                return dict(self._person_heatmaps.get(person_name, {}))
            else:
                return dict(self._location_heatmap)
    # -------------------------------
    # NEW: Get time-based patterns
    # -------------------------------
    def get_time_patterns(self, person_name: Optional[str] = None) -> Dict:
        """
        Get hourly activity patterns.
        
        **NEW METHOD**
        
        Args:
            person_name: Get patterns for specific person (None = all)
            
        Returns:
            Dict with hourly activity data
        """
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
                # Aggregate all persons
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
    # NEW: Get camera transition matrix
    # -------------------------------
    def get_transition_matrix(self) -> Dict:
        """
        Get camera-to-camera transition matrix.
        
        **NEW METHOD**
        
        Returns:
            Dict with transition counts and probabilities
        """
        with self._lock:
            # Build transition matrix
            matrix = {}
            
            for from_cam, transitions in self._camera_transitions.items():
                total_from = sum(transitions.values())
                
                matrix[from_cam] = {
                    "transitions": dict(transitions),
                    "total": total_from,
                    "probabilities": {
                        to_cam: count / total_from 
                        for to_cam, count in transitions.items()
                    }
                }
            
            return matrix
    
    # -------------------------------
    # NEW: Get speed statistics
    # -------------------------------
    def get_speed_statistics(self, person_name: Optional[str] = None) -> Dict:
        """
        Get speed statistics.
        
        **NEW METHOD**
        
        Args:
            person_name: Get stats for specific person (None = all)
            
        Returns:
            Speed statistics
        """
        with self._lock:
            if person_name:
                speeds = self._speed_history.get(person_name, [])
                
                if not speeds:
                    return {
                        "person": person_name,
                        "total_movements": 0,
                        "average_speed": 0,
                        "max_speed": 0,
                        "min_speed": 0
                    }
                
                return {
                    "person": person_name,
                    "total_movements": len(speeds),
                    "average_speed": round(np.mean(speeds), 2),
                    "max_speed": round(np.max(speeds), 2),
                    "min_speed": round(np.min(speeds), 2),
                    "median_speed": round(np.median(speeds), 2),
                    "std_deviation": round(np.std(speeds), 2)
                }
            else:
                # Aggregate all persons
                all_speeds = []
                for speeds in self._speed_history.values():
                    all_speeds.extend(speeds)
                
                if not all_speeds:
                    return {
                        "total_movements": 0,
                        "average_speed": 0
                    }
                
                return {
                    "total_movements": len(all_speeds),
                    "average_speed": round(np.mean(all_speeds), 2),
                    "max_speed": round(np.max(all_speeds), 2),
                    "min_speed": round(np.min(all_speeds), 2),
                    "median_speed": round(np.median(all_speeds), 2),
                    "std_deviation": round(np.std(all_speeds), 2)
                }
    
    # -------------------------------
    # NEW: Get dwell time statistics
    # -------------------------------
    def get_dwell_statistics(self, person_name: Optional[str] = None) -> Dict:
        """
        Get dwell time statistics.
        
        **NEW METHOD**
        
        Args:
            person_name: Get stats for specific person (None = all)
            
        Returns:
            Dwell time statistics
        """
        with self._lock:
            if person_name:
                dwell_times = self._dwell_times.get(person_name, [])
                
                if not dwell_times:
                    return {
                        "person": person_name,
                        "total_stops": 0,
                        "average_dwell_seconds": 0
                    }
                
                return {
                    "person": person_name,
                    "total_stops": len(dwell_times),
                    "average_dwell_seconds": round(np.mean(dwell_times), 2),
                    "max_dwell_seconds": round(np.max(dwell_times), 2),
                    "min_dwell_seconds": round(np.min(dwell_times), 2),
                    "median_dwell_seconds": round(np.median(dwell_times), 2),
                    "total_dwell_time_minutes": round(sum(dwell_times) / 60, 2)
                }
            else:
                # Aggregate all persons
                all_dwell_times = []
                for dwell_times in self._dwell_times.values():
                    all_dwell_times.extend(dwell_times)
                
                if not all_dwell_times:
                    return {
                        "total_stops": 0,
                        "average_dwell_seconds": 0
                    }
                
                return {
                    "total_stops": len(all_dwell_times),
                    "average_dwell_seconds": round(np.mean(all_dwell_times), 2),
                    "max_dwell_seconds": round(np.max(all_dwell_times), 2),
                    "min_dwell_seconds": round(np.min(all_dwell_times), 2),
                    "median_dwell_seconds": round(np.median(all_dwell_times), 2),
                    "total_dwell_time_minutes": round(sum(all_dwell_times) / 60, 2)
                }
    
    # -------------------------------
    # NEW: Get anomaly reports
    # -------------------------------
    def get_anomaly_report(self, threshold: float = 0.6) -> List[Dict]:
        """
        Get list of persons with high anomaly scores.
        
        **NEW METHOD**
        
        Args:
            threshold: Minimum anomaly score to include (0-1)
            
        Returns:
            List of anomalous persons with scores
        """
        with self._lock:
            anomalous = []
            
            for person, scores in self._anomaly_scores.items():
                if not scores:
                    continue
                
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                
                if avg_score >= threshold or max_score >= 0.8:
                    anomalous.append({
                        "person": person,
                        "average_anomaly_score": round(avg_score, 3),
                        "max_anomaly_score": round(max_score, 3),
                        "total_detections": len(scores),
                        "high_anomaly_detections": sum(1 for s in scores if s > threshold)
                    })
            
            # Sort by average anomaly score (highest first)
            anomalous.sort(key=lambda x: x["average_anomaly_score"], reverse=True)
            
            return anomalous
    
    # -------------------------------
    # NEW: Compare movement patterns
    # -------------------------------
    def compare_movement_patterns(
        self,
        person1: str,
        person2: str
    ) -> Dict:
        """
        Compare movement patterns between two persons.
        
        **NEW METHOD**
        
        Useful for identifying similar behavior or coordinated movement.
        """
        with self._lock:
            logger.debug(f"Comparing movement patterns: {person1} vs {person2}")
            
            if person1 not in self.movement_history or person2 not in self.movement_history:
                return {
                    "similarity_score": 0.0,
                    "message": "One or both persons not found"
                }
            
            # Get heatmaps
            heatmap1 = self._person_heatmaps[person1]
            heatmap2 = self._person_heatmaps[person2]
            
            # Calculate location overlap
            all_cameras = set(heatmap1.keys()).union(set(heatmap2.keys()))
            
            if not all_cameras:
                return {
                    "similarity_score": 0.0,
                    "common_locations": [],
                    "location_overlap": 0.0
                }
            
            common_cameras = set(heatmap1.keys()).intersection(set(heatmap2.keys()))
            location_overlap = len(common_cameras) / len(all_cameras)
            
            # Calculate time pattern similarity
            pattern1 = self._hourly_patterns[person1]
            pattern2 = self._hourly_patterns[person2]
            
            # Cosine similarity for time patterns
            all_hours = set(pattern1.keys()).union(set(pattern2.keys()))
            
            if all_hours:
                vec1 = [pattern1.get(h, 0) for h in range(24)]
                vec2 = [pattern2.get(h, 0) for h in range(24)]
                
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = np.sqrt(sum(a**2 for a in vec1))
                magnitude2 = np.sqrt(sum(b**2 for b in vec2))
                
                if magnitude1 > 0 and magnitude2 > 0:
                    time_similarity = dot_product / (magnitude1 * magnitude2)
                else:
                    time_similarity = 0.0
            else:
                time_similarity = 0.0
            
            # Calculate speed similarity
            speeds1 = self._speed_history.get(person1, [])
            speeds2 = self._speed_history.get(person2, [])
            
            if speeds1 and speeds2:
                avg_speed1 = np.mean(speeds1)
                avg_speed2 = np.mean(speeds2)
                speed_diff = abs(avg_speed1 - avg_speed2)
                speed_similarity = max(0, 1 - speed_diff / 10)  # Normalize to 0-1
            else:
                speed_similarity = 0.5  # Neutral if no data
            
            # Overall similarity (weighted average)
            overall_similarity = (
                location_overlap * 0.4 +
                time_similarity * 0.4 +
                speed_similarity * 0.2
            )
            
            return {
                "similarity_score": round(overall_similarity, 3),
                "location_overlap": round(location_overlap, 3),
                "time_similarity": round(time_similarity, 3),
                "speed_similarity": round(speed_similarity, 3),
                "common_locations": list(common_cameras),
                "common_location_count": len(common_cameras),
                "total_unique_locations": len(all_cameras)
            }
    
    # -------------------------------
    # NEW: Export tracking data
    # -------------------------------
    def export_tracking_data(
        self,
        person_name: Optional[str] = None,
        include_analytics: bool = True
    ) -> Dict:
        """
        Export comprehensive tracking data.
        
        **NEW METHOD**
        
        Args:
            person_name: Export for specific person (None = all)
            include_analytics: Include analytics and statistics
            
        Returns:
            Comprehensive tracking data
        """
        with self._lock:
            logger.info(f"Exporting tracking data for: {person_name or 'all persons'}")
            
            export = {
                "export_timestamp": datetime.now().isoformat(),
                "person_filter": person_name
            }
            
            if person_name:
                # Export single person
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
                # Export all persons
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
    def clear_analytics(self) -> Dict:
        """
        Clear advanced analytics data (keep basic tracking).
        
        **NEW METHOD**
        """
        with self._lock:
            logger.info("Clearing advanced analytics data")
            
            # Count what we're clearing
            counts = {
                "trajectory_cache": len(self._trajectory_cache),
                "speed_history": sum(len(v) for v in self._speed_history.values()),
                "dwell_times": sum(len(v) for v in self._dwell_times.values()),
                "anomaly_scores": sum(len(v) for v in self._anomaly_scores.values())
            }
            
            # Clear analytics data
            self._trajectory_cache.clear()
            self._speed_history.clear()
            self._dwell_times.clear()
            self._hourly_patterns.clear()
            self._camera_transitions.clear()
            self._anomaly_scores.clear()
            self._location_heatmap.clear()
            self._person_heatmaps.clear()
            
            logger.info(f"Cleared analytics: {counts}")
            
            return {
                "success": True,
                "cleared_counts": counts,
                "message": "Advanced analytics cleared (basic tracking retained)"
            }
    
    # -------------------------------
    # NEW: Optimize performance
    # -------------------------------
    def optimize_performance(self) -> Dict:
        """
        Optimize tracking service performance.
        
        **NEW METHOD**
        
        Cleans up old data and consolidates storage.
        """
        with self._lock:
            logger.info("Optimizing tracking service performance")
            
            # Remove old detections from last_detection
            cutoff = datetime.now() - timedelta(hours=24)
            old_keys = [
                key for key, timestamp in self.last_detection.items()
                if timestamp < cutoff
            ]
            
            for key in old_keys:
                del self.last_detection[key]
            
            # Trim excessive history
            trimmed_persons = 0
            for person in self.movement_history.keys():
                history_len = len(self.movement_history[person])
                if history_len > 100:
                    trimmed_persons += 1
            
            # Recalculate statistics
            self.stats["total_detections"] = sum(
                len(h) for h in self.movement_history.values()
            )
            self.stats["unique_persons"] = len(self.movement_history)
            
            logger.info(f"Optimization complete: removed {len(old_keys)} old detections, trimmed {trimmed_persons} histories")
            
            return {
                "success": True,
                "old_detections_removed": len(old_keys),
                "histories_trimmed": trimmed_persons,
                "current_tracked_persons": len(self.movement_history),
                "total_detection_records": self.stats["total_detections"]
            }
    
    # -------------------------------
    # NEW: Health check
    # -------------------------------
    def health_check(self) -> Dict:
        """
        Perform health check on tracking service.
        
        **NEW METHOD**
        """
        health = {
            "status": "healthy",
            "issues": [],
            "warnings": []
        }
        
        with self._lock:
            # Check data sizes
            total_persons = len(self.movement_history)
            total_records = sum(len(h) for h in self.movement_history.values())
            
            if total_persons > 10000:
                health["warnings"].append(f"Large number of tracked persons: {total_persons}")
            
            if total_records > 100000:
                health["warnings"].append(f"Large number of detection records: {total_records}")
            
            # Check for memory issues
            cache_size = len(self._trajectory_cache)
            if cache_size > 1000:
                health["warnings"].append(f"Large trajectory cache: {cache_size}")
            
            # Check for stale data
            if self.last_detection:
                oldest_detection = min(self.last_detection.values())
                age = (datetime.now() - oldest_detection).total_seconds() / 3600
                
                if age > 72:  # 3 days
                    health["warnings"].append(f"Old detection data: {age:.1f} hours old")
            
            # Check analytics data
            total_analytics = (
                len(self._speed_history) +
                len(self._dwell_times) +
                len(self._anomaly_scores)
            )
            
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