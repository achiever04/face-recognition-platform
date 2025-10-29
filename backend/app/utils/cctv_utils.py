"""
CCTV Utilities for processing live video feeds.
- Detects faces from multiple CCTV cameras.
- Filters out fake faces (photos, videos, DeepFakes) using DeepfakeDetector.
- Logs real faces for criminal identification.
"""

import cv2
from typing import List, Tuple, Dict, Any  # --- NEW: Added type hints
from app.utils.deepfake_utils import DeepfakeDetector

class CCTVProcessor:
    def __init__(self, camera_sources: List[int]):  # --- IMPROVED: Type hint
        """
        Args:
            camera_sources (list): List of camera URLs or device IDs (e.g., [0, 1, 4])
            
        IMPROVEMENT: Now stores camera_id (source) with its capture object
        and keys frame_counters by camera_id for correct tracking.
        """
        self.camera_sources = camera_sources
        self.detector = DeepfakeDetector()
        
        # --- IMPROVED: Store (source_id, capture_object) tuples ---
        self.captures: List[Tuple[int, cv2.VideoCapture]] = []
        
        # --- IMPROVED: Key counters by actual camera_id ---
        self.frame_counters: Dict[int, int] = {}
        
        # Initialize captures with error handling
        for src in self.camera_sources: # --- IMPROVED: Iterate by source, not index
            try:
                cap = cv2.VideoCapture(src)
                if cap.isOpened():
                    self.captures.append((src, cap)) # Store (id, cap)
                    self.frame_counters[src] = 0     # Key by id
                else:
                    print(f"Warning: Could not open camera source: {src}")
                    self.captures.append((src, None)) # Store (id, None)
            except Exception as e:
                print(f"Error initializing camera {src}: {e}")
                self.captures.append((src, None))

    def process_frame(self, frame: cv2.typing.MatLike, camera_id: int) -> List[Dict[str, Any]]:
        """
        Process a single frame from a camera.
        - Detect faces.
        - Classify as real/fake.
        - Log results.

        Args:
            frame (np.ndarray): BGR frame from cv2
            camera_id (int): The ACTUAL camera source ID (e.g., 0, 1, 4)

        Returns:
            List of detected faces with classification
        """
        if frame is None:
            return []
        
        detections = self.detector.detect_and_classify(frame)
        
        # Increment frame counter for this specific camera_id
        if camera_id in self.frame_counters:
            self.frame_counters[camera_id] += 1
        else:
            self.frame_counters[camera_id] = 1
        
        # current_frame_id = self.frame_counters[camera_id]
        
        # --- IMPROVEMENT: Logging logic removed from this utility class. ---
        #
        # The router 'routes/deepfake.py' now handles all logging to
        # the central MongoDB database (using 'utils/db.py').
        # This prevents duplicate logs and correctly logs FAKE faces
        # instead of REAL faces, which was a bug here.
        #
        # The original code is commented out below to adhere to the
        # "no removal" rule.
        #
        # for det in detections:
        #     # Log only real faces
        #     if not det['is_fake']:
        #         self.detector.log_deepfake_event(
        #             camera_id=camera_id,
        #             frame_id=current_frame_id,
        #             detection=det
        #         )
        
        return detections

    def run(self, max_frames=None):
        """
        Run live processing for all cameras.
        Yields processed frames with detection overlays.
        
        Args:
            max_frames (int): Maximum frames to process (None = unlimited)
        
        Yields:
            tuple: (camera_id, frame, detections)
            
        IMPROVEMENT: Yields the ACTUAL camera_id (e.g., 0, 1, 4) instead
        of the list index (0, 1, 2).
        """
        frame_count = 0
        
        try:
            while max_frames is None or frame_count < max_frames:
                
                # --- IMPROVED: Iterate over (id, cap) tuples ---
                for cam_id_src, cap in self.captures:
                    
                    if cap is None or not cap.isOpened():
                        continue
                    
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    
                    # --- IMPROVED: Pass the real camera_id ---
                    detections = self.process_frame(frame, cam_id_src)

                    # Draw bounding boxes and labels on frame
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        label = "Fake" if det['is_fake'] else "Real"
                        color = (0, 0, 255) if det['is_fake'] else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            f"{label} {det['confidence']:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                    
                    # --- IMPROVED: Yield the real camera_id ---
                    yield cam_id_src, frame, detections
                    frame_count += 1
                
                # Allow manual exit with 'q' key (if running with display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Ensure resources are released
            self.release()

    def release(self):
        """
        Release all camera resources.
        """
        for src, cap in self.captures:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()