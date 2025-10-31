"""
CCTV Utilities for processing live video feeds.

Responsibilities:
- Manage multiple camera VideoCapture objects
- Sample frames, detect faces and classify them (via DeepfakeDetector)
- Yield (camera_id, frame, detections) tuples for downstream processing

Design notes:
- Uses the ACTUAL camera source ID (e.g., 0 or an RTSP URL) as camera identifier.
- This utility intentionally does NOT perform persistent logging; routing layer
  (routes/deepfake.py) is responsible for storing results in DB/logs.
"""

from typing import List, Tuple, Dict, Any, Iterator, Optional
import logging
from pathlib import Path

import cv2
import numpy as np
from app.utils.deepfake_utils import DeepfakeDetector

logger = logging.getLogger(__name__)


class CCTVProcessor:
    def __init__(self, camera_sources: List[Any]):
        """
        Args:
            camera_sources: list of camera source identifiers (device index ints or stream URLs)
        """
        self.camera_sources: List[Any] = camera_sources
        self.detector = DeepfakeDetector()
        self.captures: List[Tuple[Any, Optional[cv2.VideoCapture]]] = []
        self.frame_counters: Dict[Any, int] = {}

        # Initialize captures with error handling
        for src in self.camera_sources:
            try:
                cap = cv2.VideoCapture(src)
                if cap is not None and cap.isOpened():
                    self.captures.append((src, cap))
                    self.frame_counters[src] = 0
                    logger.info("Opened camera source: %s", src)
                else:
                    # Store a placeholder (None) so code is aware of the camera but knows it's closed
                    self.captures.append((src, None))
                    logger.warning("Could not open camera source: %s", src)
            except Exception as e:
                self.captures.append((src, None))
                logger.exception("Error initializing camera %s: %s", src, e)

    def process_frame(self, frame: "np.ndarray", camera_id: Any) -> List[Dict[str, Any]]:
        """
        Process a single frame from a camera:
          - Detect faces using the detector
          - Return detection list produced by DeepfakeDetector

        Returns list of detection dicts:
            {
                'bbox': [x1, y1, x2, y2],
                'is_fake': bool,
                'confidence': float
            }
        """
        if frame is None:
            return []

        try:
            detections = self.detector.detect_and_classify(frame)
        except Exception as e:
            logger.exception("Detector failure on camera %s: %s", camera_id, e)
            detections = []

        # Update frame counter keyed by camera_id
        self.frame_counters[camera_id] = self.frame_counters.get(camera_id, 0) + 1

        return detections

    def run(self, max_frames: Optional[int] = None) -> Iterator[Tuple[Any, "np.ndarray", List[Dict[str, Any]]]]:
        """
        Iterate over configured cameras and yield processed frames.

        Yields:
            (camera_id, frame, detections)
        """
        frame_count = 0
        try:
            while max_frames is None or frame_count < max_frames:
                for cam_id_src, cap in self.captures:
                    if cap is None or not cap.isOpened():
                        # Skip closed/unavailable camera
                        continue

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.debug("No frame captured from camera %s", cam_id_src)
                        continue

                    detections = self.process_frame(frame, cam_id_src)

                    # Draw overlays (non-destructive - copies are not created to save memory)
                    for det in detections:
                        try:
                            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
                            # Clamp bbox to image boundaries
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w - 1, x2), min(h - 1, y2)
                            label = "Fake" if det.get("is_fake") else "Real"
                            conf = det.get("confidence", 0.0)
                            color = (0, 0, 255) if det.get("is_fake") else (0, 255, 0)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(int(y1) - 8, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception:
                            logger.exception("Failed to draw detection overlay for camera %s", cam_id_src)

                    yield cam_id_src, frame, detections
                    frame_count += 1

                # Give OpenCV a chance to process GUI events (if any); does not block if no GUI
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Received 'q' keyboard event - exiting CCTVProcessor.run() loop")
                    break

        except GeneratorExit:
            # Allow graceful generator close
            logger.debug("CCTVProcessor.run() generator closed")
        except Exception as e:
            logger.exception("Error while running CCTVProcessor: %s", e)
        finally:
            self.release()

    def release(self) -> None:
        """Release all VideoCapture objects and destroy OpenCV windows."""
        for src, cap in self.captures:
            try:
                if cap is not None:
                    cap.release()
                    logger.info("Released camera %s", src)
            except Exception:
                logger.exception("Failed to release camera %s", src)
        try:
            cv2.destroyAllWindows()
        except Exception:
            # In some headless environments this can raise; ignore
            logger.debug("cv2.destroyAllWindows() raised exception (likely headless). Ignoring.")
