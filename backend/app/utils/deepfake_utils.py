"""
DeepFake detection and face detection utilities.

- Uses InsightFace/RetinaFace for face detection when available.
- Uses a MobileNetV3-based classifier (2-class) for deepfake prediction when trained weights are present.
- Falls back to OpenCV Haar cascade for face detection and default behavior when models are unavailable.

Class: DeepfakeDetector
Methods:
    - detect_and_classify(frame) -> List[dict]
    - _classify_face(crop) -> (is_fake: bool, confidence: float)
    - _fallback_detection(frame) -> List[dict]
    - log_deepfake_event(camera_id, frame_id, detection) -> None (legacy)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    def __init__(self, device: Optional[torch.device] = None):
        # Device setup
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        logger.info("Initializing DeepfakeDetector on device: %s", self.device)

        # Try to initialize InsightFace RetinaFace (optional)
        self.face_app = None
        try:
            from insightface.app import FaceAnalysis  # type: ignore
            # Attempt to prepare with CPU provider first; if GPU desired, the user can adjust instantiation externally
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            # prepare may require ctx_id; for CPU we use default behavior; attempt a couple of variants
            try:
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception:
                # Fall back to default prepare invocation if ctx_id fails
                try:
                    self.face_app.prepare(det_size=(640, 640))
                except Exception:
                    logger.exception("InsightFace prepare failed with both ctx_id and default prepare")
            logger.info("InsightFace FaceAnalysis (RetinaFace) loaded successfully.")
        except Exception as e:
            logger.warning("InsightFace/RetinaFace initialization failed: %s. Falling back to OpenCV detection.", e)
            self.face_app = None

        # Initialize deepfake classification model
        self.model: Optional[torch.nn.Module] = None
        try:
            # Load MobileNet V3 small backbone (ImageNet pre-trained)
            model = mobilenet_v3_small(pretrained=True)
            # Attempt a safe modification of classifier to 2 classes
            try:
                # Many torchvision versions use model.classifier as nn.Sequential; handle gracefully
                if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
                    # Attempt to find final Linear and replace
                    last_idx = len(model.classifier) - 1
                    if isinstance(model.classifier[last_idx], torch.nn.Linear):
                        in_features = model.classifier[last_idx].in_features
                        model.classifier[last_idx] = torch.nn.Linear(in_features=in_features, out_features=2)
                    else:
                        # As a fallback, append a new Linear layer
                        try:
                            model.classifier.add_module("fc_out", torch.nn.Linear(1024, 2))
                        except Exception:
                            logger.debug("Could not append fc_out to classifier; continuing with best-effort modification.")
                else:
                    # If classifier shape is unexpected, try to set an attribute safely
                    try:
                        model.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 2))
                    except Exception:
                        logger.warning("Unexpected MobileNet classifier structure; classifier modification attempted but may be incorrect.")
                self.model = model
            except Exception as e:
                logger.exception("Failed to modify MobileNet classifier: %s", e)
                self.model = model  # keep original to avoid crashing
        except Exception as e:
            logger.exception("Failed to instantiate MobileNetV3 backbone: %s", e)
            self.model = None

        # Attempt to load trained weights if available
        self.model_path = Path("models") / "deepfake_mobilenet.pth"
        if self.model is not None:
            if self.model_path.exists():
                try:
                    state = torch.load(str(self.model_path), map_location=self.device)
                    # If state is a dict with 'model' key, try to handle common wrappers
                    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                        state = state["model"]
                    self.model.load_state_dict(state)
                    logger.info("Loaded deepfake model weights from %s", self.model_path)
                except Exception as e:
                    logger.exception("Failed to load model weights from %s: %s", self.model_path, e)
                    logger.warning("Model will operate without trained weights (predictions may be random).")
            else:
                logger.warning("Deepfake model weights not found at %s. The classifier will run untrained if used.", self.model_path)

            try:
                self.model = self.model.to(self.device)
                self.model.eval()
            except Exception:
                logger.exception("Failed to move model to device or set to eval mode.")

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def detect_and_classify(self, frame: "np.ndarray") -> List[Dict[str, Any]]:
        """
        Detect faces in the input BGR frame and classify each as real/fake.

        Returns:
            list of dicts: {"bbox": [x1,y1,x2,y2], "is_fake": bool, "confidence": float}
        """
        results: List[Dict[str, Any]] = []
        if frame is None or getattr(frame, "size", 0) == 0:
            return results

        try:
            # Use RetinaFace/InsightFace if available
            if self.face_app is not None:
                faces = []
                try:
                    # InsightFace FaceAnalysis expects BGR images; .get returns face objects
                    faces = self.face_app.get(frame)
                except Exception:
                    # Some versions may use .detect or .get; try .detect as fallback
                    try:
                        faces = self.face_app.detect(frame)
                    except Exception as e:
                        logger.exception("InsightFace detection invocation failed: %s", e)
                        faces = []

                for face in faces:
                    try:
                        bbox = getattr(face, "bbox", None)
                        if bbox is None:
                            # Some face objects have 'bbox' or 'bbox_2d' or similar
                            continue
                        x1, y1, x2, y2 = bbox.astype(int)
                        # basic bbox validation and clamping
                        h, w = frame.shape[:2]
                        if x2 <= x1 or y2 <= y1 or x1 >= w or y1 >= h:
                            continue
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        if self.model is not None:
                            is_fake, confidence = self._classify_face(crop)
                        else:
                            is_fake, confidence = False, 0.5
                        results.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "is_fake": is_fake,
                            "confidence": confidence
                        })
                    except Exception:
                        logger.exception("Error processing face object returned by face_app")
            else:
                # Fallback detection using Haar cascade/OpenCV
                results = self._fallback_detection(frame)
        except Exception as e:
            logger.exception("Error in detect_and_classify: %s", e)

        return results

    def _classify_face(self, crop: "np.ndarray") -> Tuple[bool, float]:
        """
        Classify a face crop (BGR) as fake/real using the model.
        Returns (is_fake, confidence) where confidence is probability of predicted class.
        """
        try:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                # By convention: index 0 -> real, index 1 -> fake (matching training)
                predicted = int(np.argmax(probs))
                is_fake = bool(predicted == 1)
                confidence = float(probs[1] if is_fake else probs[0])
            return is_fake, confidence
        except Exception as e:
            logger.exception("Classification error: %s", e)
            # On failure, default to 'real' with neutral confidence
            return False, 0.5

    def _fallback_detection(self, frame: "np.ndarray") -> List[Dict[str, Any]]:
        """
        Use OpenCV Haar Cascade to find faces when InsightFace isn't available.
        Returns simple detections with is_fake=False and confidence=0.5
        """
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if not os.path.exists(cascade_path):
                logger.warning("Haar cascade file not found at %s", cascade_path)
                return []

            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            results: List[Dict[str, Any]] = []
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "is_fake": False,
                    "confidence": 0.5
                })
            return results
        except Exception as e:
            logger.exception("Fallback detection error: %s", e)
            return []

    def log_deepfake_event(self, camera_id: int, frame_id: int, detection: Dict[str, Any]) -> None:
        """
        Legacy/local logging helper. Kept for backwards compatibility.
        Prefer centralized DB logging in routes/deepfake.py -> utils/db.py
        """
        try:
            os.makedirs("data/deepfake_logs", exist_ok=True)
            log_path = Path("data/deepfake_logs") / "events.log"
            entry = (
                f"{datetime_now_iso()} | Camera {camera_id} | Frame {frame_id} | "
                f"Fake={detection.get('is_fake')} | Confidence={float(detection.get('confidence', 0.0)):.2f} | "
                f"BBox={detection.get('bbox')}\n"
            )
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(entry)
        except Exception:
            logger.exception("Failed to write local deepfake log")

def datetime_now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
