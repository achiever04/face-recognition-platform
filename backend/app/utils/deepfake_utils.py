"""
DeepFake detection and face detection utilities.
Uses RetinaFace (from insightface) for fast & accurate face detection
and MobileNetV3 for DeepFake classification.
"""

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

class DeepfakeDetector:
    def __init__(self, device=None):
        # Device setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DeepfakeDetector] Initializing with device: {self.device}")
        
        # Initialize RetinaFace (InsightFace)
        self.face_app = None
        try:
            from insightface.app import FaceAnalysis
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("[DeepfakeDetector] InsightFace (RetinaFace) loaded successfully.")
        except Exception as e:
            print(f"[DeepfakeDetector] Warning: Could not initialize RetinaFace: {e}")
            print("[DeepfakeDetector] Falling back to basic OpenCV detection")
        
        # Initialize DeepFake Detection Model
        try:
            self.model = mobilenet_v3_small(pretrained=True) # Start with ImageNet weights
            # Modify the final layer for 2 classes (real, fake)
            self.model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=2)
            
            # --- IMPROVEMENT: Robust model weight loading ---
            self.model_path = "models/deepfake_mobilenet.pth"
            
            if os.path.exists(self.model_path):
                print(f"[DeepfakeDetector] Found model weights at {self.model_path}. Loading...")
                # Load the trained weights onto the correct device
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                print("[DeepfakeDetector] Deepfake model weights loaded successfully.")
            else:
                # The original TODO is now a functional, non-blocking warning
                print("="*50)
                print(f"[DeepfakeDetector] WARNING: Model weights file not found!")
                print(f"  Expected at: {os.path.abspath(self.model_path)}")
                print("[DeepfakeDetector] The model will give RANDOM (untrained) predictions.")
                print("[DeepfakeDetector] Please download the trained .pth file to this location")
                print("="*50)
            
            self.model = self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            
        except Exception as e:
            print(f"[DeepfakeDetector] CRITICAL Error initializing model: {e}")
            self.model = None
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def detect_and_classify(self, frame):
        """
        Run RetinaFace to detect faces and MobileNetV3 to classify as real/fake.

        Args:
            frame (np.ndarray): BGR image from cv2

        Returns:
            results (list): [
                {
                    "bbox": [x1, y1, x2, y2],
                    "is_fake": bool,
                    "confidence": float
                },
                ...
            ]
        """
        results = []
        
        if frame is None or frame.size == 0:
            return results
        
        try:
            # Use RetinaFace if available
            if self.face_app is not None:
                faces = self.face_app.get(frame)
                
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    
                    # Validate bbox
                    if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                        continue
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                    
                    # Classify if model available
                    if self.model is not None:
                        is_fake, confidence = self._classify_face(crop)
                    else:
                        # Fallback: assume real if no model
                        is_fake = False
                        confidence = 0.5
                    
                    results.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "is_fake": is_fake,
                        "confidence": confidence
                    })
            else:
                # Fallback: use basic face detection
                results = self._fallback_detection(frame)
                
        except Exception as e:
            print(f"[DeepFakeDetector] Error in detect_and_classify: {e}")
        
        return results
    
    def _classify_face(self, crop):
        """Classify a face crop as real/fake"""
        try:
            # Convert BGR crop to RGB for PyTorch model
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(img_tensor)
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Class 0 = real, Class 1 = fake
                is_fake = bool(np.argmax(probs) == 1)
                
                # Confidence is the probability of the predicted class
                confidence = float(probs[1] if is_fake else probs[0])
            
            return is_fake, confidence
        except Exception as e:
            print(f"[DeepfakeDetector] Classification error: {e}")
            return False, 0.5 # Default to 'real' on error
    
    def _fallback_detection(self, frame):
        """Basic face detection without RetinaFace"""
        try:
            # Use OpenCV's Haar Cascade as fallback
            face_cascade_path = os.path.join(
                cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'
            )
            if not os.path.exists(face_cascade_path):
                print("[DeepfakeDetector] Fallback 'haarcascade' file not found.")
                return []
                
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "is_fake": False,  # Can't classify without model
                    "confidence": 0.5
                })
            return results
        except Exception as e:
            print(f"[DeepfakeDetector] Fallback detection error: {e}")
            return []

    def log_deepfake_event(self, camera_id: int, frame_id: int, detection: dict):
        """
        --- DEPRECATED ---
        This function logs to a local text file and is no longer
        recommended. Logging is now handled by the 'log_deepfake'
        function in 'utils/db.py' from the 'routes/deepfake.py' file,
        which saves to MongoDB for a persistent, queryable audit trail.
        This function is kept for compatibility only.
        """
        try:
            os.makedirs("data/deepfake_logs", exist_ok=True)
            log_path = os.path.join("data/deepfake_logs", "events.log")
            
            log_entry = (
                f"Camera {camera_id} | Frame {frame_id} | "
                f"Fake={detection['is_fake']} | Confidence={detection['confidence']:.2f} | "
                f"BBox={detection['bbox']}\n"
            )
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[DeepfakeDetector] Failed to log event: {e}")