# backend/tools/mock_camera_worker.py
"""
Simple mock camera worker.
Sends images from backend/data/uploads/ to the backend ingestion endpoint periodically.
Configure via environment variables:
  - INGEST_URL (default: http://localhost:8000/api/camera/ingest)
  - CAM_ID (default: mock_cam_1)
  - INTERVAL (seconds, default: 2)
  - IMAGE_DIR (default: ./backend/data/uploads)
"""
import os
import time
import glob
import random
import requests
from pathlib import Path
from datetime import datetime

INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8000/api/camera/ingest")
CAM_ID = os.getenv("CAM_ID", "mock_cam_1")
INTERVAL = float(os.getenv("INTERVAL", "2.0"))
IMAGE_DIR = os.getenv("IMAGE_DIR", "/app/backend/data/uploads")  # inside container path for compose

def get_images(dirpath):
    p = Path(dirpath)
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for pat in patterns:
        files.extend(sorted(p.glob(pat)))
    return files

def send_frame(file_path):
    try:
        files = {"file": open(file_path, "rb")}
        data = {"cam_id": CAM_ID, "timestamp": datetime.utcnow().isoformat()}
        r = requests.post(INGEST_URL, files=files, data=data, timeout=10)
        print(f"[{datetime.utcnow().isoformat()}] Sent {file_path.name} -> {INGEST_URL} status={r.status_code}")
        return r.status_code, r.text
    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] Failed to send {file_path}: {e}")
        return None, str(e)

def main():
    images = get_images(IMAGE_DIR)
    if not images:
        print(f"No images found in {IMAGE_DIR}. Place some sample images there and restart.")
        return
    idx = 0
    while True:
        img = images[idx % len(images)]
        status, text = send_frame(img)
        idx += 1
        time.sleep(INTERVAL)

if __name__ == "__main__":
    print("Starting mock camera worker...")
    main()
