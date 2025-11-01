# 🧠 Intelligent Multi-Camera Face Recognition Platform  
### _Real-Time, Privacy-Preserving AI System for Public Safety, Missing Person Tracking, and Criminal Detection_

---

## 📌 Overview

The **Intelligent Multi-Camera Face Recognition Platform** is a full-stack AI-powered surveillance and analytics solution built to assist law enforcement and smart-city applications.

It integrates **real-time face recognition**, **multi-camera tracking**, **deepfake detection**, **federated learning**, and **secure alerting** — all accessible through an intuitive **React dashboard**.

Designed for **scalability**, **privacy**, and **accuracy**, this platform enables secure identification, watchlist-based alerts, and movement pattern analysis across multiple public-facing camera feeds.

---

## 🎯 Objectives

- Enable **real-time, multi-camera person recognition** and tracking.
- Provide **privacy-preserving, encrypted embedding storage**.
- Detect **deepfakes and spoofing attacks** in live or uploaded videos.
- Support **Federated Learning** to aggregate model updates from distributed edge devices.
- Deliver **instant alerts** for watchlist or geofence breaches via WebSockets and optional SMS/Email.
- Offer a **web dashboard** for centralized management and analytics.

---

## 🚀 Key Features

### 🔹 Core Intelligence
- **Multi-Camera Input:** Supports simultaneous IP/USB camera streams.
- **Real-Time Face Recognition:** Uses `face_recognition` and InsightFace (RetinaFace) for fast, accurate identification.
- **Live Tracking:** Maintains continuous identity tracking across multiple feeds.
- **Alert System:** Emits instant alerts (via WebSocket, file logs, email, SMS).
- **Movement & History Logs:** Stores detections, timestamps, and geolocation history.
- **Encrypted Embeddings:** Protects face embeddings with AES encryption (Fernet).

### 🔹 Advanced Modules
- **Deepfake Detection:** Optional ONNX-based classifier for spoofing detection.
- **Federated Learning (FL):** Aggregates client model updates using FedAvg; privacy-preserving collaborative training.
- **Async Task Manager:** Background async jobs for long-running AI tasks (model training, enhancement, etc.).
- **Model Manager:** Auto-unloads idle models to save memory and reinitializes on demand.

### 🔹 Frontend (React + Vite)
- Real-time dashboard for:
  - Uploading and managing target faces.
  - Monitoring camera feeds and detection logs.
  - Viewing watchlists, alerts, and FL aggregation status.
- Built with **Tailwind CSS** and **Socket.IO client** for live updates.

### 🔹 Backend (FastAPI + Socket.IO)
- RESTful APIs for all detection, tracking, and management operations.
- Real-time event broadcasting via WebSockets.
- Modular architecture with clear separation of routes, services, and utilities.

### 🔹 Data & Security
- **MongoDB** persistence for embeddings, alerts, tracking logs, and config.
- **Encrypted keys** stored in `data/.encryption_key`.
- **Environment-based configuration** (`.env` for backend and frontend).
- **JWT-ready authentication layer** (placeholder for future user roles).

---

## 🧩 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Backend** | FastAPI, Socket.IO (ASGI), Uvicorn, MongoDB, OpenCV, face_recognition, InsightFace, PyTorch, Pydantic |
| **Frontend** | React, Vite, Tailwind CSS, Axios, socket.io-client |
| **Database** | MongoDB (local or cloud) |
| **ML / AI** | face_recognition, InsightFace, scikit-learn, Torch, ONNXRuntime |
| **Other** | Redis (optional async tasks), Python 3.10+, Node.js 18+ |

---

## 🗂️ Folder Structure

face-recognition-platform/
├── backend/
│ ├── app/
│ │ ├── routes/ # API endpoints (camera, face, alerts, deepfake, federated, snapshot)
│ │ ├── services/ # Core AI logic (face_service, tracking_service, alert_service)
│ │ ├── utils/ # DB connection, encryption, async helpers, logging
│ │ ├── models/ # Pydantic schemas & database models
│ │ ├── state.py # Global runtime state (cameras, encodings, FL models)
│ │ ├── main.py # FastAPI + Socket.IO entry point
│ │ └── init.py
│ ├── data/ # Persistent files (encryption keys, FL weights, models)
│ ├── logs/ # Application logs and JSON event files
│ ├── .env # Environment configuration (⚠ do not commit)
│ ├── requirements.txt # Python dependencies
│ └── Dockerfile # Backend Docker build file
│
├── frontend/
│ ├── public/ # Static assets
│ ├── src/
│ │ ├── components/ # Reusable UI components
│ │ ├── App.jsx # Main application dashboard
│ │ ├── api.js # Axios + Socket.IO API wrapper
│ │ ├── index.css # Tailwind base styles
│ │ ├── main.jsx # React entry point
│ │ └── ...
│ ├── index.html # Base HTML file
│ ├── package.json # Node dependencies & scripts
│ ├── vite.config.js # Vite configuration
│ ├── tailwind.config.js # Tailwind configuration
│ ├── postcss.config.js # PostCSS setup
│ ├── eslint.config.js # Linting setup
│ └── .env # Frontend environment variables
│
├── .gitignore
└── README.md

🐍 2. Backend Setup (FastAPI)
cd backend
python3.10 -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt

Configure Environment

Create a .env file inside backend/:

APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=true
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=face_recognition_db

UPLOAD_DIR=data/uploads
ENCRYPTION_KEY_PATH=data/.encryption_key

ENABLE_DEEPFAKE_DETECTION=true
ENABLE_FEDERATED=true
LOG_LEVEL=INFO

Start MongoDB
If using local MongoDB:
sudo service mongod start
or via Docker:

docker run -d -p 27017:27017 --name mongo mongo:latest

Run Backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open: http://127.0.0.1:8000/docs
 for interactive API documentation.

💻 3. Frontend Setup (React + Vite)
cd ../frontend
npm install

Create .env inside frontend/:

VITE_API_BASE=http://127.0.0.1:8000
VITE_SOCKET_BASE=http://127.0.0.1:8000
VITE_API_TIMEOUT_MS=12000


Run the frontend:

npm run dev


Visit the web dashboard at:
➡️ http://localhost:5173

🧠 Usage Guide

Open the dashboard at http://localhost:5173.

Upload a face image under the “Upload Target” section.

Start live feeds – active cameras begin recognition automatically.

Monitor alerts for matches or geo-fence breaches in real time.

Optional:

Upload local FL weights (for federated training).

Run “Aggregate Weights” to simulate FedAvg.

View Deepfake detections or snapshots.

🪄 API Reference

FastAPI provides an automatic Swagger UI at:

http://127.0.0.1:8000/docs


Key Routes:

Endpoint	Method	Description
/face/upload	POST	Upload & encode a new face
/face/list	GET	List all stored faces
/camera/status	GET	Get camera health info
/alerts/latest	GET	Fetch most recent alerts
/face/fl/aggregate	POST	Trigger federated model aggregation
/deepfake/detect-image	POST	Analyze image for spoofing
🧰 Developer Notes
Python Environment

Recommended Python: 3.10

Uses opencv-python-headless to prevent GUI conflicts on headless servers.

Compatible with FastAPI 0.116.1 and Starlette 0.45.0.

Frontend Development

Built with Vite for fast HMR and modern bundling.

Uses Socket.IO v4 client for real-time communication.

Environment variables prefixed with VITE_ for runtime injection.

MongoDB Notes

If you see IndexOptionsConflict warnings, they are safe (indexes already exist).

To reset, run:

use face_recognition_db
db.logs.dropIndexes()

🧩 Troubleshooting
Issue	Possible Fix
WebSocket 403	Ensure FRONTEND_ORIGINS in backend .env includes both localhost and 127.0.0.1
process is not defined (frontend)	Fixed in api.js via safe import guard
Cameras not detected	Check /dev/video* or update IDs in app/state.py
Aggregation failed	Ensure at least one local weight file exists before calling aggregate
Face not recognized	Restart backend after uploads or ensure ENCODINGS are reloaded
📦 Deployment
Docker (optional)
docker build -t face-recognition-backend ./backend
docker run -d -p 8000:8000 face-recognition-backend

Production Suggestions

Use Gunicorn + UvicornWorkers for FastAPI.

Serve frontend build via Nginx.

Secure backend with HTTPS and JWT authentication.

Use MongoDB Atlas for cloud storage.

🔮 Future Enhancements

 Add full JWT-based authentication and user roles.

 Replace placeholder Deepfake model with production-trained CNN/Transformer.

 GPU acceleration for camera pipelines.

 Advanced analytics dashboard (heatmaps, movement patterns).

 Cloud deployment (Docker Compose, Kubernetes, CI/CD).
