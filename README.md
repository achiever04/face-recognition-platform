# 🧠 Intelligent Multi-Camera Face Recognition Platform  
### _Real-Time, Privacy-Preserving AI System for Public Safety, Missing Person Tracking, and Criminal Detection_

---

## 📌 Overview

The **Intelligent Multi-Camera Face Recognition Platform** is a full-stack AI-powered surveillance and analytics system built to assist law enforcement and smart-city applications.

It integrates **real-time face recognition**, **multi-camera tracking**, **deepfake detection**, **federated learning**, and **secure alerting** — all accessible through an intuitive **React dashboard**.

Designed for **scalability**, **privacy**, and **accuracy**, this platform enables secure identification, watchlist-based alerts, and movement pattern analysis across multiple public-facing camera feeds.

---

## 🎯 Objectives

- Enable **real-time, multi-camera person recognition** and tracking.  
- Provide **privacy-preserving, encrypted embedding storage**.  
- Detect **deepfakes and spoofing attacks** in live or uploaded videos.  
- Support **Federated Learning (FL)** for distributed, privacy-preserving model training.  
- Deliver **instant alerts** via WebSockets, SMS, or Email.  
- Offer a **web dashboard** for centralized management and analytics.  

---

## 🚀 Key Features

### 🔹 Core Intelligence
- **Multi-Camera Input:** Supports simultaneous IP/USB camera streams.  
- **Real-Time Face Recognition:** Uses `face_recognition` and **InsightFace (RetinaFace)** for high accuracy.  
- **Live Tracking:** Maintains continuous identity tracking across multiple feeds.  
- **Alert System:** Emits instant alerts via WebSocket, email, SMS, or logs.  
- **Movement & History Logs:** Tracks detections, timestamps, and geolocation.  
- **Encrypted Embeddings:** Protects embeddings with AES (Fernet) encryption.  

### 🔹 Advanced Modules
- **Deepfake Detection:** ONNX-based classifier for spoofing detection.  
- **Federated Learning:** Uses **FedAvg** aggregation for decentralized training.  
- **Async Task Manager:** Handles background tasks (model training, updates).  
- **Model Manager:** Auto-unloads idle models to optimize resource use.  

### 🔹 Frontend (React + Vite)
- Real-time dashboard to:  
  - Upload and manage target faces.  
  - Monitor camera feeds and detection logs.  
  - View watchlists, alerts, and FL aggregation status.  
- Built with **Tailwind CSS** and **Socket.IO client** for live updates.  

### 🔹 Backend (FastAPI + Socket.IO)
- RESTful APIs for detection, tracking, and management.  
- Real-time event broadcasting via WebSockets.  
- Modular, service-oriented architecture.  

### 🔹 Data & Security
- **MongoDB** for persistence of embeddings, alerts, and logs.  
- **Encrypted keys** stored in `data/.encryption_key`.  
- **Environment-based configuration** via `.env`.  
- **JWT-ready authentication layer** (future feature).  

---

## 🧩 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Backend** | FastAPI, Socket.IO (ASGI), Uvicorn, MongoDB, OpenCV, face_recognition, InsightFace, PyTorch, Pydantic |
| **Frontend** | React, Vite, Tailwind CSS, Axios, socket.io-client |
| **Database** | MongoDB (local or cloud) |
| **ML / AI** | face_recognition, InsightFace, scikit-learn, Torch, ONNXRuntime |
| **Other** | Redis (optional for async tasks), Python 3.10+, Node.js 18+ |

---

## 🗂️ Folder Structure

```bash
face-recognition-platform/
├── backend/
│   ├── app/
│   │   ├── routes/          # API endpoints (camera, face, alerts, deepfake, federated, snapshot)
│   │   ├── services/        # Core AI logic (face_service, tracking_service, alert_service)
│   │   ├── utils/           # DB connection, encryption, async helpers, logging
│   │   ├── models/          # Pydantic schemas & database models
│   │   ├── state.py         # Global runtime state (cameras, encodings, FL models)
│   │   ├── main.py          # FastAPI + Socket.IO entry point
│   │   └── __init__.py
│   ├── data/                # Persistent files (keys, FL weights, models)
│   ├── logs/                # Application logs and events
│   ├── .env                 # Backend configuration (⚠️ Do not commit)
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Backend Docker build file
│
├── frontend/
│   ├── public/              # Static assets
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── App.jsx          # Main dashboard component
│   │   ├── api.js           # Axios + Socket.IO wrapper
│   │   ├── index.css        # Tailwind base styles
│   │   ├── main.jsx         # React entry point
│   │   └── ...
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── eslint.config.js
│   └── .env                 # Frontend environment variables
│
├── .gitignore
└── README.md
