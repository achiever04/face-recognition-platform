# README.md

# Intelligent Multi-Camera Face Recognition Platform

## Project Title

Intelligent Multi-Camera Face Recognition Platform for Public Safety, Missing Person, and Criminal Identification

## Objective

Develop a powerful software application capable of ingesting live streams from multiple public-facing cameras, performing real-time, privacy-preserving face recognition—even in challenging scenarios—and providing law enforcement with accurate detection, movement tracking, and instant alerts.

## Features

* **Multi-Camera Input:** Ingests streams from multiple IP or USB cameras.
* **Real-Time Face Recognition:** Utilizes `face_recognition` library for detection and matching.
* **Live Tracking:** Tracks individuals across different camera views.
* **Alert System:** Generates real-time alerts (console, file logs, optional Email/SMS) for watchlist matches or geo-fence breaches.
* **Web Dashboard:** React-based frontend for monitoring feeds, managing targets, and viewing alerts/logs.
* **WebSocket Updates:** Real-time UI updates via Socket.IO.
* **Database Persistence:** Stores face embeddings (encrypted), tracking history, alerts, and configuration in MongoDB.
* **Privacy Features:** Encrypted storage of face embeddings.
* **(Optional) Deepfake Detection:** Module to identify potential spoofing attempts (requires trained model).
* **(Optional) Federated Learning:** API endpoints for collecting and aggregating model updates from edge devices.

## Tech Stack

* **Backend:** FastAPI (Python), Socket.IO, OpenCV, face\_recognition, MongoDB, Pydantic, Uvicorn
* **Frontend:** React, Vite, Tailwind CSS, Axios, socket.io-client
* **Database:** MongoDB

## Project Structure

```
face-recognition-platform/
├── backend/
│   ├── app/
│   │   ├── models/       # Pydantic models for DB
│   │   ├── routes/       # API endpoint definitions
│   │   ├── services/     # Core logic (face, tracking, alerts)
│   │   ├── utils/        # Helpers (DB connection, logging)
│   │   ├── __init__.py
│   │   ├── main.py       # FastAPI entry point
│   │   └── state.py      # Shared in-memory state
│   ├── data/             # Persistent data (encryption key, FL weights)
│   ├── logs/             # Application log files
│   ├── models/           # AI model files (e.g., .pth, .pb)
│   ├── temp_uploads/     # Temporary storage for uploads
│   ├── .env              # Environment variables (!!! ADD TO .gitignore !!!)
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── public/           # Static assets
│   ├── src/              # React source code
│   │   ├── components/   # Reusable UI components
│   │   ├── App.jsx       # Main dashboard component
│   │   ├── api.js        # API connection functions
│   │   ├── index.css     # Global styles (Tailwind setup)
│   │   └── main.jsx      # React entry point
│   ├── .eslintrc.cjs     # ESLint config
│   ├── index.html        # Main HTML file
│   ├── package.json      # Node dependencies & scripts
│   ├── postcss.config.js # PostCSS config (for Tailwind)
│   ├── tailwind.config.js# Tailwind config
│   └── vite.config.js    # Vite config
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Setup & Installation

**(Instructions need to be added here)**

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd face-recognition-platform
    ```
2.  **Backend Setup:**
    * Create a virtual environment: `python -m venv venv`
    * Activate it: `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows)
    * Install dependencies: `pip install -r backend/requirements.txt`
    * Set up MongoDB (e.g., run locally or using Docker).
    * Create a `.env` file in `backend/` based on `.env.example` (You should create this file).
    * Run the backend: `cd backend` then `uvicorn app.main:app --reload`
3.  **Frontend Setup:**
    * Navigate to frontend: `cd ../frontend`
    * Install dependencies: `npm install`
    * Run the frontend: `npm run dev`

## Usage

**(Instructions need to be added here)**

* Access the frontend dashboard at `http://localhost:5173` (or the port Vite assigns).
* Upload target faces using the "Upload Target Photo" section.
* Monitor camera feeds and alerts.

## Future Enhancements

* Implement actual AI models for Video Enhancement and Anti-Spoofing.
* Integrate a real SMS provider (e.g., Twilio).
* Add user authentication and roles.
* Deploy using Docker/Kubernetes.
* Improve UI/UX, add configuration options via UI.