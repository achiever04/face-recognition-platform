import axios from "axios";
// --- NEW: Import Socket.IO client ---
import { io } from "socket.io-client";

const BASE_URL = "http://127.0.0.1:8000";

const API = axios.create({
  baseURL: BASE_URL,
});

// --- NEW: Socket.IO Client Initialization ---
// Create and export a single socket instance to be shared by the app
export const socket = io(BASE_URL);

// ------------------------
// Basic Status
// ------------------------
export const getStatus = () => API.get("/");

export const getCameraStatus = () => API.get("/camera/status");

// ------------------------
// Face Management
// ------------------------
export const uploadFace = (formData) =>
  API.post("/face/upload", formData, {
    // Note: 'Content-Type' is set automatically by browser for FormData
  });

export const compareFaces = (formData) =>
  API.post("/face/compare", formData, {
    // Note: 'Content-Type' is set automatically by browser for FormData
  });

export const listFaces = () => API.get("/face/list");

export const deleteFace = (targetName) =>
  API.delete(`/face/delete/${targetName}`);

// ------------------------
// Camera Alerts & Tracking
// ------------------------
export const getCameraAlerts = () => API.get("/camera/alerts");

export const getCameraFeed = (cameraId) => API.get(`/camera/${cameraId}/feed`);

export const getTrackingStats = () => API.get("/camera/stats");

export const getPersonMovement = (personName, limit = 20) =>
  API.get(`/camera/movement/${personName}`, { params: { limit } });

export const analyzePatterns = (personName) =>
  API.get(`/camera/analyze/${personName}`);

// ------------------------
// Federated Learning
// ------------------------
export const getFederatedStatus = (clientId) =>
  API.get("/face/fl/status", { params: { client_id: clientId } });

export const getFLWeights = (target) =>
  API.get("/face/fl/get_weights", { params: { target } });

export const uploadFLWeights = (target, weights) =>
  API.post(
    "/face/fl/upload_weights",
    { target, weights },
    { headers: { "Content-Type": "application/json" } }
  );

// --- NEW: FL Aggregation functions ---
export const aggregateWeights = (clientIds = null, newVersion = 1) =>
  API.post(
    "/face/fl/aggregate",
    { client_ids: clientIds, new_version: newVersion },
    { headers: { "Content-Type": "application/json" } }
  );

export const getAggregatedModel = () => API.get("/face/fl/aggregated_model");

// ------------------------
// Deepfake Detection
// ------------------------
export const detectDeepfake = (formData) =>
  API.post("/deepfake/detect", formData, {
    // Note: 'Content-Type' is set automatically by browser for FormData
  });

// --- NEW: Deepfake Image Detection ---
export const detectDeepfakeImage = (formData) =>
  API.post("/deepfake/detect-image", formData, {
    // Note: 'Content-Type' is set automatically by browser for FormData
  });

export const detectDeepfakeCCTV = (cameras) =>
  API.get("/deepfake/cctv", { params: { cameras } });

// ------------------------
// Alert Management (NEW functions added)
// ------------------------
export const getAlerts = (params = {}) => API.get("/alerts", { params });

export const getLatestAlert = (target = null) =>
  API.get("/alerts/latest", { params: { target } });

// --- NEW: Watchlist functions ---
export const getWatchlist = () => API.get("/alerts/watchlist");

export const addToWatchlist = (targetName) =>
  API.post(`/alerts/watchlist/${targetName}`);

export const removeFromWatchlist = (targetName) =>
  API.delete(`/alerts/watchlist/${targetName}`);

// --- NEW: Geofence functions ---
export const getGeofences = () => API.get("/alerts/geofences");

export const createGeofence = (zoneData) =>
  API.post("/alerts/geofences", zoneData, {
    headers: { "Content-Type": "application/json" },
  });

// --- NEW: Config functions ---
export const getAlertStats = () => API.get("/alerts/stats");

export const configureEmail = (configData) =>
  API.post("/alerts/config/email", configData, {
    headers: { "Content-Type": "application/json" },
  });

export const configureSms = (configData) =>
  API.post("/alerts/config/sms", configData, {
    headers: { "Content-Type": "application/json" },
  });

// Default export
export default API;