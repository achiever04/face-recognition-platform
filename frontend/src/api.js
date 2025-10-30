// frontend/src/api.js
import axios from "axios";
import { io } from "socket.io-client";

/* ===========================
   Environment / Configuration
   =========================== */
const VITE_BASE =
  typeof import.meta !== "undefined" && import.meta.env ? import.meta.env.VITE_API_BASE : undefined;
const REACT_BASE = process.env.REACT_APP_API_BASE;
const BASE_URL = VITE_BASE || REACT_BASE || "http://127.0.0.1:8000";

const TIMEOUT_MS = Number(
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_TIMEOUT_MS) ||
    process.env.REACT_APP_API_TIMEOUT_MS ||
    12000
);

const MAX_RETRIES = Number(
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_MAX_RETRIES) ||
    process.env.REACT_APP_API_MAX_RETRIES ||
    2
);

const RETRY_BASE_DELAY_MS = 300;

/* ===========================
   Axios instance & helpers
   =========================== */
const API = axios.create({
  baseURL: BASE_URL,
  timeout: TIMEOUT_MS,
  headers: {
    Accept: "application/json",
  },
});

API.interceptors.request.use(
  (config) => {
    try {
      const token = localStorage.getItem("auth_token");
      if (token) {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${token}`;
      }
    } catch (e) {}
    return config;
  },
  (error) => Promise.reject(error)
);

API.interceptors.response.use(
  (resp) => resp,
  async (error) => {
    const config = error.config || {};
    if (!config) return Promise.reject(error);
    config.retryAllowed = config.retryAllowed !== false;
    config.__retryCount = config.__retryCount || 0;

    const status = error.response ? error.response.status : null;
    const isNetworkError = !error.response;
    const shouldRetry =
      config.retryAllowed &&
      (isNetworkError || (status >= 500 && status < 600)) &&
      config.__retryCount < MAX_RETRIES;

    if (shouldRetry) {
      config.__retryCount += 1;
      const backoff = RETRY_BASE_DELAY_MS * Math.pow(2, config.__retryCount - 1);
      await new Promise((res) => setTimeout(res, backoff));
      return API(config);
    }

    return Promise.reject(error);
  }
);

/* ===========================
   Response normalizer
   =========================== */
async function safeRequest(promise) {
  try {
    const resp = await promise;
    // Return normalized object: ok, data, error, status
    return { ok: true, data: resp.data, error: null, status: resp.status };
  } catch (err) {
    let message = "Unknown error";
    let code = null;
    if (err.response) {
      code = err.response.status;
      try {
        const d = err.response.data;
        if (d && typeof d === "object" && (d.detail || d.message)) {
          message = d.detail || d.message;
        } else if (d && typeof d === "string") {
          message = d;
        } else {
          message = JSON.stringify(d);
        }
      } catch (e) {
        message = err.response.statusText || `HTTP ${err.response.status}`;
      }
    } else if (err.request) {
      message = "No response from server (network or timeout)";
    } else {
      message = err.message || String(err);
    }
    return { ok: false, data: null, error: { message, code } };
  }
}

/* ===========================
   CancelToken helper
   =========================== */
const CancelToken = axios.CancelToken;
function makeCancelable() {
  let cancel;
  const token = new CancelToken(function executor(c) {
    cancel = c;
  });
  return { token, cancel };
}

/* ===========================
   Socket.IO (single shared factory)
   =========================== */
let socket = null;
function createSocket({ path = "/", query = {}, authToken = null, reconnectionAttempts = 5 } = {}) {
  if (socket && socket.connected) return socket;

  if (!authToken) {
    try {
      authToken = localStorage.getItem("auth_token");
    } catch (e) {
      authToken = null;
    }
  }

  const opts = {
    path,
    transports: ["websocket", "polling"],
    reconnection: true,
    reconnectionAttempts,
    autoConnect: true,
    auth: authToken ? { token: authToken } : undefined,
    query,
  };

  socket = io(BASE_URL, opts);

  socket.on("connect", () => {
    console.info("[socket] connected", socket.id);
  });
  socket.on("disconnect", (reason) => {
    console.info("[socket] disconnected", reason);
  });
  socket.on("connect_error", (err) => {
    console.warn("[socket] connect_error", err?.message || err);
  });

  return socket;
}

const defaultSocket = createSocket();
export { defaultSocket as socket, createSocket };

/* ===========================
   Utility helpers
   =========================== */
function cacheBustingUrl(url) {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}_cb=${Date.now()}`;
}

/* ===========================
   Existing API exports (wrapped)
   =========================== */

/* Basic Status */
export const getStatus = () => safeRequest(API.get("/"));
export const getCameraStatus = () => safeRequest(API.get("/camera/status"));

/* Face Management */
export const uploadFace = (formData) =>
  safeRequest(API.post("/face/upload", formData, { headers: { "Content-Type": "multipart/form-data" } }));

export const compareFaces = (formData) =>
  safeRequest(API.post("/face/compare", formData, { headers: { "Content-Type": "multipart/form-data" } }));

export const listFaces = () => safeRequest(API.get("/face/list"));

export const deleteFace = (targetName) =>
  safeRequest(API.delete(`/face/delete/${encodeURIComponent(targetName)}`));

/* Camera Alerts & Tracking */
export const getCameraAlerts = () => safeRequest(API.get("/camera/alerts"));

export const getCameraFeed = (cameraId) => safeRequest(API.get(`/camera/${encodeURIComponent(cameraId)}/feed`));

export const getTrackingStats = () => safeRequest(API.get("/camera/stats"));

export const getPersonMovement = (personName, limit = 20) =>
  safeRequest(API.get(`/camera/movement/${encodeURIComponent(personName)}`, { params: { limit } }));

export const analyzePatterns = (personName) =>
  safeRequest(API.get(`/camera/analyze/${encodeURIComponent(personName)}`));

/* Federated Learning */
export const getFederatedStatus = (clientId) =>
  safeRequest(API.get("/face/fl/status", { params: { client_id: clientId } }));

export const getFLWeights = (target) => safeRequest(API.get("/face/fl/get_weights", { params: { target } }));

export const uploadFLWeights = (target, weights) =>
  safeRequest(
    API.post(
      "/face/fl/upload_weights",
      { target, weights },
      { headers: { "Content-Type": "application/json" } }
    )
  );

export const aggregateWeights = (clientIds = null, newVersion = 1) =>
  safeRequest(
    API.post(
      "/face/fl/aggregate",
      { client_ids: clientIds, new_version: newVersion },
      { headers: { "Content-Type": "application/json" } }
    )
  );

export const getAggregatedModel = () => safeRequest(API.get("/face/fl/aggregated_model"));

/* Deepfake Detection */
export const detectDeepfake = (formData) =>
  safeRequest(API.post("/deepfake/detect", formData, { headers: { "Content-Type": "multipart/form-data" } }));

export const detectDeepfakeImage = (formData) =>
  safeRequest(API.post("/deepfake/detect-image", formData, { headers: { "Content-Type": "multipart/form-data" } }));

export const detectDeepfakeCCTV = (cameras) =>
  safeRequest(API.get("/deepfake/cctv", { params: { cameras } }));

/* Alert Management */
export const getAlerts = (params = {}) => safeRequest(API.get("/alerts", { params }));
export const getLatestAlert = (target = null) => safeRequest(API.get("/alerts/latest", { params: { target } }));
export const getWatchlist = () => safeRequest(API.get("/alerts/watchlist"));
export const addToWatchlist = (targetName) =>
  safeRequest(API.post(`/alerts/watchlist/${encodeURIComponent(targetName)}`));
export const removeFromWatchlist = (targetName) =>
  safeRequest(API.delete(`/alerts/watchlist/${encodeURIComponent(targetName)}`));
export const getGeofences = () => safeRequest(API.get("/alerts/geofences"));
export const createGeofence = (zoneData) =>
  safeRequest(API.post("/alerts/geofences", zoneData, { headers: { "Content-Type": "application/json" } }));
export const getAlertStats = () => safeRequest(API.get("/alerts/stats"));
export const configureEmail = (configData) =>
  safeRequest(API.post("/alerts/config/email", configData, { headers: { "Content-Type": "application/json" } }));
export const configureSms = (configData) =>
  safeRequest(API.post("/alerts/config/sms", configData, { headers: { "Content-Type": "application/json" } }));

/* New helpers: uploads, snapshots, async tasks, polling */
export function uploadWithProgress(url, formData, onProgress) {
  return safeRequest(
    API.post(url, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (progressEvent) => {
        try {
          if (onProgress) onProgress(progressEvent.loaded, progressEvent.total || 0);
        } catch (e) {}
      },
    })
  );
}

export function fetchCameraSnapshot(cameraId, { enhance = false } = {}) {
  const url = `/api/camera/${encodeURIComponent(cameraId)}/snapshot${enhance ? "?enhance=1" : ""}`;
  const cbUrl = cacheBustingUrl(url);
  return safeRequest(API.get(cbUrl, { responseType: "blob" }));
}

export function enqueueAsyncFaceSearch(fileOrFormData, onProgress) {
  let formData;
  if (fileOrFormData instanceof FormData) {
    formData = fileOrFormData;
  } else {
    formData = new FormData();
    formData.append("file", fileOrFormData);
  }
  return uploadWithProgress("/async/face/search", formData, onProgress);
}

export async function pollJob(jobId, { interval = 800, timeout = 60000, onUpdate = null } = {}) {
  const start = Date.now();
  while (true) {
    const resp = await safeRequest(API.get(`/async/jobs/${encodeURIComponent(jobId)}`));
    if (onUpdate) {
      try {
        onUpdate(resp);
      } catch (e) {}
    }
    if (!resp.ok) {
      return resp;
    }
    const status = resp.data.status;
    if (status === "finished" || status === "failed") {
      return resp;
    }
    if (Date.now() - start > timeout) {
      return { ok: false, data: null, error: { message: "timeout waiting for job", code: "timeout" } };
    }
    await new Promise((r) => setTimeout(r, interval));
  }
}

/* Backwards-compatible default export */
export default API;
