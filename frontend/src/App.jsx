// frontend/src/App.jsx
import React, { useState, useEffect, useRef } from "react";
import {
  Camera,
  UploadCloud,
  Bell,
  GitBranch,
  Radar,
  Server,
  RefreshCw,
  Download,
} from "lucide-react";

import {
  socket,
  getCameraStatus,
  uploadFace,
  getCameraAlerts,
  uploadFLWeights as apiUploadFLWeights,
  getFLWeights as apiGetFLWeights,
  aggregateWeights,
  getAggregatedModel,
} from "./api";

function App() {
  // File upload state
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");

  // Camera / alerts / tracking
  const [cameraStatus, setCameraStatus] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [history, setHistory] = useState({});
  const [movementLog, setMovementLog] = useState([]);
  const [randomCameras, setRandomCameras] = useState([0, 0, 0, 0]);
  const [trackingCamera, setTrackingCamera] = useState(null);
  const [trackingTarget, setTrackingTarget] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [brokenFeeds, setBrokenFeeds] = useState(new Set());
  const [socketConnected, setSocketConnected] = useState(false);
  const feedRefreshTickerRef = useRef(0); // used to cache-bust image URLs

  // FL (Federated Learning) state
  const [flClientId, setFlClientId] = useState("client_1");
  const [flWeights, setFlWeights] = useState({});
  const [aggregating, setAggregating] = useState(false);
  const [aggregatedModel, setAggregatedModel] = useState(null);

  // Backend base (single source of truth, can be overridden by env var)
  const BACKEND_BASE =
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_BACKEND_URL) ||
  (typeof process !== "undefined" && process.env ? process.env.REACT_APP_BACKEND_URL : undefined) ||
  "http://127.0.0.1:8000";


  // ------------------------
  // Poll / refresh camera status
  // ------------------------
  useEffect(() => {
    let mounted = true;

    const fetchCameraStatus = async () => {
      try {
        const res = await getCameraStatus();
        if (!mounted) return;
        // API may return different structures; handle safely
        const statusObj = res?.data?.status ?? res?.data ?? {};
        setCameraStatus(statusObj);
      } catch (err) {
        console.error("Camera status error:", err);
        setCameraStatus({ error: "❌ Camera API not reachable" });
      }
    };

    fetchCameraStatus();
    const interval = setInterval(fetchCameraStatus, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // ------------------------
  // Upload & Encode Target Photo
  // ------------------------
  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus("Uploading...");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_name", file.name);
    formData.append("save_raw", false);

    try {
      const res = await uploadFace(formData);
      const data = res?.data ?? {};
      if (data.status === "success") {
        setUploadStatus(`✅ ${data.message || `Uploaded ${data.filename}`}`);
        setFile(null);
        const el = document.getElementById("upload-input");
        if (el) el.value = null;
      } else {
        setUploadStatus(`❌ ${data.message || "Upload failed"}`);
      }
    } catch (err) {
      console.error("Upload error:", err);
      const message =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        "Error uploading file";
      setUploadStatus(`❌ ${message}`);
    }
  };

  // ------------------------
  // WebSocket + initial data fetch
  // ------------------------
  useEffect(() => {
    let mounted = true;

    const fetchInitialData = async () => {
      try {
        const res = await getCameraAlerts();
        if (!mounted) return;
        const data = res?.data ?? {};
        if (data.status === "success") {
          setAlerts(data.alerts ?? []);
          setHistory(data.history ?? {});
          setMovementLog(data.movement_log ?? []);
          if (data.latest_detection) {
            setTrackingCamera(data.latest_detection.camera_id);
            setTrackingTarget(data.latest_detection.target);
          }
        } else {
          // Accept non-success but still try to fill partial fields
          setAlerts(data.alerts ?? []);
          setHistory(data.history ?? {});
          setMovementLog(data.movement_log ?? {});
        }
      } catch (err) {
        console.error("Error fetching initial alerts:", err);
      }
    };

    const onConnect = () => {
      console.log("Socket connected:", socket.id);
      setSocketConnected(true);
      // Refresh data on reconnect
      fetchInitialData();
      // Also refresh camera status
      getCameraStatus()
        .then((r) => {
          const statusObj = r?.data?.status ?? r?.data ?? {};
          setCameraStatus(statusObj);
        })
        .catch((e) => console.warn("Failed to refresh camera status on reconnect", e));
    };

    const onDisconnect = (reason) => {
      console.log("Socket disconnected:", reason);
      setSocketConnected(false);
    };

    const onNewAlert = (payload) => {
      const alert = payload?.alert ?? payload;
      if (!alert) return;
      setAlerts((prev) => [alert, ...prev].slice(0, 50));
      setHistory((prevHistory) => ({
        ...prevHistory,
        [alert.target]: [alert, ...(prevHistory[alert.target] || [])].slice(0, 10),
      }));
    };

    const onUpdateMovementLog = (payload) => {
      const log = payload?.log ?? payload;
      if (!log) return;
      setMovementLog((prev) => [log, ...prev].slice(0, 100));
    };

    const onUpdateTrackingFeed = (payload) => {
      const detection = payload?.detection ?? payload;
      if (!detection) return;
      setTrackingCamera(detection.camera_id);
      setTrackingTarget(detection.person || detection.target);
    };

    // Attach listeners
    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("new_alert", onNewAlert);
    socket.on("update_movement_log", onUpdateMovementLog);
    socket.on("update_tracking_feed", onUpdateTrackingFeed);

    // set initial connection state & fetch initial data once
    setSocketConnected(!!socket.connected);
    fetchInitialData();

    return () => {
      mounted = false;
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("new_alert", onNewAlert);
      socket.off("update_movement_log", onUpdateMovementLog);
      socket.off("update_tracking_feed", onUpdateTrackingFeed);
    };
  }, []);

  // ------------------------
  // Random 4 camera feeds selector (recomputed when cameraStatus changes)
  // ------------------------
  useEffect(() => {
    const updateRandomCameras = () => {
      const available = Object.keys(cameraStatus || {})
        .filter((id) => {
          const entry = cameraStatus[id];
          // Support both shapes: entry.state or entry.status
          const state = entry?.state ?? entry?.status ?? null;
          return state === "ok" || state === "online" || state === "available";
        })
        .map((id) => parseInt(id))
        .filter((n) => !Number.isNaN(n));

      if (available.length === 0) {
        // Keep previously selected or early fallback to 0
        return;
      }

      const shuffled = [...available].sort(() => 0.5 - Math.random());
      const selected = shuffled.slice(0, 4);

      while (selected.length < 4 && available.length > 0) {
        selected.push(available[0]);
      }

      setRandomCameras(selected);
    };

    updateRandomCameras();
  }, [cameraStatus]);

  // ------------------------
  // Feed cache-busting ticker (updates every 5s to refresh MJPEG <img> src param)
  // ------------------------
  useEffect(() => {
    const t = setInterval(() => {
      feedRefreshTickerRef.current = Date.now();
      // trigger a re-render (small trick: update a state)
      // but avoid extra state, we can call setBrokenFeeds to itself to trigger render if needed:
      setBrokenFeeds((s) => new Set(s)); // safe no-op state update to refresh img srcs
    }, 5000);
    return () => clearInterval(t);
  }, []);

  // ------------------------
  // Badge helper
  // ------------------------
  const getBadge = (distance) => {
    if (distance === undefined || distance === null) return null;
    if (distance < 0.4)
      return (
        <span className="ml-2 px-2.5 py-0.5 bg-red-500/20 text-red-400 rounded-full text-xs font-semibold">
          High
        </span>
      );
    if (distance < 0.6)
      return (
        <span className="ml-2 px-2.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded-full text-xs font-semibold">
          Medium
        </span>
      );
    return (
      <span className="ml-2 px-2.5 py-0.5 bg-gray-500/20 text-gray-400 rounded-full text-xs font-semibold">
        Low
      </span>
    );
  };

  // ------------------------
  // Federated Learning helpers
  // ------------------------
  const uploadFLWeights = async () => {
    try {
      const target = flClientId;
      const weights = {
        layer1: [Math.random(), Math.random()],
        layer2: [Math.random(), Math.random()],
      };
      const res = await apiUploadFLWeights(target, weights);
      const data = res?.data ?? {};
      if (data.status === "success") {
        alert("✅ Federated weights uploaded successfully!");
        await fetchFLWeights();
      } else {
        alert(`❌ Failed to upload FL weights: ${data.message || "unknown"}`);
      }
    } catch (err) {
      console.error("Error uploading FL weights:", err);
      const message = err?.response?.data?.detail || err?.message || "Upload failed";
      alert(`❌ Failed to upload FL weights: ${message}`);
    }
  };

  const fetchFLWeights = async () => {
    try {
      const res = await apiGetFLWeights(flClientId);
      const data = res?.data ?? {};
      if (data.status === "success") {
        setFlWeights(data.weights ?? {});
      } else {
        setFlWeights({});
        console.warn(`Could not fetch weights for ${flClientId}: ${data.message}`);
      }
    } catch (err) {
      console.error("Error fetching FL weights:", err);
      setFlWeights({});
    }
  };

  const handleAggregate = async () => {
    setAggregating(true);
    setAggregatedModel(null);
    try {
      const res = await aggregateWeights(null, Date.now());
      const data = res?.data ?? {};
      if (data.status === "success") {
        alert(`✅ Aggregation successful! New model version: ${data.new_model_version}`);
        await handleGetAggregatedModel();
      } else {
        alert(`❌ Aggregation failed: ${data.message || "unknown"}`);
      }
    } catch (err) {
      console.error("Error aggregating weights:", err);
      const message = err?.response?.data?.detail || err?.message || "Aggregation failed";
      alert(`❌ Aggregation failed: ${message}`);
    } finally {
      setAggregating(false);
    }
  };

  const handleGetAggregatedModel = async () => {
    try {
      const res = await getAggregatedModel();
      const data = res?.data ?? {};
      if (data.status === "success") {
        setAggregatedModel(data);
      } else {
        alert(`❌ Failed to get aggregated model: ${data.message || "unknown"}`);
        setAggregatedModel(null);
      }
    } catch (err) {
      console.error("Error getting aggregated model:", err);
      const message = err?.response?.data?.detail || err?.message || "Failed to get aggregated model";
      alert(`❌ Failed to get aggregated model: ${message}`);
      setAggregatedModel(null);
    }
  };

  // ------------------------
  // Camera feed error handler (fix Set update usage)
  // ------------------------
  const handleFeedError = (camId) => {
    setBrokenFeeds((prev) => {
      const copy = new Set(prev);
      if (!copy.has(camId)) {
        console.warn(`Camera feed error for Cam ID: ${camId}`);
        copy.add(camId);
      }
      return copy;
    });
  };

  // Periodically attempt simple "reconnect" by clearing brokenFeeds so <img> tags reload
  useEffect(() => {
    const retryInterval = setInterval(() => {
      setBrokenFeeds((prev) => {
        if (prev.size === 0) return prev;
        return new Set(); // clear and cause re-render / reload of feed images
      });
    }, 15000);
    return () => clearInterval(retryInterval);
  }, []);

  // Helper to build feed URL (centralized)
  const buildFeedUrl = (cameraId) => {
    const timestamp = feedRefreshTickerRef.current || Date.now();
    return `${BACKEND_BASE}/camera/${cameraId}/feed?_=${timestamp}`;
  };

  // ------------------------
  // Render
  // ------------------------
  return (
    <div className="bg-gray-900 text-gray-200 min-h-screen font-sans">
      <div className="max-w-screen-xl mx-auto p-4 sm:p-6 lg:p-8">
        {/* Header */}
        <header className="border-b border-gray-700 pb-6 mb-8">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-teal-500">
            Multi-Camera Face Recognition Platform
          </h1>
          <p className="text-gray-400 mt-2 flex items-center text-sm">
            <Server className="w-4 h-4 mr-2 text-cyan-500" />
            Backend: {BACKEND_BASE} | WebSocket:{" "}
            <span
              className={`ml-1 font-semibold ${
                socketConnected ? "text-green-400" : "text-red-400"
              }`}
            >
              {socketConnected ? "Connected" : "Disconnected"}
            </span>
          </p>
        </header>

        {/* Main Content Grid */}
        <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-8">
            {/* Tracking Feed */}
            <section className="bg-gray-800 shadow-xl rounded-2xl p-6">
              <h2 className="text-2xl font-semibold mb-4 flex items-center">
                <Radar className="w-6 h-6 mr-3 text-cyan-400" />
                Live Tracking Feed
              </h2>
              {trackingCamera !== null && trackingTarget !== null ? (
                <div className="bg-black/20 rounded-lg overflow-hidden">
                  <div className="p-3 bg-gray-700/50 border-b border-gray-600">
                    <p className="text-sm">
                      Tracking{" "}
                      <b className="text-cyan-400">{trackingTarget}</b> at{" "}
                      <b className="text-cyan-400">
                        {cameraStatus?.[trackingCamera]?.name || `Camera ${trackingCamera}`}
                      </b>
                    </p>
                  </div>
                  {!brokenFeeds.has(trackingCamera) ? (
                    <img
                      src={buildFeedUrl(trackingCamera)}
                      alt={`Tracking Camera ${trackingCamera}`}
                      className="w-full h-auto object-cover"
                      onError={() => handleFeedError(trackingCamera)}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-64 bg-gray-700/30">
                      <p className="text-gray-500">Camera feed unavailable</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-64 bg-gray-700/30 rounded-lg">
                  <p className="text-gray-400">Awaiting target detection...</p>
                </div>
              )}
            </section>

            {/* Random Feeds */}
            <section className="bg-gray-800 shadow-xl rounded-2xl p-6">
              <h2 className="text-2xl font-semibold mb-4 flex items-center">
                <Camera className="w-6 h-6 mr-3 text-cyan-400" />
                Camera Grid
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {randomCameras.map((camId, idx) => (
                  <div
                    key={`${camId}-${idx}`}
                    className="bg-black/20 border-2 border-gray-700 rounded-lg overflow-hidden shadow-lg"
                  >
                    <p className="font-medium text-sm p-2 bg-gray-700/50">
                      {cameraStatus?.[camId]?.name || `Camera ${camId}`}
                    </p>
                    {!brokenFeeds.has(camId) ? (
                      <img
                        src={buildFeedUrl(camId)}
                        alt={`Camera ${camId}`}
                        className="w-full h-auto"
                        onError={() => handleFeedError(camId)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-32 bg-gray-700/30">
                        <p className="text-xs text-gray-500">Feed offline</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* Right Column */}
          <div className="lg:col-span-1 space-y-8">
            {/* Upload */}
            <section className="bg-gray-800 shadow-xl rounded-2xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <UploadCloud className="w-5 h-5 mr-3 text-cyan-400" />
                Upload Target Photo
              </h2>
              <div className="space-y-4">
                <input
                  id="upload-input"
                  type="file"
                  accept="image/*"
                  className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-cyan-500/10 file:text-cyan-300 hover:file:bg-cyan-500/20 cursor-pointer"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <button
                  className="w-full px-4 py-2 bg-cyan-600 text-white font-bold rounded-lg hover:bg-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed transition duration-300 ease-in-out"
                  onClick={handleUpload}
                  disabled={!file}
                >
                  Upload & Encode
                </button>
                {uploadStatus && (
                  <p className="mt-2 text-sm text-center text-gray-400">
                    {uploadStatus}
                  </p>
                )}
              </div>
            </section>

            {/* FL Dashboard */}
            <section className="bg-gray-800 shadow-xl rounded-2xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <GitBranch className="w-5 h-5 mr-3 text-cyan-400" />
                Federated Learning
              </h2>

              <input
                type="text"
                value={flClientId}
                onChange={(e) => setFlClientId(e.target.value)}
                placeholder="Enter Client ID"
                className="w-full px-3 py-2 mb-3 bg-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />

              <div className="space-y-3">
                <button
                  className="w-full px-4 py-2 bg-teal-600 text-white font-bold rounded-lg hover:bg-teal-700 transition duration-300 disabled:opacity-50"
                  onClick={uploadFLWeights}
                  disabled={!flClientId.trim()}
                >
                  Upload Local Weights (Demo)
                </button>

                <button
                  className="w-full px-4 py-2 bg-purple-600 text-white font-bold rounded-lg hover:bg-purple-700 transition duration-300 disabled:opacity-50 flex items-center justify-center"
                  onClick={handleAggregate}
                  disabled={aggregating}
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${aggregating ? "animate-spin" : ""}`} />
                  {aggregating ? "Aggregating..." : "Aggregate Weights"}
                </button>

                <button
                  className="w-full px-4 py-2 bg-gray-600 text-white font-bold rounded-lg hover:bg-gray-700 transition duration-300 flex items-center justify-center"
                  onClick={handleGetAggregatedModel}
                >
                  <Download className="w-4 h-4 mr-2" />
                  View Aggregated Model
                </button>

                <div className="mt-4">
                  <h3 className="text-sm text-gray-400 font-semibold mb-2">
                    Weights for '{flClientId}':
                    <button onClick={fetchFLWeights} className="ml-2 text-xs text-cyan-400 hover:underline">
                      (Refresh)
                    </button>
                  </h3>
                  <pre className="bg-gray-700 p-3 rounded-lg text-xs max-h-40 overflow-y-auto">
                    {JSON.stringify(flWeights, null, 2)}
                  </pre>

                  {aggregatedModel && (
                    <>
                      <h3 className="text-sm text-gray-400 font-semibold mt-4 mb-2">
                        Latest Aggregated Model (v{aggregatedModel.model_version}):
                      </h3>
                      <pre className="bg-gray-700 p-3 rounded-lg text-xs max-h-40 overflow-y-auto">
                        {JSON.stringify(aggregatedModel, null, 2)}
                      </pre>
                    </>
                  )}
                </div>
              </div>
            </section>

            {/* Alerts */}
            <section className="bg-gray-800 shadow-xl rounded-2xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Bell className="w-5 h-5 mr-3 text-cyan-400" />
                Alerts
              </h2>

              {alerts.length === 0 ? (
                <p className="text-gray-500 text-sm">No matches detected yet.</p>
              ) : (
                <ul className="space-y-3 max-h-96 overflow-y-auto pr-2">
                  {alerts.map((a, idx) => {
                    const key = a.alert_id || `${a.target}-${a.camera_id}-${idx}`;
                    const expandedKey = a.alert_id || a.target;
                    return (
                      <li key={key} className="p-3 bg-gray-700/50 rounded-lg shadow-sm">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">
                            <b className="text-cyan-400">{a.target}</b> detected at{" "}
                            <span className="italic">{a.camera_name || `Cam ${a.camera_id}`}</span>
                            {getBadge(a.distance)}
                          </span>

                          <button
                            className="text-xs text-cyan-400 hover:underline"
                            onClick={() =>
                              setExpanded((prev) => ({ ...prev, [expandedKey]: !prev[expandedKey] }))
                            }
                          >
                            {expanded[expandedKey] ? "Hide" : "History"}
                          </button>
                        </div>

                        {expanded[expandedKey] && (
                          <ul className="mt-3 pl-5 text-xs text-gray-400 list-disc space-y-1">
                            {(Array.isArray(history?.[a.target]) ? history[a.target] : []).map((h, hIdx) => (
                              <li key={h.alert_id || `${a.target}-${hIdx}`}>
                                <span className="font-mono">
                                  [{new Date(h.timestamp).toLocaleTimeString()}]
                                </span>{" "}
                                at <span className="font-semibold">{h.camera_name || `Cam ${h.camera_id}`}</span>{" "}
                                - Dist: <span className="text-cyan-300">{h.distance?.toFixed(2)}</span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>

            {/* Movement Path Log */}
            <section className="bg-gray-800 shadow-xl rounded-2xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <GitBranch className="w-5 h-5 mr-3 text-cyan-400" />
                Movement Log
              </h2>
              {movementLog.length === 0 ? (
                <p className="text-gray-500 text-sm">No movement detected yet.</p>
              ) : (
                <ul className="list-inside space-y-2 text-sm max-h-60 overflow-y-auto pr-2">
                  {movementLog.map((log, idx) => (
                    <li key={`${log.timestamp}-${log.target}-${idx}`} className="text-gray-400">
                      <span className="font-mono text-cyan-400/70 mr-2">
                        [{new Date(log.timestamp).toLocaleTimeString()}]
                      </span>
                      <b className="text-gray-200">{log.target}</b> moved through{" "}
                      <span className="font-semibold">{log.camera_name || `Cam ${log.camera_id}`}</span>.
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
