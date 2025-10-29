import React, { useState, useEffect } from "react";
// import axios from "axios"; // No longer needed, use api.js
import {
  Camera,
  UploadCloud,
  Bell,
  GitBranch,
  Radar,
  Server,
  RefreshCw, // --- NEW: Icon for Aggregation
  Download, // --- NEW: Icon for Download Model
} from "lucide-react";

// --- NEW: Import API functions and the socket instance ---
import {
  socket, // The Socket.IO client instance
  getCameraStatus,
  uploadFace,
  getCameraAlerts,
  uploadFLWeights as apiUploadFLWeights, // Rename to avoid conflict
  getFLWeights as apiGetFLWeights,       // Rename to avoid conflict
  aggregateWeights,
  getAggregatedModel,
} from "./api"; // Use the central API file

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [cameraStatus, setCameraStatus] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [history, setHistory] = useState({});
  const [movementLog, setMovementLog] = useState([]);
  const [randomCameras, setRandomCameras] = useState([0, 0, 0, 0]);
  const [trackingCamera, setTrackingCamera] = useState(null);
  const [trackingTarget, setTrackingTarget] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [flWeights, setFlWeights] = useState({});
  const [brokenFeeds, setBrokenFeeds] = useState(new Set());
  const [socketConnected, setSocketConnected] = useState(false); // --- NEW: Track socket status

  // --- NEW: State for FL ---
  const [flClientId, setFlClientId] = useState("client_1"); // Default client
  const [aggregating, setAggregating] = useState(false);
  const [aggregatedModel, setAggregatedModel] = useState(null);

  // const backendUrl = "http://127.0.0.1:8000"; // No longer needed directly

  // ------------------------
  // Check camera status (Using api.js)
  // ------------------------
  useEffect(() => {
    const checkCamera = async () => {
      try {
        // const res = await axios.get(`${backendUrl}/camera/status`); // OLD
        const res = await getCameraStatus(); // --- IMPROVED: Use api.js
        setCameraStatus(res.data.status || {});
      } catch (error) {
        console.error("Camera status error:", error);
        setCameraStatus({ error: "❌ Camera API not reachable" });
      }
    };

    checkCamera();
    const interval = setInterval(checkCamera, 5000);
    return () => clearInterval(interval);
  }, []);

  // ------------------------
  // Upload & Encode Target Photo (Using api.js)
  // ------------------------
  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus("Uploading...");
    const formData = new FormData();
    formData.append("file", file);
    // Use filename as default target name if not provided elsewhere
    formData.append("target_name", file.name);
    formData.append("save_raw", false); // Or add state for this

    try {
      // const res = await axios.post(`${backendUrl}/face/upload`, formData, { // OLD
      //   headers: { "Content-Type": "multipart/form-data" },
      // });
      const res = await uploadFace(formData); // --- IMPROVED: Use api.js

      if (res.data.status === "success") {
        setUploadStatus(
          `✅ ${res.data.message || `Uploaded ${res.data.filename}`}`
        );
        // --- NEW: Clear file input on success ---
        setFile(null);
        if (document.getElementById("upload-input")) {
            document.getElementById("upload-input").value = null;
        }
      } else {
        setUploadStatus(`❌ ${res.data.message || "Upload failed"}`);
      }
    } catch (err) {
      console.error(err);
      setUploadStatus(
        `❌ ${err.response?.data?.detail || err.response?.data?.message || "Error uploading file"}`
      );
    }
  };

  // ------------------------------------
  // --- REAL-TIME VIA WEBSOCKETS ---
  // ------------------------------------
  useEffect(() => {
    // --- 1. Load initial data via REST API ---
    const fetchInitialData = async () => {
      console.log("Fetching initial data...");
      try {
        const res = await getCameraAlerts(); // Use api.js
        if (res.data.status === "success") {
          setAlerts(res.data.alerts || []);
          setHistory(res.data.history || {});
          setMovementLog(res.data.movement_log || []);

          if (res.data.latest_detection) {
            setTrackingCamera(res.data.latest_detection.camera_id);
            setTrackingTarget(res.data.latest_detection.target);
          }
           console.log("Initial data loaded.");
        }
      } catch (err) {
        console.error("Error fetching initial alerts:", err);
      }
    };

    fetchInitialData();

    // --- 2. Setup Socket.IO listeners ---
    console.log("Setting up Socket.IO listeners...");

    const handleConnect = () => {
        console.log("Socket.IO Connected:", socket.id);
        setSocketConnected(true);
        // Fetch initial data again on reconnect in case something was missed
        fetchInitialData();
    };

    const handleDisconnect = (reason) => {
        console.log("Socket.IO Disconnected:", reason);
        setSocketConnected(false);
    };

    const handleNewAlert = ({ alert }) => {
        console.log("Received new_alert:", alert);
        setAlerts((prevAlerts) => [alert, ...prevAlerts.slice(0, 49)]); // Add to top, limit to 50
        // Update history for the specific target
        setHistory(prevHistory => ({
            ...prevHistory,
            [alert.target]: [alert, ...(prevHistory[alert.target] || [])].slice(0, 10) // Add to history, limit
        }));
    };

    const handleUpdateMovementLog = ({ log }) => {
        console.log("Received update_movement_log:", log);
        setMovementLog((prevLog) => [log, ...prevLog.slice(0, 99)]); // Add to top, limit
    };

    const handleUpdateTrackingFeed = ({ detection }) => {
        console.log("Received update_tracking_feed:", detection);
        if (detection) {
            setTrackingCamera(detection.camera_id);
            setTrackingTarget(detection.person);
        }
    };

    // Attach listeners
    socket.on("connect", handleConnect);
    socket.on("disconnect", handleDisconnect);
    socket.on("new_alert", handleNewAlert);
    socket.on("update_movement_log", handleUpdateMovementLog);
    socket.on("update_tracking_feed", handleUpdateTrackingFeed);

    // Initial connection state
    setSocketConnected(socket.connected);

    // --- 3. Cleanup listeners on component unmount ---
    return () => {
      console.log("Cleaning up Socket.IO listeners...");
      socket.off("connect", handleConnect);
      socket.off("disconnect", handleDisconnect);
      socket.off("new_alert", handleNewAlert);
      socket.off("update_movement_log", handleUpdateMovementLog);
      socket.off("update_tracking_feed", handleUpdateTrackingFeed);
    };
  }, []); // Run only once on mount


  // -----------------------------------------
  // --- POLLING LOGIC (NOW COMMENTED OUT) ---
  // -----------------------------------------
  // useEffect(() => {
  //   const fetchAlerts = async () => {
  //     try {
  //       const res = await axios.get(`${backendUrl}/camera/alerts`);
  //       if (res.data.status === "success") {
  //         setAlerts(res.data.alerts || []);
  //         setHistory(res.data.history || {});
  //         setMovementLog(res.data.movement_log || []);

  //         if (res.data.latest_detection) {
  //           setTrackingCamera(res.data.latest_detection.camera_id);
  //           setTrackingTarget(res.data.latest_detection.target);
  //         }
  //       }
  //     } catch (err) {
  //       console.error("Error fetching alerts:", err);
  //     }
  //   };

  //   fetchAlerts();
  //   const interval = setInterval(fetchAlerts, 3000);
  //   return () => clearInterval(interval);
  // }, []);


  // ------------------------
  // Random 4 camera feeds
  // ------------------------
  useEffect(() => {
    const updateRandomCameras = () => {
      const availableCams = Object.keys(cameraStatus || {}).filter(
        (id) => cameraStatus[id]?.state === "ok" && !isNaN(parseInt(id)) // Ensure it's a number ID
      );

      if (availableCams.length === 0) return;

      const shuffled = [...availableCams].sort(() => 0.5 - Math.random());
      const selected = shuffled.slice(0, 4).map((id) => parseInt(id));
      // Ensure we always have 4, even if fewer cameras are available
      while (selected.length < 4 && availableCams.length > 0) {
          selected.push(parseInt(availableCams[0]));
      }
      // Handle edge case where no cameras are available initially but become available later
      if (selected.length > 0) {
        setRandomCameras(selected);
      }
    };

    // Update immediately and then set interval
    updateRandomCameras();
    const interval = setInterval(updateRandomCameras, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, [cameraStatus]); // Re-run when cameraStatus changes

  // ------------------------
  // Get confidence badge
  // ------------------------
  const getBadge = (distance) => {
    if (distance === undefined || distance === null) return null; // Handle missing distance
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
  // Federated Learning: Upload local weights (Using api.js)
  // ------------------------
  const uploadFLWeights = async () => {
    try {
      const target = flClientId; // Use state variable
      // Dummy weights for demo
      const weights = { layer1: [Math.random(), Math.random()], layer2: [Math.random(), Math.random()] };

      // const res = await axios.post( // OLD
      //   `${backendUrl}/face/fl/upload_weights`,
      //   { target: target, weights: weights },
      //   { headers: { "Content-Type": "application/json" } }
      // );
      const res = await apiUploadFLWeights(target, weights); // --- IMPROVED: Use api.js

      if (res.data.status === "success") {
        alert("✅ Federated weights uploaded successfully!");
        fetchFLWeights(); // Refresh display
      } else {
        alert(`❌ Failed to upload FL weights: ${res.data.message}`);
      }
    } catch (err) {
      console.error("Error uploading FL weights:", err);
      alert(`❌ Failed to upload FL weights: ${err.response?.data?.detail || err.message}`);
    }
  };

  // ------------------------
  // Federated Learning: Fetch current weights (Using api.js)
  // ------------------------
  const fetchFLWeights = async () => {
    try {
      const target = flClientId; // Use state variable
      // const res = await axios.get(`${backendUrl}/face/fl/get_weights`, { // OLD
      //   params: { target },
      // });
      const res = await apiGetFLWeights(target); // --- IMPROVED: Use api.js
      if (res.data.status === "success") {
        setFlWeights(res.data.weights || {});
      } else {
        setFlWeights({});
        console.warn(`Could not fetch weights for ${target}: ${res.data.message}`);
      }
    } catch (err) {
      console.error("Error fetching FL weights:", err);
      setFlWeights({});
    }
  };

  // Removed the polling useEffect for FL weights, fetch manually or on upload

  // --- NEW: FL Aggregation ---
  const handleAggregate = async () => {
    setAggregating(true);
    setAggregatedModel(null);
    try {
      // Example: Aggregate all known clients or specify some
      const res = await aggregateWeights(null, Date.now()); // Use timestamp as version
      if (res.data.status === "success") {
        alert(`✅ Aggregation successful! New model version: ${res.data.new_model_version}`);
        // Optionally fetch the new model details
        handleGetAggregatedModel();
      } else {
        alert(`❌ Aggregation failed: ${res.data.message}`);
      }
    } catch (err) {
      console.error("Error aggregating weights:", err);
      alert(`❌ Aggregation failed: ${err.response?.data?.detail || err.message}`);
    }
    setAggregating(false);
  };

  // --- NEW: FL Get Aggregated Model ---
   const handleGetAggregatedModel = async () => {
    try {
        const res = await getAggregatedModel();
        if (res.data.status === 'success') {
            setAggregatedModel(res.data);
        } else {
            alert(`❌ Failed to get aggregated model: ${res.data.message}`);
            setAggregatedModel(null);
        }
    } catch (err) {
        console.error("Error getting aggregated model:", err);
        alert(`❌ Failed to get aggregated model: ${err.response?.data?.detail || err.message}`);
        setAggregatedModel(null);
    }
  };


  // ------------------------
  // Handle camera feed errors
  // ------------------------
  const handleFeedError = (camId) => {
    // Only add if not already broken to prevent excessive state updates
    if (!brokenFeeds.has(camId)) {
        console.warn(`Camera feed error for Cam ID: ${camId}`);
        setBrokenFeeds((prev) => new Set(prev).add(camId));
    }
  };

  // --- NEW: Attempt to reconnect broken feeds periodically ---
  useEffect(() => {
    const retryInterval = setInterval(() => {
        if (brokenFeeds.size > 0) {
            console.log("Attempting to reconnect broken camera feeds...");
            // Simply remove from broken set, the img tag's src will cause a reload attempt
            setBrokenFeeds(new Set());
        }
    }, 15000); // Retry every 15 seconds

    return () => clearInterval(retryInterval);
  }, [brokenFeeds]);


  return (
    <div className="bg-gray-900 text-gray-200 min-h-screen font-sans">
      <div className="max-w-screen-xl mx-auto p-4 sm:p-6 lg:p-8">
        {/* Header */}
        <header className="border-b border-gray-700 pb-6 mb-8">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-teal-500">
            Multi-Camera Face Recognition Platform
          </h1>
          {/* --- IMPROVED: Show Socket.IO status --- */}
          <p className="text-gray-400 mt-2 flex items-center text-sm">
            <Server className="w-4 h-4 mr-2 text-cyan-500" />
            Backend: {`http://127.0.0.1:8000`} | WebSocket:{" "}
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
              {trackingCamera !== null && trackingTarget !== null ? ( // Ensure target is also set
                <div className="bg-black/20 rounded-lg overflow-hidden">
                  <div className="p-3 bg-gray-700/50 border-b border-gray-600">
                    <p className="text-sm">
                      Tracking{" "}
                      <b className="text-cyan-400">{trackingTarget}</b> at{" "}
                      <b className="text-cyan-400">
                        {cameraStatus[trackingCamera]?.name ||
                          `Camera ${trackingCamera}`}
                      </b>
                    </p>
                  </div>
                  {!brokenFeeds.has(trackingCamera) ? (
                    <img
                      // --- NEW: Add cache-busting query param ---
                      src={`http://127.0.0.1:8000/camera/${trackingCamera}/feed?_=${Date.now()}`}
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
                    // Use a more stable key if possible, but index is okay here
                    key={`${camId}-${idx}`}
                    className="bg-black/20 border-2 border-gray-700 rounded-lg overflow-hidden shadow-lg"
                  >
                    <p className="font-medium text-sm p-2 bg-gray-700/50">
                      {cameraStatus[camId]?.name || `Camera ${camId}`}
                    </p>
                    {!brokenFeeds.has(camId) ? (
                      <img
                        // --- NEW: Add cache-busting query param ---
                        src={`http://127.0.0.1:8000/camera/${camId}/feed?_=${Date.now()}`}
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
                  id="upload-input" // --- NEW: Added ID ---
                  type="file"
                  accept="image/*"
                  className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-cyan-500/10 file:text-cyan-300 hover:file:bg-cyan-500/20 cursor-pointer"
                  onChange={(e) => setFile(e.target.files[0])}
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
              {/* --- NEW: Input for Client ID --- */}
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
                {/* --- NEW: FL Aggregation Buttons --- */}
                 <button
                  className="w-full px-4 py-2 bg-purple-600 text-white font-bold rounded-lg hover:bg-purple-700 transition duration-300 disabled:opacity-50 flex items-center justify-center"
                  onClick={handleAggregate}
                  disabled={aggregating}
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${aggregating ? 'animate-spin' : ''}`} />
                  {aggregating ? "Aggregating..." : "Aggregate Weights"}
                </button>
                 <button
                  className="w-full px-4 py-2 bg-gray-600 text-white font-bold rounded-lg hover:bg-gray-700 transition duration-300 flex items-center justify-center"
                  onClick={handleGetAggregatedModel}
                >
                    <Download className="w-4 h-4 mr-2"/>
                  View Aggregated Model
                </button>

                <div className="mt-4">
                  <h3 className="text-sm text-gray-400 font-semibold mb-2">
                    Weights for '{flClientId}':
                     {/* --- NEW: Button to manually fetch --- */}
                     <button onClick={fetchFLWeights} className="ml-2 text-xs text-cyan-400 hover:underline">(Refresh)</button>
                  </h3>
                  <pre className="bg-gray-700 p-3 rounded-lg text-xs max-h-40 overflow-y-auto">
                    {JSON.stringify(flWeights, null, 2)}
                  </pre>
                  {/* --- NEW: Display aggregated model --- */}
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
                <ul className="space-y-3 max-h-96 overflow-y-auto pr-2"> {/* Added padding for scrollbar */}
                  {alerts.map((a, idx) => (
                    <li
                      // Use a more stable key like alert_id if available
                      key={a.alert_id || idx}
                      className="p-3 bg-gray-700/50 rounded-lg shadow-sm"
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-sm">
                          <b className="text-cyan-400">{a.target}</b> detected at{" "}
                          <span className="italic">
                            {a.camera_name || `Cam ${a.camera_id}`}
                          </span>
                          {getBadge(a.distance)}
                        </span>
                        <button
                          className="text-xs text-cyan-400 hover:underline"
                          onClick={() =>
                            setExpanded((prev) => ({
                              ...prev,
                              // Use alert_id for stable key
                              [a.alert_id || a.target]: !prev[a.alert_id || a.target],
                            }))
                          }
                        >
                          {/* Use alert_id for stable key */}
                          {expanded[a.alert_id || a.target] ? "Hide" : "History"}
                        </button>
                      </div>
                      {/* Use alert_id for stable key */}
                      {expanded[a.alert_id || a.target] && (
                        <ul className="mt-3 pl-5 text-xs text-gray-400 list-disc space-y-1">
                          {/* Ensure history[a.target] is an array */}
                          {(Array.isArray(history[a.target]) ? history[a.target] : []).map((h, hIdx) => (
                            <li key={h.alert_id || hIdx}> {/* Use stable key if available */}
                              <span className="font-mono">
                                [{new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}] {/* Format time */}
                              </span>{" "}
                              at{" "}
                              <span className="font-semibold">
                                {h.camera_name || `Cam ${h.camera_id}`}
                              </span>{" "}
                              - Dist:{" "}
                              <span className="text-cyan-300">
                                {h.distance?.toFixed(2)}
                              </span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </li>
                  ))}
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
                <p className="text-gray-500 text-sm">
                  No movement detected yet.
                </p>
              ) : (
                <ul className="list-inside space-y-2 text-sm max-h-60 overflow-y-auto pr-2"> {/* Added padding */}
                  {movementLog.map((log, idx) => (
                     // Use timestamp + target as a potentially more unique key
                    <li key={`${log.timestamp}-${log.target}-${idx}`} className="text-gray-400">
                      <span className="font-mono text-cyan-400/70 mr-2">
                        {/* Format time */}
                        [{new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}]
                      </span>
                      <b className="text-gray-200">{log.target}</b> moved through{" "}
                      <span className="font-semibold">
                        {log.camera_name || `Cam ${log.camera_id}`}
                      </span>
                      .
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