// frontend/src/components/CameraGrid.jsx
import React, { useEffect, useState, useRef } from "react";
// Try to import your api helper; fallback to fetch
let api = null;
try {
  // Adjust path if your api.js exports differently
  // e.g., export default axiosInstance; or named exports.
  api = require("../api").default || require("../api").api || null;
} catch (e) {
  api = null;
}

function randomPick(arr, n) {
  const copy = arr.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

export default function CameraGrid({ refreshInterval = 5000 }) {
  const [cameras, setCameras] = useState([]);
  const [displaySet, setDisplaySet] = useState([]);
  const [snapshots, setSnapshots] = useState({}); // camId -> {url, status}
  const timerRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    async function fetchCams() {
      try {
        let res;
        if (api && api.get) {
          res = await api.get("/api/cameras");
          // assume data is in res.data or res
          const data = res.data || res;
          if (mounted) setCameras(data || []);
        } else {
          const r = await fetch("/api/cameras");
          const json = await r.json();
          if (mounted) setCameras(json || []);
        }
      } catch (e) {
        console.error("Failed to fetch cameras:", e);
        setCameras([]);
      }
    }
    fetchCams();
    return () => (mounted = false);
  }, []);

  useEffect(() => {
    if (cameras.length === 0) return;

    function refreshSet() {
      const picked = randomPick(cameras, Math.min(4, cameras.length));
      setDisplaySet(picked);
      // prime snapshots
      picked.forEach((cam) => {
        fetchSnapshot(cam);
      });
    }

    refreshSet();
    timerRef.current = setInterval(refreshSet, refreshInterval);
    return () => clearInterval(timerRef.current);
  }, [cameras, refreshInterval]);

  async function fetchSnapshot(cam) {
    const camId = cam.id || cam.cam_id || cam.camera_id || cam._id;
    if (!camId) return;
    const url = `/api/camera/${encodeURIComponent(camId)}/snapshot?ts=${Date.now()}`;
    // optimistic set loading
    setSnapshots((s) => ({
      ...s,
      [camId]: { url: null, status: "loading" },
    }));

    try {
      // Try using direct fetch to get headers, fallback to img tag if CORS forbids
      const resp = await fetch(url);
      if (!resp.ok) {
        setSnapshots((s) => ({ ...s, [camId]: { url: null, status: "error" } }));
        return;
      }
      // check header for enhancement hint
      const enhance = resp.headers.get("X-Enhance-Requested");
      const blob = await resp.blob();
      const objectUrl = URL.createObjectURL(blob);
      setSnapshots((s) => ({
        ...s,
        [camId]: { url: objectUrl, status: enhance ? "enhancing" : "ok" },
      }));
      // revoke objectUrl after some time to avoid leak
      setTimeout(() => URL.revokeObjectURL(objectUrl), 60 * 1000);
    } catch (err) {
      console.error("Snapshot fetch error", err);
      setSnapshots((s) => ({ ...s, [camId]: { url: null, status: "error" } }));
    }
  }

  return (
    <div className="camera-grid p-2 grid grid-cols-2 gap-2">
      {displaySet.map((cam) => {
        const camId = cam.id || cam.cam_id || cam.camera_id || cam._id;
        const snap = snapshots[camId];
        return (
          <div key={camId} className="camera-card bg-gray-900 rounded p-1">
            <div className="camera-header text-white text-sm mb-1 flex justify-between">
              <span>{cam.name || cam.location || camId}</span>
              <span className="text-xs opacity-70">{cam.geo || cam.coords || ""}</span>
            </div>
            <div className="camera-body w-full h-48 bg-black flex items-center justify-center">
              {snap && snap.status === "loading" && <div className="text-white">Loading...</div>}
              {snap && snap.status === "enhancing" && <div className="text-white">Enhancing...</div>}
              {snap && snap.status === "error" && <div className="text-white">Error</div>}
              {snap && snap.url && (
                <img src={snap.url} alt={`cam-${camId}`} className="object-cover w-full h-full rounded" />
              )}
              {!snap && <div className="text-white">No snapshot</div>}
            </div>
            <div className="camera-footer mt-1 text-xs text-gray-300">
              <button
                className="px-2 py-1 bg-blue-600 rounded text-white text-xs"
                onClick={() => fetchSnapshot(cam)}
              >
                Refresh
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
