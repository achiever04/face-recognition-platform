// frontend/src/components/CameraGrid.jsx
import React, { useEffect, useState, useRef } from "react";
import { getCameraStatus as apiGetCameraStatus } from "../api"; // from centralized api.js (optional)
 
function randomPick(arr, n) {
  const copy = arr.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}
 
export default function CameraGrid({ refreshInterval = 5000 }) {
  const [cameras, setCameras] = useState([]); // full camera objects
  const [displaySet, setDisplaySet] = useState([]);
  const [snapshots, setSnapshots] = useState({}); // camId -> { url, status, placeholder }
  const timerRef = useRef(null);
  const activeFetches = useRef(new Map()); // camId -> AbortController
  const objectUrls = useRef(new Map()); // camId -> objectUrl to revoke later
  const mountedRef = useRef(true);
 
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      // cleanup object URLs
      objectUrls.current.forEach((url) => {
        try {
          URL.revokeObjectURL(url);
        } catch (e) {}
      });
      objectUrls.current.clear();
      // abort inflight fetches
      activeFetches.current.forEach((ctrl) => {
        try {
          ctrl.abort();
        } catch (e) {}
      });
      activeFetches.current.clear();
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);
 
  // Fetch cameras list. Prefer API helper; fallback to plain fetch
  useEffect(() => {
    let mounted = true;
    async function fetchCams() {
      try {
        let list = [];
        if (apiGetCameraStatus) {
          // apiGetCameraStatus returns normalized { ok, data, error } in our api.js
          try {
            const resp = await apiGetCameraStatus();
            if (resp && resp.ok && resp.data) {
              // resp.data is status -> map of cam_id -> details
              const statusObj = resp.data.status || resp.data;
              // convert to array of camera objects
              for (const [cid, meta] of Object.entries(statusObj || {})) {
                list.push({ id: cid, name: meta.name, geo: meta.geo, healthy: meta.state === "ok" });
              }
            }
          } catch (e) {
            // fallback to fetch below
          }
        }
        if (list.length === 0) {
          // fallback endpoint
          const r = await fetch("/api/cameras");
          if (r.ok) {
            const json = await r.json();
            list = Array.isArray(json) ? json : json.cameras || json;
          }
        }
        if (mounted) setCameras(list || []);
      } catch (e) {
        console.error("Failed to fetch cameras:", e);
        if (mounted) setCameras([]);
      }
    }
    fetchCams();
    // refresh camera list every 60s to pick up added/removed cameras
    const camTimer = setInterval(fetchCams, 60 * 1000);
    return () => {
      mounted = false;
      clearInterval(camTimer);
    };
  }, []);
 
  // Rotation logic
  useEffect(() => {
    if (!cameras || cameras.length === 0) return;
    function refreshSet() {
      // Prefer healthy cameras for display
      const healthy = cameras.filter((c) => c.healthy !== false);
      const pool = healthy.length > 0 ? healthy : cameras;
      const picked = randomPick(pool, Math.min(4, pool.length));
      setDisplaySet(picked);
      // prime snapshot fetches
      picked.forEach((cam) => fetchSnapshot(cam));
    }
    refreshSet();
    timerRef.current = setInterval(refreshSet, refreshInterval);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [cameras, refreshInterval]);
 
  async function fetchSnapshot(cam, { cacheBust = true } = {}) {
    const camId = cam.id || cam.cam_id || cam.camera_id || cam._id;
    if (!camId) return;
    // skip if camera flagged unhealthy
    if (cam.healthy === false) {
      setSnapshots((s) => ({ ...s, [camId]: { url: null, status: "unhealthy", placeholder: true } }));
      return;
    }
    // abort any previous fetch for this cam
    const prevCtrl = activeFetches.current.get(camId);
    if (prevCtrl) {
      try {
        prevCtrl.abort();
      } catch (e) {}
      activeFetches.current.delete(camId);
    }
    const controller = new AbortController();
    activeFetches.current.set(camId, controller);
    const ts = cacheBust ? `?ts=${Date.now()}` : "";
    const url = `/api/camera/${encodeURIComponent(camId)}/snapshot${ts}`;
    setSnapshots((s) => ({ ...s, [camId]: { url: null, status: "loading", placeholder: false } }));
    try {
      const resp = await fetch(url, { signal: controller.signal });
      activeFetches.current.delete(camId);
      if (!resp.ok) {
        // placeholder case: server may return 404 with SVG placeholder
        const placeholder = resp.headers.get("X-Placeholder") === "1" || resp.status === 404;
        const enhanceHint = resp.headers.get("X-Enhance-Requested") === "1";
        if (placeholder) {
          setSnapshots((s) => ({ ...s, [camId]: { url: null, status: "placeholder", placeholder: true } }));
          return;
        }
        setSnapshots((s) => ({ ...s, [camId]: { url: null, status: "error", placeholder: false } }));
        return;
      }
      // check headers
      const isPlaceholderHeader = resp.headers.get("X-Placeholder") === "1";
      const enhanceHint = resp.headers.get("X-Enhance-Requested") === "1";
      const blob = await resp.blob();
      // create object URL, revoke previous
      const prevUrl = objectUrls.current.get(camId);
      if (prevUrl) {
        try {
          URL.revokeObjectURL(prevUrl);
        } catch (e) {}
      }
      const objectUrl = URL.createObjectURL(blob);
      objectUrls.current.set(camId, objectUrl);
      setSnapshots((s) => ({
        ...s,
        [camId]: { url: objectUrl, status: enhanceHint ? "enhancing" : "ok", placeholder: isPlaceholderHeader },
      }));
      // schedule revoke after 60s
      setTimeout(() => {
        const cur = objectUrls.current.get(camId);
        if (cur === objectUrl) {
          try {
            URL.revokeObjectURL(objectUrl);
          } catch (e) {}
          objectUrls.current.delete(camId);
          // we keep snapshots state url as-is; next refresh will update it
        }
      }, 60 * 1000);
    } catch (err) {
      activeFetches.current.delete(camId);
      if (err.name === "AbortError") {
        // aborted, ignore
        return;
      }
      console.error("Snapshot fetch error for", camId, err);
      setSnapshots((s) => ({ ...s, [camId]: { url: null, status: "error", placeholder: false } }));
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
            <div className="camera-body w-full h-48 bg-black flex items-center justify-center overflow-hidden">
              {snap && snap.status === "loading" && <div className="text-white">Loading...</div>}
              {snap && snap.status === "enhancing" && <div className="text-white">Enhancing...</div>}
              {snap && snap.status === "unhealthy" && <div className="text-white">Camera offline</div>}
              {snap && snap.status === "error" && <div className="text-white">Error</div>}
              {snap && snap.placeholder && <div className="text-white">No snapshot</div>}
              {snap && snap.url && (
                <img src={snap.url} alt={`cam-${camId}`} className="object-cover w-full h-full rounded" />
              )}
              {!snap && <div className="text-white">No snapshot</div>}
            </div>
            <div className="camera-footer mt-1 text-xs text-gray-300 flex gap-2">
              <button
                className="px-2 py-1 bg-blue-600 rounded text-white text-xs"
                onClick={() => fetchSnapshot(cam, { cacheBust: true })}
              >
                Refresh
              </button>
              <div style={{ marginLeft: "auto" }}>
                <small className="text-xs opacity-70">
                  {cam.healthy === false ? "Unhealthy" : "Live"}
                </small>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
