// frontend/src/components/FaceUpload.jsx
import React, { useState, useEffect, useRef } from "react";
// Use centralized api client
import {
  uploadFace,
  compareFaces,
  listFaces,
  deleteFace,
  uploadWithProgress,
  enqueueAsyncFaceSearch,
  pollJob,
  socket,
} from "../api";

const FaceUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [compareFile, setCompareFile] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [comparing, setComparing] = useState(false);

  // --- NEW: State for new features ---
  const [targetName, setTargetName] = useState("");
  const [saveRaw, setSaveRaw] = useState(false);
  const [enrolledFaces, setEnrolledFaces] = useState([]);
  const [loadingFaces, setLoadingFaces] = useState(false);

  // --- NEW: Async-upload mode + progress
  const [useAsyncUpload, setUseAsyncUpload] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentJobId, setCurrentJobId] = useState(null);
  const fileInputRef = useRef(null);

  // small helper: normalize API responses (supports both old axios style and new safeRequest)
  const normalizeApiResponse = (res) => {
    // New style: { ok, data, error }
    if (res && (res.ok === true || res.ok === false)) {
      return {
        ok: res.ok,
        payload: res.data,
        error: res.error,
      };
    }
    // Legacy axios response: res.data contains payload
    if (res && res.data !== undefined) {
      return { ok: true, payload: res.data, error: null };
    }
    return { ok: false, payload: null, error: { message: "No response" } };
  };

  // --- NEW: Function to fetch all enrolled faces ---
  const fetchEnrolledFaces = async () => {
    setLoadingFaces(true);
    try {
      const res = await listFaces();
      const { ok, payload } = normalizeApiResponse(res);
      if (ok && payload) {
        // Support old and new payload shapes (targets or data.targets)
        const targets = payload.targets || payload || [];
        setEnrolledFaces(Array.isArray(targets) ? targets : []);
      } else {
        setEnrolledFaces([]);
      }
    } catch (error) {
      console.error("Error fetching faces:", error);
      setEnrolledFaces([]); // Set to empty on error
    }
    setLoadingFaces(false);
  };

  // --- NEW: Load faces on component mount ---
  useEffect(() => {
    fetchEnrolledFaces();
  }, []);

  // --- NEW: Socket listener for async job completion ---
  useEffect(() => {
    function onJobFinished(data) {
      try {
        // data expected { job_id, result } from backend
        if (!data || !data.job_id) return;
        if (currentJobId && data.job_id === currentJobId) {
          // Normalize result if needed
          const result = data.result ?? data;
          setUploadResult(result);
          setUploading(false);
          setUploadProgress(100);
          setCurrentJobId(null);
          // refresh enrolled faces if detection resulted in new enrollment
          fetchEnrolledFaces();
        }
      } catch (e) {
        console.warn("Error handling job_finished socket event:", e);
      }
    }

    // register only if socket present
    try {
      if (socket && socket.on) {
        socket.on("job_finished", onJobFinished);
      }
    } catch (e) {
      console.warn("Socket attach failed (continuing without socket):", e);
    }

    return () => {
      try {
        if (socket && socket.off) {
          socket.off("job_finished", onJobFinished);
        }
      } catch (e) {
        // ignore
      }
    };
  }, [currentJobId]);

  // --- NEW: Function to delete a face ---
  const handleDelete = async (target) => {
    if (!window.confirm(`Are you sure you want to delete ${target}?`)) {
      return;
    }
    try {
      const res = await deleteFace(target);
      const { ok, payload, error } = normalizeApiResponse(res);
      if (ok && payload && payload.status === "success") {
        alert(payload.message);
        fetchEnrolledFaces(); // Re-fetch the list
      } else {
        const msg = payload?.message || error?.message || "Failed to delete";
        alert(`Error: ${msg}`);
      }
    } catch (error) {
      const errorMsg =
        error.response?.data?.detail || "Failed to delete face.";
      console.error("Error deleting face:", error);
      alert(`Error: ${errorMsg}`);
    }
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setSelectedFile(f);
    setUploadResult(null);
    setCompareResult(null);
    // --- NEW: Auto-fill target name ---
    if (f && !targetName) {
      setTargetName(f.name);
    }
  };

  const handleCompareFileChange = (e) => {
    setCompareFile(e.target.files[0]);
    setCompareResult(null);
  };

  // --- UPGRADED: handleUploadAndEncode ---
  const handleUploadAndEncode = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadResult(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      // Use user-provided name, fallback to filename
      formData.append("target_name", targetName || selectedFile.name);
      formData.append("save_raw", saveRaw);

      if (useAsyncUpload) {
        // Enqueue async job using async task manager
        setUploadProgress(1);
        const enqueueResp = await enqueueAsyncFaceSearch(formData, (loaded, total) => {
          const perc = total ? Math.round((loaded / total) * 100) : 0;
          setUploadProgress(perc);
        });

        const { ok: enqueuedOk, payload: enqueuePayload, error: enqueueError } =
          normalizeApiResponse(enqueueResp);

        if (!enqueuedOk) {
          // Enqueue failed
          const errMsg = enqueueError?.message || "Failed to enqueue job";
          setUploadResult({ status: "error", message: errMsg });
          alert(`❌ ${errMsg}`);
          setUploading(false);
          return;
        }

        // Expect job_id from payload (legacy or new)
        const jobId = enqueuePayload?.job_id || enqueuePayload?.jobId || enqueuePayload?.job || null;
        if (!jobId) {
          // If backend immediately returns result, handle it
          setUploadResult(enqueuePayload);
          fetchEnrolledFaces();
          setUploading(false);
          setUploadProgress(100);
          return;
        }

        setCurrentJobId(jobId);
        // Try to rely on socket event for job completion; if not, poll
        const pollResp = await pollJob(jobId, {
          interval: 800,
          timeout: 120000,
          onUpdate: (update) => {
            // update: normalized safeRequest response {ok,data,error}
            const { ok, payload } = normalizeApiResponse(update);
            if (ok && payload && payload.status) {
              // If server supplies progress via payload.status, show it
              // Not all servers provide this; it's optional
            }
          },
        });

        const { ok: pollOk, payload: pollPayload, error: pollError } = normalizeApiResponse(pollResp);
        if (!pollOk) {
          const msg = pollError?.message || "Job polling failed";
          setUploadResult({ status: "error", message: msg });
          alert(`❌ ${msg}`);
        } else {
          // pollPayload likely contains final job object with status and result
          setUploadResult(pollPayload);
          if (pollPayload && (pollPayload.status === "success" || pollPayload.status === "finished")) {
            // Refresh enrolled faces if applicable
            fetchEnrolledFaces();
            // Clear inputs on success
            setSelectedFile(null);
            setTargetName("");
            if (fileInputRef.current) fileInputRef.current.value = null;
          }
        }
      } else {
        // Synchronous direct upload (legacy)
        const onProgress = (loaded, total) => {
          const perc = total ? Math.round((loaded / total) * 100) : 0;
          setUploadProgress(perc);
        };
        const resp = await uploadWithProgress("/face/upload", formData, onProgress);
        const { ok, payload, error } = normalizeApiResponse(resp);

        if (ok && payload) {
          // payload expected to have { status: "success"|"warning"|"error", ... }
          setUploadResult(payload);
          if (payload.status === "success") {
            alert(`✅ Face encoded for ${payload.target}`);
            fetchEnrolledFaces();
            // Clear inputs on success
            setSelectedFile(null);
            setTargetName("");
            if (fileInputRef.current) fileInputRef.current.value = null;
          } else {
            const message = payload.message || "Upload returned warning/error";
            alert(`❌ ${message}`);
          }
        } else {
          const msg = (error && error.message) || "Upload failed";
          setUploadResult({ status: "error", message: msg });
          alert(`❌ ${msg}`);
        }
      }
    } catch (error) {
      console.error(error);
      // Support axios-style errors and normalized errors
      const errMsg =
        error?.response?.data?.detail ||
        error?.response?.data?.message ||
        error?.message ||
        "Something went wrong!";
      setUploadResult({
        status: "error",
        message: errMsg,
      });
      alert(`❌ ${errMsg}`);
    } finally {
      setUploading(false);
      setUploadProgress(0);
      setCurrentJobId(null);
    }
  };

  // --- UPGRADED: handleCompare ---
  const handleCompare = async () => {
    if (!compareFile) return;

    setComparing(true);
    setCompareResult(null);

    const formData = new FormData();
    formData.append("file", compareFile);

    try {
      const res = await compareFaces(formData);
      const { ok, payload, error } = normalizeApiResponse(res);
      if (ok) {
        setCompareResult(payload);
      } else {
        const message = error?.message || "Comparison failed!";
        setCompareResult({ status: "error", message });
        alert(`❌ ${message}`);
      }
    } catch (error) {
      console.error(error);
      const errorMsg =
        error?.response?.data?.detail ||
        error?.response?.data?.message ||
        error?.message ||
        "Comparison failed!";
      setCompareResult({
        status: "error",
        message: errorMsg,
      });
      alert(`❌ ${errorMsg}`);
    } finally {
      setComparing(false);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "800px", margin: "0 auto" }}>
      <h2>📤 Upload Target Photo</h2>

      {/* Upload Section */}
      <div
        style={{
          marginBottom: "30px",
          padding: "20px",
          border: "1px solid #ddd",
          borderRadius: "8px",
        }}
      >
        <h3>Step 1: Upload & Encode Face</h3>
        <input
          id="upload-file-input"
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ marginBottom: "10px" }}
        />
        {/* Target Name Input */}
        <input
          type="text"
          placeholder="Enter Target Name (defaults to filename)"
          value={targetName}
          onChange={(e) => setTargetName(e.target.value)}
          style={{
            marginBottom: "10px",
            width: "calc(100% - 16px)",
            padding: "8px",
            boxSizing: "border-box",
          }}
        />
        {/* Save Raw Checkbox */}
        <label style={{ marginBottom: "10px", display: "block" }}>
          <input
            type="checkbox"
            checked={saveRaw}
            onChange={(e) => setSaveRaw(e.target.checked)}
            style={{ marginRight: "5px" }}
          />
          Save Raw Image on Server
        </label>

        {/* Async Upload Toggle */}
        <label style={{ marginBottom: "10px", display: "block" }}>
          <input
            type="checkbox"
            checked={useAsyncUpload}
            onChange={(e) => setUseAsyncUpload(e.target.checked)}
            style={{ marginRight: "5px" }}
          />
          Use Async Upload (enqueue job and process in background)
        </label>

        <button
          onClick={handleUploadAndEncode}
          disabled={uploading || !selectedFile}
          style={{
            padding: "10px 20px",
            backgroundColor: uploading || !selectedFile ? "#ccc" : "#4CAF50",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: uploading || !selectedFile ? "not-allowed" : "pointer",
          }}
        >
          {uploading ? "Uploading..." : "Upload & Encode"}
        </button>

        {/* Progress bar */}
        {uploading && (
          <div style={{ marginTop: "10px" }}>
            <div
              style={{
                height: "10px",
                background: "#eee",
                borderRadius: "6px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "10px",
                  width: `${uploadProgress}%`,
                  background: "#4caf50",
                }}
              />
            </div>
            <div style={{ fontSize: "12px", marginTop: "6px" }}>
              {uploadProgress}% complete
            </div>
          </div>
        )}

        {uploadResult && (
          <div
            style={{
              marginTop: "20px",
              padding: "15px",
              backgroundColor:
                uploadResult.status === "success" ||
                uploadResult.status === "warning"
                  ? "#d4edda"
                  : "#f8d7da",
              borderRadius: "4px",
            }}
          >
            <h4>Upload Result:</h4>
            {uploadResult.status === "warning" && (
              <p style={{ color: "#856404", fontWeight: "bold" }}>
                {uploadResult.message} (Score: {uploadResult.quality_score})
                <br />
                Issues: {uploadResult.issues && uploadResult.issues.join(", ")}
              </p>
            )}
            <pre
              style={{
                whiteSpace: "pre-wrap",
                wordWrap: "break-word",
                fontSize: "12px",
              }}
            >
              {JSON.stringify(uploadResult, null, 2)}
            </pre>
          </div>
        )}
      </div>

      {/* Compare Section */}
      <div
        style={{
          padding: "20px",
          border: "1px solid #ddd",
          borderRadius: "8px",
        }}
      >
        <h3>Step 2: Compare Another Face</h3>
        <p style={{ fontSize: "14px", color: "#666" }}>
          Upload a different image to compare against all stored faces
        </p>
        <input
          type="file"
          accept="image/*"
          onChange={handleCompareFileChange}
          style={{ marginBottom: "10px" }}
        />
        <button
          onClick={handleCompare}
          disabled={comparing || !compareFile}
          style={{
            padding: "10px 20px",
            backgroundColor: comparing || !compareFile ? "#ccc" : "#2196F3",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: comparing || !compareFile ? "not-allowed" : "pointer",
          }}
        >
          {comparing ? "Comparing..." : "Compare Face"}
        </button>

        {compareResult && (
          <div
            style={{
              marginTop: "20px",
              padding: "15px",
              backgroundColor:
                compareResult.status === "success" ? "#d4edda" : "#f8d7da",
              borderRadius: "4px",
            }}
          >
            <h4>Comparison Results:</h4>
            {compareResult.status === "success" &&
            compareResult.comparisons ? (
              <div>
                {compareResult.comparisons.length === 0 ? (
                  <p>No matches found</p>
                ) : (
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      marginTop: "10px",
                    }}
                  >
                    <thead>
                      <tr style={{ backgroundColor: "#f0f0f0" }}>
                        <th style={{ padding: "8px", border: "1px solid #ddd" }}>
                          Target
                        </th>
                        <th style={{ padding: "8px", border: "1px solid #ddd" }}>
                          Match
                        </th>
                        <th style={{ padding: "8px", border: "1px solid #ddd" }}>
                          Distance
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareResult.comparisons.map((comp, idx) => (
                        <tr key={idx}>
                          <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                            {comp.target}
                          </td>
                          <td
                            style={{
                              padding: "8px",
                              border: "1px solid #ddd",
                              color: comp.match ? "green" : "red",
                              fontWeight: "bold",
                            }}
                          >
                            {comp.match ? "✓ Yes" : "✗ No"}
                          </td>
                          <td style={{ padding: "8px", border: "1px solid #ddd" }}>
                            {typeof comp.distance === "number"
                              ? comp.distance.toFixed(3)
                              : comp.distance}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ) : (
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  wordWrap: "break-word",
                  fontSize: "12px",
                }}
              >
                {JSON.stringify(compareResult, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>

      {/* Step 3 - Manage Enrolled Faces */}
      <div
        style={{
          marginTop: "30px",
          padding: "20px",
          border: "1px solid #ddd",
          borderRadius: "8px",
        }}
      >
        <h3>Step 3: Manage Enrolled Faces</h3>
        {loadingFaces ? (
          <p>Loading faces...</p>
        ) : (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {enrolledFaces.length === 0 ? (
              <p style={{ color: "#666" }}>No faces enrolled yet.</p>
            ) : (
              enrolledFaces.map((target) => (
                <li
                  key={target}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "8px 4px",
                    borderBottom: "1px solid #eee",
                  }}
                >
                  <span style={{ fontFamily: "monospace" }}>{target}</span>
                  <button
                    onClick={() => handleDelete(target)}
                    style={{
                      padding: "4px 8px",
                      backgroundColor: "#f44336",
                      color: "white",
                      border: "none",
                      borderRadius: "4px",
                      cursor: "pointer",
                      fontSize: "12px",
                    }}
                  >
                    Delete
                  </button>
                </li>
              ))
            )}
          </ul>
        )}
      </div>
    </div>
  );
};

export default FaceUpload;
