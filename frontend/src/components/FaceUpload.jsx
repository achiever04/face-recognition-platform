// --- IMPROVED: Added useEffect and imported from api.js ---
import React, { useState, useEffect } from "react";
// import axios from "axios"; // No longer needed, we use api.js
import {
  uploadFace,
  compareFaces,
  listFaces,
  deleteFace,
} from "../api"; // Use the central API file

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

  // const backendURL = "http://127.0.0.1:8000"; // No longer needed

  // --- NEW: Function to fetch all enrolled faces ---
  const fetchEnrolledFaces = async () => {
    setLoadingFaces(true);
    try {
      const res = await listFaces(); // from api.js
      if (res.data.status === "success") {
        setEnrolledFaces(res.data.targets || []);
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

  // --- NEW: Function to delete a face ---
  const handleDelete = async (target) => {
    if (!window.confirm(`Are you sure you want to delete ${target}?`)) {
      return;
    }
    try {
      const res = await deleteFace(target); // from api.js
      if (res.data.status === "success") {
        alert(res.data.message);
        fetchEnrolledFaces(); // Re-fetch the list
      } else {
        alert(`Error: ${res.data.message}`);
      }
    } catch (error) {
      const errorMsg =
        error.response?.data?.detail || "Failed to delete face.";
      console.error("Error deleting face:", error);
      alert(`Error: ${errorMsg}`);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadResult(null);
    setCompareResult(null);
    // --- NEW: Auto-fill target name ---
    if (e.target.files[0] && !targetName) {
      setTargetName(e.target.files[0].name);
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

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      // --- NEW: Add target_name and save_raw to FormData ---
      // Use user-provided name, fallback to filename
      formData.append("target_name", targetName || selectedFile.name);
      formData.append("save_raw", saveRaw);

      // --- IMPROVED: Use imported API function ---
      const uploadRes = await uploadFace(formData);

      setUploadResult(uploadRes.data);

      if (uploadRes.data.status === "success") {
        alert(`✅ Face encoded for ${uploadRes.data.target}`);
        fetchEnrolledFaces(); // --- NEW: Refresh list on success
        // --- NEW: Clear inputs on success ---
        setSelectedFile(null);
        setTargetName("");
        document.getElementById("upload-file-input").value = null; // Clear file input
      } else {
        // Handle quality warnings from the new endpoint
        const message =
          uploadRes.data.status === "warning"
            ? `${uploadRes.data.message} (Score: ${uploadRes.data.quality_score})`
            : uploadRes.data.message;
        alert(`❌ ${message}`);
      }
    } catch (error) {
      console.error(error);
      // Handle detailed errors from the new endpoint
      const errorMsg =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        "Something went wrong!";
      setUploadResult({
        status: "error",
        message: errorMsg,
      });
      alert(`❌ ${errorMsg}`);
    }

    setUploading(false);
  };

  // --- UPGRADED: handleCompare ---
  const handleCompare = async () => {
    if (!compareFile) return;

    setComparing(true);
    setCompareResult(null);

    const formData = new FormData();
    formData.append("file", compareFile);

    try {
      // --- IMPROVED: Use imported API function ---
      const res = await compareFaces(formData);

      setCompareResult(res.data);
    } catch (error) {
      console.error(error);
      const errorMsg =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        "Comparison failed!";
      setCompareResult({
        status: "error",
        message: errorMsg,
      });
      alert(`❌ ${errorMsg}`);
    }

    setComparing(false);
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
          id="upload-file-input" // --- NEW: Added id to allow clearing
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ marginBottom: "10px" }}
        />
        {/* --- NEW: Target Name Input --- */}
        <input
          type="text"
          placeholder="Enter Target Name (defaults to filename)"
          value={targetName}
          onChange={(e) => setTargetName(e.target.value)}
          style={{
            marginBottom: "10px",
            width: "calc(100% - 16px)", // Fix width
            padding: "8px",
            boxSizing: "border-box", // Fix box model
          }}
        />
        {/* --- NEW: Save Raw Checkbox --- */}
        <label style={{ marginBottom: "10px", display: "block" }}>
          <input
            type="checkbox"
            checked={saveRaw}
            onChange={(e) => setSaveRaw(e.target.checked)}
            style={{ marginRight: "5px" }}
          />
          Save Raw Image on Server
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
            {/* --- NEW: Better display for quality warnings --- */}
            {uploadResult.status === "warning" && (
              <p style={{ color: "#856404", fontWeight: "bold" }}>
                {uploadResult.message} (Score: {uploadResult.quality_score})
                <br />
                Issues: {uploadResult.issues.join(", ")}
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
                        <th
                          style={{ padding: "8px", border: "1px solid #ddd" }}
                        >
                          Target
                        </th>
                        <th
                          style={{ padding: "8px", border: "1px solid #ddd" }}
                        >
                          Match
                        </th>
                        <th
                          style={{ padding: "8px", border: "1.px solid #ddd" }}
                        >
                          Distance
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareResult.comparisons.map((comp, idx) => (
                        <tr key={idx}>
                          <td
                            style={{ padding: "8px", border: "1px solid #ddd" }}
                          >
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
                          <td
                            style={{ padding: "8px", border: "1px solid #ddd" }}
                          >
                            {comp.distance.toFixed(3)}
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

      {/* --- NEW: Step 3 - Manage Enrolled Faces --- */}
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