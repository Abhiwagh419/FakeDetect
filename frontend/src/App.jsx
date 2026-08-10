import React, { useState } from "react";

const API_URL = "http://localhost:5000/api/predict";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Could not reach the detection server. Is the Flask backend running?");
    } finally {
      setLoading(false);
    }
  };

  const isReal = result?.prediction?.toUpperCase() === "REAL";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center py-12 px-4">
      <h1 className="text-3xl font-bold mb-2">AI-Generated Image Detector</h1>
      <p className="text-slate-400 mb-8 text-center max-w-md">
        Upload an image. A locally-trained CNN decides if it's real or AI-generated — no external API involved.
      </p>

      <div className="bg-slate-900 rounded-xl p-6 w-full max-w-md border border-slate-800">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="block w-full text-sm text-slate-300 mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-slate-800 file:text-slate-100 hover:file:bg-slate-700"
        />

        {preview && (
          <img src={preview} alt="preview" className="rounded-lg mb-4 w-full object-cover max-h-64" />
        )}

        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:cursor-not-allowed font-medium transition"
        >
          {loading ? "Analyzing..." : "Detect"}
        </button>

        {error && <p className="text-red-400 text-sm mt-3">{error}</p>}
      </div>

      {result && (
        <div className="bg-slate-900 rounded-xl p-6 w-full max-w-md border border-slate-800 mt-6">
          <div className="flex items-center justify-between mb-4">
            <span className="text-slate-400 text-sm">Prediction</span>
            <span
              className={`px-3 py-1 rounded-full text-sm font-semibold ${
                isReal ? "bg-emerald-900 text-emerald-300" : "bg-rose-900 text-rose-300"
              }`}
            >
              {result.prediction}
            </span>
          </div>

          <div className="mb-4">
            <div className="flex justify-between text-sm text-slate-400 mb-1">
              <span>Confidence</span>
              <span>{result.confidence}%</span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${isReal ? "bg-emerald-500" : "bg-rose-500"}`}
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>

          <p className="text-slate-400 text-sm mb-2">Grad-CAM — regions that influenced the decision:</p>
          <img
            src={`data:image/png;base64,${result.heatmap}`}
            alt="Grad-CAM heatmap"
            className="rounded-lg w-full"
          />
        </div>
      )}
    </div>
  );
}
