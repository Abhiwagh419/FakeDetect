const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function handleResponse(res) {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.error) message = body.error;
    } catch {
      // response wasn't JSON — keep default message
    }
    throw new ApiError(message, res.status);
  }
  return res;
}

async function withNetworkErrors(fn) {
  try {
    return await fn();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      "Could not reach the FakeDetect backend. Is the Flask server running?",
      0
    );
  }
}

export function healthCheck() {
  return withNetworkErrors(async () => {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    await handleResponse(res);
    return res.json();
  });
}

export function analyzeImage(file, { normalizeDomain = false } = {}) {
  return withNetworkErrors(async () => {
    const formData = new FormData();
    formData.append("image", file);
    if (normalizeDomain) formData.append("normalize_domain", "true");
    const res = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: "POST",
      body: formData,
    });
    await handleResponse(res);
    return res.json();
  });
}

export function generateReport({ file, analysisId }) {
  return withNetworkErrors(async () => {
    const formData = new FormData();
    if (analysisId) {
      formData.append("analysis_id", analysisId);
    } else if (file) {
      formData.append("image", file);
    }
    const res = await fetch(`${API_BASE_URL}/api/report`, {
      method: "POST",
      body: formData,
    });
    await handleResponse(res);
    const blob = await res.blob();
    return blob;
  });
}

export function downloadReportById(analysisId) {
  return withNetworkErrors(async () => {
    const res = await fetch(`${API_BASE_URL}/api/report/${analysisId}`);
    await handleResponse(res);
    return res.blob();
  });
}

export function getHistory({ prediction, search, limit } = {}) {
  return withNetworkErrors(async () => {
    const params = new URLSearchParams();
    if (prediction) params.set("prediction", prediction);
    if (search) params.set("search", search);
    if (limit) params.set("limit", limit);
    const res = await fetch(`${API_BASE_URL}/api/history?${params.toString()}`);
    await handleResponse(res);
    return res.json();
  });
}

export function getAnalysis(analysisId) {
  return withNetworkErrors(async () => {
    const res = await fetch(`${API_BASE_URL}/api/history/${analysisId}`);
    await handleResponse(res);
    return res.json();
  });
}

export function deleteAnalysis(analysisId) {
  return withNetworkErrors(async () => {
    const res = await fetch(`${API_BASE_URL}/api/history/${analysisId}`, {
      method: "DELETE",
    });
    await handleResponse(res);
    return res.json();
  });
}

export function getDashboard() {
  return withNetworkErrors(async () => {
    const res = await fetch(`${API_BASE_URL}/api/dashboard`);
    await handleResponse(res);
    return res.json();
  });
}

export function getModelInfo() {
  return withNetworkErrors(async () => {
    const res = await fetch(`${API_BASE_URL}/api/model-info`);
    await handleResponse(res);
    return res.json();
  });
}

export function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export { ApiError, API_BASE_URL };
