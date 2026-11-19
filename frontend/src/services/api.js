const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8001/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "The EchoSafe service is unavailable.");
  return data;
}

export const analyzeText = (text, context = []) =>
  request("/analyze", { method: "POST", body: JSON.stringify({ text, context }) });
export const analyzeConversation = (messages) =>
  request("/analyze/conversation", { method: "POST", body: JSON.stringify({ messages }) });
export const getHealth = () => request("/health");
export const getResources = () => request("/resources");
export const getHistory = (limit = 100) => request(`/history?limit=${limit}`);
export const saveHistory = (entry) =>
  request("/history", { method: "POST", body: JSON.stringify(entry) });
export const deleteHistory = (id) => request(`/history/${id}`, { method: "DELETE" });
export const clearHistory = () => request("/history", { method: "DELETE" });
export const getExperiments = () => request("/experiments");
