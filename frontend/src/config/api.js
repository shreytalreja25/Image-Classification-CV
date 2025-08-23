export const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export const WS_URL = (API_BASE_URL.startsWith("https")
  ? API_BASE_URL.replace(/^https/, "wss")
  : API_BASE_URL.replace(/^http/, "ws")) + "/ws";


