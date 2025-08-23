import { WS_URL } from "../config/api";

export function connectWebSocket(onMessage) {
  const socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    // Connected
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage?.(data);
    } catch (e) {
      // Ignore non-JSON pings or other messages
    }
  };

  socket.onerror = () => {
    // Ignore; optional reconnection logic can be added later
  };

  return () => {
    try { socket.close(); } catch {}
  };
}


