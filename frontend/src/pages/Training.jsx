import { useEffect, useState } from "react";
import { connectWebSocket } from "../services/ws";

export default function Training() {
  const [events, setEvents] = useState([]);

  useEffect(() => {
    const disconnect = connectWebSocket((msg) => {
      if (msg?.type === "training_progress") {
        setEvents((prev) => [msg.data, ...prev].slice(0, 100));
      }
    });
    return disconnect;
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Training Progress</h2>
        <p className="text-sm text-slate-500">Live updates from backend WebSocket</p>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <ul className="space-y-2 text-sm">
          {events.map((e, idx) => (
            <li key={idx} className="rounded-md border border-slate-100 p-2">
              <div className="font-medium text-slate-700">{e.model_name}</div>
              <div className="text-slate-600">Epoch {e.epoch}/{e.total_epochs} · Loss {e.loss} · Accuracy {e.accuracy}</div>
              <div className="text-xs text-slate-400">{new Date(e.timestamp).toLocaleString?.() || e.timestamp}</div>
            </li>
          ))}
          {events.length === 0 && <li className="text-slate-400">Waiting for training events...</li>}
        </ul>
      </div>
    </div>
  );
}


