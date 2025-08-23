import { useEffect, useState } from "react";
import { getDashboardStats } from "../services/api";

export default function Predictions() {
  const [items, setItems] = useState([]);

  const load = () => getDashboardStats().then(d => setItems(d?.recent_predictions || [])).catch(() => {});
  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Recent Predictions</h2>
        <p className="text-sm text-slate-500">Latest predictions served by the backend</p>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b text-slate-500">
              <tr>
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 pr-4">Predicted Class</th>
                <th className="py-2 pr-4">Confidence</th>
                <th className="py-2 pr-4">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {items.map((r, idx) => (
                <tr key={r._id || idx}>
                  <td className="py-2 pr-4 text-slate-800">{r.model_name}</td>
                  <td className="py-2 pr-4 text-slate-800">{r.predicted_class}</td>
                  <td className="py-2 pr-4 text-slate-800">{(r.confidence * 100).toFixed?.(2)}%</td>
                  <td className="py-2 pr-4 text-slate-500">{new Date(r.timestamp).toLocaleString?.() || r.timestamp}</td>
                </tr>
              ))}
              {items.length === 0 && (
                <tr>
                  <td colSpan={4} className="py-6 text-center text-slate-400">No predictions yet</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}


