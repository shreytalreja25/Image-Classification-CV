import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

export default function PerformanceChart({ data }) {
  const chartData = useMemo(() => {
    if (!Array.isArray(data)) return [];
    // Map to a simple structure; assumes each item has model_name and accuracy
    return data.map((d, idx) => ({ name: d.model_name || `Model ${idx+1}`, accuracy: d.accuracy ?? 0 }));
  }, [data]);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-2 text-sm font-medium text-gray-700">Model Performance (Accuracy)</div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" height={60} />
            <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
            <Tooltip formatter={(v) => `${v.toFixed?.(2) ?? v}%`} />
            <Legend />
            <Line type="monotone" dataKey="accuracy" stroke="#2563eb" strokeWidth={2} dot={false} name="Accuracy" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}


