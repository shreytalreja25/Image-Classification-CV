export default function RecentPredictions({ items }) {
  const rows = Array.isArray(items) ? items : [];
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-2 text-sm font-medium text-gray-700">Recent Predictions</div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="border-b text-gray-500">
            <tr>
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">Predicted Class</th>
              <th className="py-2 pr-4">Confidence</th>
              <th className="py-2 pr-4">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {rows.map((r, idx) => (
              <tr key={r._id || idx} className="">
                <td className="py-2 pr-4 text-gray-800">{r.model_name}</td>
                <td className="py-2 pr-4 text-gray-800">{r.predicted_class}</td>
                <td className="py-2 pr-4 text-gray-800">{(r.confidence * 100).toFixed?.(2) ?? r.confidence}%</td>
                <td className="py-2 pr-4 text-gray-500">{new Date(r.timestamp).toLocaleString?.() || r.timestamp}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={4} className="py-6 text-center text-gray-400">No predictions yet</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}


