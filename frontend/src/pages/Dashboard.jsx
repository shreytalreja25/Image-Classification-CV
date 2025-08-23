import { useEffect, useMemo, useState } from "react";
import { getDashboardStats } from "../services/api";
import { connectWebSocket } from "../services/ws";
import KpiCard from "../components/KpiCard";
import PerformanceChart from "../components/PerformanceChart";
import RecentPredictions from "../components/RecentPredictions";

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  // Fetch initial stats
  useEffect(() => {
    let active = true;
    setLoading(true);
    getDashboardStats()
      .then((data) => { if (active) { setStats(data); setError(null); } })
      .catch((e) => { if (active) { setError(e?.message || "Failed to load"); } })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  // Minimal live updates via WS: refetch on heartbeat occasionally
  useEffect(() => {
    const disconnect = connectWebSocket((msg) => {
      if (msg?.type === "training_progress") {
        // Refresh stats when training progress events arrive
        getDashboardStats().then(setStats).catch(() => {});
      }
    });
    return disconnect;
  }, []);

  const kpis = useMemo(() => ({
    totalModels: stats?.total_models ?? 0,
    totalPredictions: stats?.total_predictions ?? 0,
    averageAccuracy: stats?.average_accuracy ? `${stats.average_accuracy.toFixed?.(2)}%` : "0.00%",
    bestModel: stats?.best_model ?? "None",
  }), [stats]);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-7xl p-4">
        <div className="mb-4">
          <h1 className="text-2xl font-semibold text-gray-900">Aerial Classification Dashboard</h1>
          <p className="text-sm text-gray-500">Real-time model performance and recent predictions</p>
        </div>

        {loading && (
          <div className="rounded-lg border border-gray-200 bg-white p-4 text-gray-500">Loading...</div>
        )}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">{error}</div>
        )}

        {!loading && !error && (
          <div className="space-y-4">
            {/* KPIs */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <KpiCard title="Total Models" value={kpis.totalModels} />
              <KpiCard title="Total Predictions" value={kpis.totalPredictions} />
              <KpiCard title="Average Accuracy" value={kpis.averageAccuracy} />
              <KpiCard title="Best Model" value={kpis.bestModel} />
            </div>

            {/* Performance Chart */}
            <PerformanceChart data={stats?.model_performance} />

            {/* Recent Predictions */}
            <RecentPredictions items={stats?.recent_predictions} />
          </div>
        )}
      </div>
    </div>
  );
}


