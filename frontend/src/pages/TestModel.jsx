import { useEffect, useMemo, useState } from "react";
import { getModels, getRandomTestImage, predict } from "../services/api";
import { API_BASE_URL } from "../config/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function TestModel() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [img, setImg] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModels().then((m) => { setModels(m || []); setModelName(m?.[0] || ""); });
  }, []);

  const loadRandom = async () => {
    setError(null);
    setResult(null);
    try {
      const data = await getRandomTestImage({ random: true });
      const url = `${API_BASE_URL}/images/${encodeURIComponent(data.category)}/${encodeURIComponent(data.filename)}`;
      setImg({ url, category: data.category, filename: data.filename });
    } catch (e) {
      setError(e?.message || "Failed to load image");
    }
  };

  const runPredict = async () => {
    if (!modelName) return;
    setPredicting(true);
    setError(null);
    try {
      const data = await predict({ model_name: modelName, image_url: img?.url });
      setResult(data);
    } catch (e) {
      setError(e?.message || "Prediction failed");
    } finally {
      setPredicting(false);
    }
  };

  const chartData = useMemo(() => {
    if (!result?.all_predictions) return [];
    return Object.entries(result.all_predictions).map(([k, v]) => ({ label: k, value: Math.round((v || 0) * 10000) / 100 }));
  }, [result]);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Test Model</h2>
        <p className="text-sm text-slate-500">Pick a model, load a sample image, and run prediction</p>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <label className="text-sm text-slate-600">Model</label>
            <select value={modelName} onChange={(e) => setModelName(e.target.value)} className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500">
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
            <div className="flex gap-2 pt-2">
              <button onClick={loadRandom} className="rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-700 hover:bg-slate-50">Load Random Image</button>
              <button onClick={runPredict} disabled={predicting} className="rounded-md bg-indigo-600 px-3 py-2 text-sm text-white hover:bg-indigo-700 disabled:opacity-60">{predicting ? "Predicting..." : "Predict"}</button>
            </div>
            {error && <div className="text-sm text-red-600">{error}</div>}
            {result && (
              <div className="mt-2 text-sm text-slate-600">
                <div><span className="font-medium">Predicted:</span> {result.predicted_class}</div>
                <div><span className="font-medium">Confidence:</span> {(result.confidence * 100).toFixed?.(2)}%</div>
                <div><span className="font-medium">Time:</span> {result.processing_time?.toFixed?.(2)}s</div>
              </div>
            )}
          </div>

          <div className="md:col-span-1">
            <div className="aspect-video overflow-hidden rounded-md border border-slate-200 bg-slate-100">
              {img ? (
                <img src={img.url} alt="test" className="h-full w-full object-contain" />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-400">No image loaded</div>
              )}
            </div>
          </div>

          <div className="md:col-span-1">
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" angle={-20} textAnchor="end" height={60} />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="value" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


