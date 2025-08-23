import { useEffect, useState } from "react";
import { getModels } from "../services/api";
import { Brain } from "lucide-react";
import { Link } from "react-router-dom";

export default function Models() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    getModels()
      .then((data) => { if (active) setModels(data || []); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Models</h2>
        <p className="text-sm text-slate-500">Available models served by backend</p>
      </div>

      {loading ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4">Loading...</div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {models.map((m) => (
            <div key={m} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="rounded-md bg-indigo-50 p-2 text-indigo-700"><Brain className="h-5 w-5" /></div>
                <div className="font-medium">{m}</div>
              </div>
              <div className="mt-3 flex gap-2">
                <Link to="/test" className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm text-white hover:bg-indigo-700">Test</Link>
                <Link to="/" className="rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50">View Metrics</Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


