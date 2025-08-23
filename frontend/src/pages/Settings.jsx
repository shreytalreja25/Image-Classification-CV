import { API_BASE_URL, WS_URL } from "../config/api";

export default function Settings() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Settings</h2>
        <p className="text-sm text-slate-500">Environment and service endpoints</p>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">Backend URL</dt>
            <dd className="text-sm text-slate-800">{API_BASE_URL}</dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">WebSocket URL</dt>
            <dd className="text-sm text-slate-800">{WS_URL}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}


