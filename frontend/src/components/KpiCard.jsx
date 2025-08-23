export default function KpiCard({ title, value, subtitle }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="text-sm text-gray-500">{title}</div>
      <div className="mt-1 text-3xl font-semibold text-gray-900">{value}</div>
      {subtitle ? (
        <div className="mt-1 text-xs text-gray-400">{subtitle}</div>
      ) : null}
    </div>
  );
}


