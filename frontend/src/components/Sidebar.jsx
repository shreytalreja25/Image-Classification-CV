import { NavLink } from "react-router-dom";
import { Activity, BarChart3, Brain, Cog, Gauge, Image as ImageIcon, LayoutDashboard } from "lucide-react";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/models", label: "Models", icon: Brain },
  { to: "/test", label: "Test Model", icon: ImageIcon },
  { to: "/predictions", label: "Predictions", icon: BarChart3 },
  { to: "/training", label: "Training", icon: Activity },
  { to: "/settings", label: "Settings", icon: Cog },
];

export default function Sidebar({ open, onNavigate }) {
  return (
    <aside className={`${open ? "translate-x-0" : "-translate-x-full"} fixed inset-y-0 z-40 w-64 transform border-r border-slate-200 bg-white p-4 shadow-lg transition-transform duration-200 ease-in-out md:static md:translate-x-0`}>
      <div className="mb-6 flex items-center gap-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-indigo-600 text-white">AI</div>
        <div className="text-lg font-semibold text-slate-800">Aerial Lab</div>
      </div>
      <nav className="space-y-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium ${isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-50"}`}
            onClick={onNavigate}
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="mt-8 rounded-md bg-slate-50 p-3 text-xs text-slate-500">
        <div className="font-medium text-slate-700">Color Guide</div>
        <div>Primary: Indigo 600</div>
        <div>Background: Slate 50</div>
        <div>Text: Slate 800/600</div>
        <div>Accent: Emerald 500</div>
      </div>
    </aside>
  );
}


