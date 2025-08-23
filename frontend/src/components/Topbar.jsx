import { Menu } from "lucide-react";

export default function Topbar({ onToggleSidebar }) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
        <button className="inline-flex items-center gap-2 rounded-md p-2 text-slate-600 hover:bg-slate-100 md:hidden" onClick={onToggleSidebar} aria-label="Toggle sidebar">
          <Menu className="h-5 w-5" />
        </button>
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span className="hidden sm:inline">Aerial Classification Dashboard</span>
        </div>
        <div className="text-xs text-slate-400">v1.0</div>
      </div>
    </header>
  );
}


