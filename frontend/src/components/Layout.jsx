import { useState } from "react";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";

export default function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-800">
      <Sidebar open={sidebarOpen} onNavigate={() => setSidebarOpen(false)} />
      <div className="flex w-full flex-col md:pl-64">
        <Topbar onToggleSidebar={() => setSidebarOpen((v) => !v)} />
        <main className="mx-auto w-full max-w-7xl p-4">
          {children}
        </main>
      </div>
    </div>
  );
}


