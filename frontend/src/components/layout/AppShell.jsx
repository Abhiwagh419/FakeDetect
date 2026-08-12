import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard, ScanSearch, History as HistoryIcon,
  FileText, Cpu, ShieldCheck, Menu, X,
} from "lucide-react";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/analyze", label: "Analyze", icon: ScanSearch },
  { to: "/history", label: "History", icon: HistoryIcon },
  { to: "/reports", label: "Reports", icon: FileText },
  { to: "/model", label: "Model", icon: Cpu },
];

export default function AppShell({ children }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="min-h-screen flex">
      {/* Mobile top bar */}
      <div className="lg:hidden fixed top-0 inset-x-0 h-14 border-b hairline bg-[var(--ink)]/95 backdrop-blur z-30 flex items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-[var(--accent-dim)] border border-[var(--accent)]/25 flex items-center justify-center">
            <ShieldCheck size={14} className="text-[var(--accent-strong)]" />
          </div>
          <span className="font-display font-bold text-sm tracking-tight">FAKEDETECT</span>
        </div>
        <button
          onClick={() => setOpen(true)}
          className="p-2 rounded-lg hover:bg-white/5 text-[var(--text)]"
          aria-label="Open navigation menu"
        >
          <Menu size={20} />
        </button>
      </div>

      {/* Mobile drawer backdrop */}
      {open && (
        <div
          className="lg:hidden fixed inset-0 bg-black/60 z-40"
          onClick={() => setOpen(false)}
        />
      )}

      <aside
        className={`w-64 shrink-0 border-r hairline flex flex-col fixed h-screen bg-[var(--ink)] z-50 transition-transform duration-200 ${
          open ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0`}
      >
        <div className="px-5 py-6 border-b hairline flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-[var(--accent-dim)] border border-[var(--accent)]/25 flex items-center justify-center">
              <ShieldCheck size={18} className="text-[var(--accent-strong)]" />
            </div>
            <div>
              <div className="font-display font-bold text-[15px] tracking-tight leading-none">FAKEDETECT</div>
              <div className="text-[10px] text-[var(--text-muted)] mt-1 tracking-wide uppercase font-mono">
                AI Image Forensics
              </div>
            </div>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="lg:hidden p-1.5 rounded-md hover:bg-white/5 text-[var(--text-muted)]"
            aria-label="Close navigation menu"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-[var(--accent-dim)] text-[var(--accent-strong)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/[0.04]"
                }`
              }
            >
              <Icon size={17} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t hairline">
          <p className="text-[11px] text-[var(--text-faint)] leading-relaxed">
            ResNet18 · CIFAKE dataset<br />
            Local inference, no external API calls.
          </p>
        </div>
      </aside>

      <div className="flex-1 lg:ml-64 min-h-screen flex flex-col pt-14 lg:pt-0">
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1400px] w-full mx-auto">{children}</main>
      </div>
    </div>
  );
}
