export default function KeyValueRow({ label, value, mono = false, className = "" }) {
  return (
    <div className={`flex items-start justify-between gap-3 py-2 border-b border-white/[0.06] last:border-0 ${className}`}>
      <span className="text-xs text-[var(--text-muted)] shrink-0 pt-0.5">{label}</span>
      <span className={`text-sm text-[var(--text)] text-right break-all ${mono ? "font-mono text-xs" : ""}`}>
        {value ?? "—"}
      </span>
    </div>
  );
}
