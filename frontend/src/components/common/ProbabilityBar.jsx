export default function ProbabilityBar({ label, value, variant }) {
  const color = variant === "fake" ? "var(--fake)" : "var(--real)";
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-semibold font-mono tracking-wide" style={{ color }}>
          {label}
        </span>
        <span className="text-xs font-mono text-[var(--text-muted)]">{value.toFixed(2)}%</span>
      </div>
      <div className="progress-track h-2">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${value}%`, background: color }}
        />
      </div>
    </div>
  );
}
