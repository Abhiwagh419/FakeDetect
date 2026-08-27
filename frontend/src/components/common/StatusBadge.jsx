const VARIANTS = {
  fake: "bg-[var(--fake-dim)] text-[var(--fake)] border-[var(--fake)]/30",
  real: "bg-[var(--real-dim)] text-[var(--real)] border-[var(--real)]/30",
  amber: "bg-[var(--amber-dim)] text-[var(--amber)] border-[var(--amber)]/30",
  neutral: "bg-black/[0.03] text-[var(--text-muted)] border-black/[0.08]",
  accent: "bg-[var(--accent-dim)] text-[var(--accent-strong)] border-[var(--accent)]/30",
};

export default function StatusBadge({ variant = "neutral", children, className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-semibold tracking-wide font-mono ${VARIANTS[variant]} ${className}`}
    >
      {children}
    </span>
  );
}
