import Card from "../common/Card";
import { Activity } from "lucide-react";

function StatBar({ label, pct, note }) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-[var(--text-muted)]">{label}</span>
        <span className="text-xs font-mono text-[var(--text)]">{pct}%</span>
      </div>
      <div className="progress-track h-1.5">
        <div
          className="h-full rounded-full bg-[var(--accent)]"
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <p className="text-[10px] text-[var(--text-faint)] mt-1">{note}</p>
    </div>
  );
}

export default function StatisticsCard({ statistics }) {
  const { brightness, contrast, sharpness, noise, average_rgb } = statistics;
  return (
    <Card title="Image Statistics" subtitle="Computed from actual pixel data" icon={Activity}>
      <StatBar label="Brightness" pct={brightness.display_percent} note={brightness.label} />
      <StatBar label="Contrast" pct={contrast.display_percent} note={contrast.label} />
      <StatBar label="Sharpness" pct={sharpness.display_percent} note={sharpness.label} />
      <StatBar label="Noise" pct={noise.display_percent} note={noise.label} />

      <div className="flex items-center gap-3 mt-4 pt-4 border-t border-black/[0.06]">
        <span className="text-xs text-[var(--text-muted)]">Average RGB</span>
        <div
          className="w-6 h-6 rounded border border-black/[0.08]"
          style={{ background: `rgb(${average_rgb.r}, ${average_rgb.g}, ${average_rgb.b})` }}
        />
        <span className="text-xs font-mono text-[var(--text)]">
          ({average_rgb.r.toFixed(0)}, {average_rgb.g.toFixed(0)}, {average_rgb.b.toFixed(0)})
        </span>
      </div>
    </Card>
  );
}
