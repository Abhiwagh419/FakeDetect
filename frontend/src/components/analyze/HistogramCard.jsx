import { AreaChart, Area, XAxis, ResponsiveContainer, Tooltip } from "recharts";
import Card from "../common/Card";
import { BarChart3 } from "lucide-react";

export default function HistogramCard({ histogram }) {
  const data = histogram.r.map((_, i) => ({
    bin: i,
    r: histogram.r[i],
    g: histogram.g[i],
    b: histogram.b[i],
  }));

  return (
    <Card title="Color Histogram" subtitle="RGB channel distribution" icon={BarChart3}>
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
          <XAxis dataKey="bin" hide />
          <Tooltip
            contentStyle={{ background: "#ffffff", border: "1px solid rgba(15,23,42,0.12)", borderRadius: 8, fontSize: 11, boxShadow: "0 4px 16px -4px rgba(16,24,40,0.15)" }}
            labelFormatter={() => ""}
          />
          <Area type="monotone" dataKey="r" stroke="#DC2626" fill="#DC2626" fillOpacity={0.12} strokeWidth={1.5} />
          <Area type="monotone" dataKey="g" stroke="#16A34A" fill="#16A34A" fillOpacity={0.12} strokeWidth={1.5} />
          <Area type="monotone" dataKey="b" stroke="#2563EB" fill="#2563EB" fillOpacity={0.12} strokeWidth={1.5} />
        </AreaChart>
      </ResponsiveContainer>

      <div className="flex items-center gap-2 mt-3">
        <span className="text-xs text-[var(--text-muted)] mr-1">Dominant colors</span>
        {histogram.dominant_colors.map((c) => (
          <div
            key={c.hex}
            title={`${c.hex} · ${c.percent}%`}
            className="w-6 h-6 rounded border border-black/[0.08]"
            style={{ background: c.hex }}
          />
        ))}
      </div>
    </Card>
  );
}
