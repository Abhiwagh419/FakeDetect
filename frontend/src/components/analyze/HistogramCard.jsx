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
            contentStyle={{ background: "#0d1220", border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8, fontSize: 11 }}
            labelFormatter={() => ""}
          />
          <Area type="monotone" dataKey="r" stroke="#fb4570" fill="#fb4570" fillOpacity={0.15} strokeWidth={1.5} />
          <Area type="monotone" dataKey="g" stroke="#34d399" fill="#34d399" fillOpacity={0.15} strokeWidth={1.5} />
          <Area type="monotone" dataKey="b" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.15} strokeWidth={1.5} />
        </AreaChart>
      </ResponsiveContainer>

      <div className="flex items-center gap-2 mt-3">
        <span className="text-xs text-[var(--text-muted)] mr-1">Dominant colors</span>
        {histogram.dominant_colors.map((c) => (
          <div
            key={c.hex}
            title={`${c.hex} · ${c.percent}%`}
            className="w-6 h-6 rounded border border-white/10"
            style={{ background: c.hex }}
          />
        ))}
      </div>
    </Card>
  );
}
