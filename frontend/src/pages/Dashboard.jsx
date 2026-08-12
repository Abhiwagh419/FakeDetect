import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, PieChart, Pie, Cell } from "recharts";
import { Activity, ShieldAlert, ShieldCheck, Gauge, ScanSearch } from "lucide-react";
import Card from "../components/common/Card";
import StatusBadge from "../components/common/StatusBadge";
import Spinner from "../components/common/Spinner";
import { getDashboard } from "../services/api";

function StatCard({ label, value, icon: Icon, accent }) {
  return (
    <div className="glass-panel p-5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-[var(--text-muted)]">{label}</span>
        <Icon size={16} className={accent} />
      </div>
      <p className="font-display font-bold text-2xl">{value}</p>
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getDashboard()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner size={24} className="text-[var(--accent)]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel p-6 text-center">
        <p className="text-sm text-[var(--fake)]">{error}</p>
      </div>
    );
  }

  const pieData = [
    { name: "FAKE", value: data.fake_detected, color: "#fb4570" },
    { name: "REAL", value: data.real_detected, color: "#34d399" },
  ].filter((d) => d.value > 0);

  const timelineByDay = {};
  for (const t of data.timeline) {
    timelineByDay[t.day] = timelineByDay[t.day] || { day: t.day, FAKE: 0, REAL: 0 };
    timelineByDay[t.day][t.prediction] = t.c;
  }
  const timelineData = Object.values(timelineByDay);

  return (
    <div>
      <header className="mb-6">
        <h1 className="font-display font-bold text-2xl">Dashboard</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">Overview of all detection activity.</p>
      </header>

      {data.total_analyses === 0 ? (
        <div className="glass-panel flex flex-col items-center justify-center py-24 text-center">
          <ScanSearch size={32} className="text-[var(--text-faint)] mb-3" />
          <p className="text-sm text-[var(--text-muted)] mb-4">No analyses yet. Run your first detection to populate the dashboard.</p>
          <Link
            to="/analyze"
            className="px-4 py-2 rounded-lg bg-[var(--accent)] text-black text-sm font-semibold hover:bg-[var(--accent-strong)] transition-colors"
          >
            Go to Analyze
          </Link>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <StatCard label="Total Analyses" value={data.total_analyses} icon={Activity} accent="text-[var(--accent-strong)]" />
            <StatCard label="FAKE Detected" value={data.fake_detected} icon={ShieldAlert} accent="text-[var(--fake)]" />
            <StatCard label="REAL Detected" value={data.real_detected} icon={ShieldCheck} accent="text-[var(--real)]" />
            <StatCard label="Avg. Model Probability" value={`${data.average_model_probability}%`} icon={Gauge} accent="text-[var(--amber)]" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-6">
            <Card title="FAKE vs REAL" className="lg:col-span-1">
              <ResponsiveContainer width="100%" height={180} debounce={1}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={3}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#0d1220", border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8, fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Analyses Over Time" className="lg:col-span-2">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={timelineData}>
                  <XAxis dataKey="day" tick={{ fontSize: 10, fill: "#8993a8" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#8993a8" }} axisLine={false} tickLine={false} allowDecimals={false} />
                  <Tooltip contentStyle={{ background: "#0d1220", border: "1px solid rgba(148,163,184,0.2)", borderRadius: 8, fontSize: 12 }} />
                  <Bar dataKey="FAKE" stackId="a" fill="#fb4570" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="REAL" stackId="a" fill="#34d399" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Recent Analyses">
            <div className="divide-y divide-white/[0.06]">
              {data.recent.map((item) => (
                <div key={item.analysis_id} className="flex items-center justify-between py-3">
                  <div>
                    <p className="text-sm text-[var(--text)]">{item.filename}</p>
                    <p className="text-[11px] text-[var(--text-faint)] font-mono mt-0.5">
                      {new Date(item.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-[var(--text-muted)]">
                      {Math.max(item.fake_probability, item.real_probability).toFixed(2)}%
                    </span>
                    <StatusBadge variant={item.prediction === "FAKE" ? "fake" : "real"}>{item.prediction}</StatusBadge>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
