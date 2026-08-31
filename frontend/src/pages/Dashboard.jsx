import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, PieChart, Pie, Cell } from "recharts";
import { Activity, ShieldAlert, ShieldCheck, Gauge, ScanSearch } from "lucide-react";
import Card from "../components/common/Card";
import StatusBadge from "../components/common/StatusBadge";
import Spinner from "../components/common/Spinner";
import AnimatedNumber from "../components/common/AnimatedNumber";
import { getDashboard } from "../services/api";

function StatCard({ label, value, decimals = 0, suffix = "", icon: Icon, accent, index = 0 }) {
  return (
    <div className="glass-panel p-5 stagger-in lift-on-hover" style={{ "--stagger-index": index }}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-[var(--text-muted)]">{label}</span>
        <Icon size={16} className={accent} />
      </div>
      <p className="font-display font-bold text-2xl">
        <AnimatedNumber value={value} decimals={decimals} suffix={suffix} />
      </p>
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
    { name: "FAKE", value: data.fake_detected, color: "#DC2A3F" },
    { name: "REAL", value: data.real_detected, color: "#0F9D58" },
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
        <div className="glass-panel flex flex-col items-center justify-center py-24 text-center stagger-in">
          <ScanSearch size={32} className="text-[var(--text-faint)] mb-3" />
          <p className="text-sm text-[var(--text-muted)] mb-4">No analyses yet. Run your first detection to populate the dashboard.</p>
          <Link
            to="/analyze"
            className="px-4 py-2 rounded-lg bg-[var(--accent)] text-white text-sm font-semibold hover:bg-[var(--accent-strong)] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-[var(--shadow-md)]"
          >
            Go to Analyze
          </Link>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <StatCard index={0} label="Total Analyses" value={data.total_analyses} icon={Activity} accent="text-[var(--accent-strong)]" />
            <StatCard index={1} label="FAKE Detected" value={data.fake_detected} icon={ShieldAlert} accent="text-[var(--fake)]" />
            <StatCard index={2} label="REAL Detected" value={data.real_detected} icon={ShieldCheck} accent="text-[var(--real)]" />
            <StatCard index={3} label="Avg. Model Probability" value={data.average_model_probability} decimals={1} suffix="%" icon={Gauge} accent="text-[var(--amber)]" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-6">
            <Card title="FAKE vs REAL" className="lg:col-span-1 stagger-in lift-on-hover" style={{ "--stagger-index": 4 }}>
              <ResponsiveContainer width="100%" height={180} debounce={1}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={3}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid rgba(15,23,42,0.12)", borderRadius: 8, fontSize: 12, boxShadow: "0 4px 16px -4px rgba(16,24,40,0.15)" }} />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Analyses Over Time" className="lg:col-span-2 stagger-in lift-on-hover" style={{ "--stagger-index": 5 }}>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={timelineData}>
                  <XAxis dataKey="day" tick={{ fontSize: 10, fill: "#5B6577" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#5B6577" }} axisLine={false} tickLine={false} allowDecimals={false} />
                  <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid rgba(15,23,42,0.12)", borderRadius: 8, fontSize: 12, boxShadow: "0 4px 16px -4px rgba(16,24,40,0.15)" }} />
                  <Bar dataKey="FAKE" stackId="a" fill="#DC2A3F" radius={[3, 3, 0, 0]} animationDuration={700} />
                  <Bar dataKey="REAL" stackId="a" fill="#0F9D58" radius={[3, 3, 0, 0]} animationDuration={700} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Recent Analyses" className="stagger-in" style={{ "--stagger-index": 6 }}>
            <div className="divide-y divide-black/[0.06]">
              {data.recent.map((item) => (
                <div key={item.analysis_id} className="flex items-center justify-between py-3 px-2 -mx-2 rounded-lg transition-colors duration-150 hover:bg-black/[0.02]">
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
