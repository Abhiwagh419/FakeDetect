import { useCallback, useEffect, useState } from "react";
import { Search, Trash2, FileDown, Eye, X } from "lucide-react";
import StatusBadge from "../components/common/StatusBadge";
import Spinner from "../components/common/Spinner";
import { getHistory, deleteAnalysis, getAnalysis, downloadReportById, triggerDownload } from "../services/api";
import DetectionResult from "../components/analyze/DetectionResult";
import ImageDetailsCard from "../components/analyze/ImageDetailsCard";
import StatisticsCard from "../components/analyze/StatisticsCard";
import MetadataCard from "../components/analyze/MetadataCard";
import GradCamCard from "../components/analyze/GradCamCard";

export default function History() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    getHistory({ search: search || undefined, prediction: filter || undefined })
      .then(setItems)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [search, filter]);

  useEffect(() => {
    const t = setTimeout(load, 250);
    return () => clearTimeout(t);
  }, [load]);

  const handleDelete = async (id) => {
    if (!confirm(`Delete analysis ${id}? This cannot be undone.`)) return;
    try {
      await deleteAnalysis(id);
      setItems((prev) => prev.filter((i) => i.analysis_id !== id));
      if (selected?.analysis_id === id) setSelected(null);
    } catch (err) {
      alert(err.message);
    }
  };

  const handleView = async (id) => {
    setDetailLoading(true);
    try {
      const full = await getAnalysis(id);
      setSelected(full);
    } catch (err) {
      alert(err.message);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDownloadReport = async (id) => {
    try {
      const blob = await downloadReportById(id);
      triggerDownload(blob, `${id}.pdf`);
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <div>
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="font-display font-bold text-2xl">Analysis History</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">Every detection run, stored locally.</p>
        </div>
      </header>

      <div className="glass-panel p-4 mb-5 flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)]" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by filename…"
            className="w-full bg-white/[0.03] border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm outline-none focus:border-[var(--accent)]/50"
          />
        </div>
        <div className="flex gap-2">
          {["", "FAKE", "REAL"].map((f) => (
            <button
              key={f || "all"}
              onClick={() => setFilter(f)}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-colors ${
                filter === f ? "bg-[var(--accent-dim)] text-[var(--accent-strong)]" : "text-[var(--text-muted)] hover:bg-white/5"
              }`}
            >
              {f || "All"}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-16">
          <Spinner size={22} className="text-[var(--accent)]" />
        </div>
      ) : error ? (
        <p className="text-sm text-[var(--fake)] text-center py-10">{error}</p>
      ) : items.length === 0 ? (
        <div className="glass-panel py-16 text-center">
          <p className="text-sm text-[var(--text-muted)]">No matching analyses found.</p>
        </div>
      ) : (
        <div className="glass-panel overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/[0.08] text-left">
                <th className="px-4 py-3 text-xs font-medium text-[var(--text-muted)]">Date</th>
                <th className="px-4 py-3 text-xs font-medium text-[var(--text-muted)]">File</th>
                <th className="px-4 py-3 text-xs font-medium text-[var(--text-muted)]">Result</th>
                <th className="px-4 py-3 text-xs font-medium text-[var(--text-muted)]">Probability</th>
                <th className="px-4 py-3 text-xs font-medium text-[var(--text-muted)] text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.analysis_id} className="border-b border-white/[0.05] last:border-0 hover:bg-white/[0.02]">
                  <td className="px-4 py-3 text-xs font-mono text-[var(--text-muted)]">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 max-w-[220px] truncate">{item.filename}</td>
                  <td className="px-4 py-3">
                    <StatusBadge variant={item.prediction === "FAKE" ? "fake" : "real"}>{item.prediction}</StatusBadge>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs">
                    {Math.max(item.fake_probability, item.real_probability).toFixed(2)}%
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <button onClick={() => handleView(item.analysis_id)} className="p-1.5 rounded-md hover:bg-white/5 text-[var(--text-muted)] hover:text-[var(--text)]" aria-label="View analysis">
                        <Eye size={14} />
                      </button>
                      <button onClick={() => handleDownloadReport(item.analysis_id)} className="p-1.5 rounded-md hover:bg-white/5 text-[var(--text-muted)] hover:text-[var(--text)]" aria-label="Download report">
                        <FileDown size={14} />
                      </button>
                      <button onClick={() => handleDelete(item.analysis_id)} className="p-1.5 rounded-md hover:bg-white/5 text-[var(--text-muted)] hover:text-[var(--fake)]" aria-label="Delete analysis">
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {(selected || detailLoading) && (
        <div className="fixed inset-0 bg-black/70 z-40 flex items-center justify-center p-6" onClick={() => setSelected(null)}>
          <div
            className="glass-panel bg-[var(--panel-solid)] max-w-4xl w-full max-h-[85vh] overflow-y-auto p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-display font-bold text-lg">Analysis Detail</h2>
              <button onClick={() => setSelected(null)} className="p-1.5 rounded-md hover:bg-white/5">
                <X size={18} />
              </button>
            </div>
            {detailLoading ? (
              <div className="flex justify-center py-16">
                <Spinner size={22} className="text-[var(--accent)]" />
              </div>
            ) : (
              selected && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <DetectionResult result={selected} />
                  <ImageDetailsCard image={selected.image} hash={selected.hash} />
                  <StatisticsCard statistics={selected.statistics} />
                  <MetadataCard metadata={selected.metadata} />
                  <div className="md:col-span-2">
                    <GradCamCard gradcam={selected.gradcam} />
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}
