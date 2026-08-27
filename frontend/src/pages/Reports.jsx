import { useEffect, useState } from "react";
import { FileText, FileDown, CheckCircle2, Circle } from "lucide-react";
import Card from "../components/common/Card";
import StatusBadge from "../components/common/StatusBadge";
import Spinner from "../components/common/Spinner";
import { getHistory, downloadReportById, triggerDownload } from "../services/api";

export default function Reports() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloadingId, setDownloadingId] = useState(null);

  useEffect(() => {
    getHistory()
      .then(setItems)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const handleDownload = async (id) => {
    setDownloadingId(id);
    try {
      const blob = await downloadReportById(id);
      triggerDownload(blob, `${id}.pdf`);
      setItems((prev) => prev.map((i) => (i.analysis_id === id ? { ...i, report_available: true } : i)));
    } catch (err) {
      alert(err.message);
    } finally {
      setDownloadingId(null);
    }
  };

  return (
    <div>
      <header className="mb-6">
        <h1 className="font-display font-bold text-2xl">Reports</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Generate or re-download professional PDF forensic reports for any past analysis.
        </p>
      </header>

      {loading ? (
        <div className="flex justify-center py-16">
          <Spinner size={22} className="text-[var(--accent)]" />
        </div>
      ) : error ? (
        <p className="text-sm text-[var(--fake)] text-center py-10">{error}</p>
      ) : items.length === 0 ? (
        <div className="glass-panel py-16 text-center">
          <FileText size={28} className="text-[var(--text-faint)] mx-auto mb-3" />
          <p className="text-sm text-[var(--text-muted)]">No analyses to report on yet.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {items.map((item) => (
            <Card key={item.analysis_id} className="flex flex-col">
              <div className="flex items-start justify-between mb-3">
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{item.filename}</p>
                  <p className="text-[11px] text-[var(--text-faint)] font-mono mt-0.5">{item.analysis_id}</p>
                </div>
                <StatusBadge variant={item.prediction === "FAKE" ? "fake" : "real"}>{item.prediction}</StatusBadge>
              </div>

              <div className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] mb-4">
                {item.report_available ? (
                  <CheckCircle2 size={13} className="text-[var(--real)]" />
                ) : (
                  <Circle size={13} className="text-[var(--text-faint)]" />
                )}
                {item.report_available ? "Report generated" : "Not yet generated"}
              </div>

              <button
                onClick={() => handleDownload(item.analysis_id)}
                disabled={downloadingId === item.analysis_id}
                className="mt-auto flex items-center justify-center gap-2 py-2 rounded-lg bg-[var(--accent-dim)] text-[var(--accent-strong)] text-xs font-semibold hover:bg-[var(--accent)] hover:text-white transition-colors disabled:opacity-50"
              >
                {downloadingId === item.analysis_id ? <Spinner size={13} /> : <FileDown size={13} />}
                {item.report_available ? "Download Report" : "Generate & Download"}
              </button>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
