import { useState } from "react";
import Card from "../common/Card";
import StatusBadge from "../common/StatusBadge";
import KeyValueRow from "../common/KeyValueRow";
import Spinner from "../common/Spinner";
import { ClipboardList, FileDown, Download } from "lucide-react";
import { generateReport, triggerDownload } from "../../services/api";

export default function ForensicSummaryCard({ result }) {
  const [generating, setGenerating] = useState(false);
  const [reportError, setReportError] = useState(null);

  const handleGenerateReport = async () => {
    setGenerating(true);
    setReportError(null);
    try {
      const blob = await generateReport({ analysisId: result.analysis_id });
      triggerDownload(blob, `${result.analysis_id}.pdf`);
    } catch (err) {
      setReportError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  const handleDownloadJson = () => {
    const payload = { ...result };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    triggerDownload(blob, `${result.analysis_id}.json`);
  };

  const isFake = result.prediction === "FAKE";

  return (
    <Card title="Forensic Summary" icon={ClipboardList}>
      <KeyValueRow label="AI Detection" value={<StatusBadge variant={isFake ? "fake" : "real"}>{isFake ? "AI-GENERATED" : "REAL"}</StatusBadge>} />
      <KeyValueRow label="Model Probability" value={`${Math.max(result.fake_probability, result.real_probability).toFixed(2)}%`} mono />
      <KeyValueRow label="Metadata" value={result.metadata.availability.replace("_", " ")} />
      <KeyValueRow label="Image Format" value={result.image.format} />
      <KeyValueRow label="Resolution" value={`${result.image.width} × ${result.image.height}`} />
      <KeyValueRow label="Analysis ID" value={result.analysis_id} mono />
      <KeyValueRow label="Grad-CAM" value="Available" />

      <div className="flex flex-col gap-2 mt-4 pt-4 border-t border-white/[0.06]">
        <button
          onClick={handleGenerateReport}
          disabled={generating}
          className="flex items-center justify-center gap-2 py-2.5 rounded-lg bg-[var(--accent)] text-black font-semibold text-sm hover:bg-[var(--accent-strong)] disabled:opacity-60 transition-colors"
        >
          {generating ? <Spinner size={15} /> : <FileDown size={15} />}
          {generating ? "Generating report..." : "Generate PDF Report"}
        </button>
        <button
          onClick={handleDownloadJson}
          className="flex items-center justify-center gap-2 py-2.5 rounded-lg border border-white/10 text-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:border-white/20 transition-colors"
        >
          <Download size={14} /> Download Analysis JSON
        </button>
        {reportError && <p className="text-xs text-[var(--fake)]">{reportError}</p>}
      </div>
    </Card>
  );
}
