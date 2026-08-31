import { useCallback, useEffect, useState } from "react";
import { ScanSearch, AlertTriangle, FlaskConical } from "lucide-react";
import UploadZone from "../components/analyze/UploadZone";
import DetectionResult from "../components/analyze/DetectionResult";
import ImageDetailsCard from "../components/analyze/ImageDetailsCard";
import StatisticsCard from "../components/analyze/StatisticsCard";
import HistogramCard from "../components/analyze/HistogramCard";
import MetadataCard from "../components/analyze/MetadataCard";
import GradCamCard from "../components/analyze/GradCamCard";
import ForensicsCard from "../components/analyze/ForensicsCard";
import ForensicSummaryCard from "../components/analyze/ForensicSummaryCard";
import Spinner from "../components/common/Spinner";
import { analyzeImage } from "../services/api";

export default function Analyze() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [normalizeDomain, setNormalizeDomain] = useState(true);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFileSelected = useCallback((selected) => {
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeImage(file, { normalizeDomain });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <header className="mb-6">
        <h1 className="font-display font-bold text-2xl">Analyze</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Upload an image for AI-detection and digital forensic analysis.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[420px_1fr] gap-6 items-start">
        <div className="glass-panel p-5">
          <UploadZone
            file={file}
            preview={preview}
            onFileSelected={handleFileSelected}
            onClear={handleClear}
            disabled={loading}
            verdict={result?.prediction}
          />

          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="w-full mt-4 flex items-center justify-center gap-2 py-3 rounded-lg bg-[var(--accent)] text-white font-semibold text-sm hover:bg-[var(--accent-strong)] disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 enabled:hover:-translate-y-0.5 enabled:hover:shadow-[var(--shadow-md)]"
          >
            {loading ? <Spinner size={16} /> : <ScanSearch size={16} />}
            {loading ? "Analyzing..." : "Run Detection"}
          </button>

          <label className="flex items-start gap-2.5 mt-4 p-3 rounded-lg bg-[var(--amber-dim)] border border-[var(--amber)]/20 cursor-pointer">
            <input
              type="checkbox"
              checked={normalizeDomain}
              onChange={(e) => setNormalizeDomain(e.target.checked)}
              disabled={loading}
              className="mt-0.5 accent-[var(--amber)]"
            />
            <span className="text-xs text-[var(--text-muted)] leading-relaxed">
              <span className="inline-flex items-center gap-1 font-semibold text-[var(--amber)]">
                <FlaskConical size={12} /> Experimental: resolution-normalized inference
              </span>
              <br />
              CIFAKE's training images are 32×32px, so the model treats natural
              sharpness/detail in modern phone photos as a FAKE signal. This
              option routes the image through the same low-resolution
              bottleneck before classification to reduce that bias — at the
              cost of some sensitivity on already-ambiguous images. It does
              not make the model a general-purpose modern AI-image detector.
            </span>
          </label>

          {error && (
            <div className="flex items-start gap-2 mt-3 p-3 rounded-lg bg-[var(--fake-dim)] border border-[var(--fake)]/20">
              <AlertTriangle size={15} className="text-[var(--fake)] shrink-0 mt-0.5" />
              <p className="text-xs text-[var(--fake)]">{error}</p>
            </div>
          )}
        </div>

        <div className="min-h-[300px]">
          {!result && !loading && (
            <div className="glass-panel h-full flex flex-col items-center justify-center py-20 text-center">
              <ScanSearch size={32} className="text-[var(--text-faint)] mb-3" />
              <p className="text-sm text-[var(--text-muted)]">Select an image to begin analysis.</p>
            </div>
          )}

          {loading && (
            <div className="glass-panel h-full flex flex-col items-center justify-center py-20 relative overflow-hidden">
              <div className="scan-sweep" />
              <Spinner size={28} className="text-[var(--accent)] mb-3" />
              <p className="text-sm text-[var(--text-muted)] font-mono">Running inference &amp; forensic analysis…</p>
            </div>
          )}

          {result && !loading && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="stagger-in lift-on-hover" style={{ "--stagger-index": 0 }}><DetectionResult result={result} /></div>
              <div className="stagger-in lift-on-hover" style={{ "--stagger-index": 1 }}><ForensicSummaryCard result={result} /></div>
              <div className="stagger-in lift-on-hover" style={{ "--stagger-index": 2 }}><ImageDetailsCard image={result.image} hash={result.hash} /></div>
              <div className="stagger-in lift-on-hover" style={{ "--stagger-index": 3 }}><StatisticsCard statistics={result.statistics} /></div>
              <div className="stagger-in lift-on-hover" style={{ "--stagger-index": 4 }}><HistogramCard histogram={result.histogram} /></div>
              <div className="stagger-in lift-on-hover" style={{ "--stagger-index": 5 }}><MetadataCard metadata={result.metadata} /></div>
              <div className="md:col-span-2 stagger-in lift-on-hover" style={{ "--stagger-index": 6 }}>
                <GradCamCard gradcam={result.gradcam} preview={preview} />
              </div>
              <div className="md:col-span-2 stagger-in lift-on-hover" style={{ "--stagger-index": 7 }}>
                <ForensicsCard jpegForensics={result.jpeg_forensics} ela={result.ela} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
