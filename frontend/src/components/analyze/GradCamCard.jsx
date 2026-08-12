import { useState } from "react";
import Card from "../common/Card";
import { Eye, Download, Maximize2 } from "lucide-react";

export default function GradCamCard({ gradcam, preview }) {
  const [enlarged, setEnlarged] = useState(false);
  const heatmapSrc = `data:image/png;base64,${gradcam.image_b64}`;

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = heatmapSrc;
    a.download = "gradcam.png";
    a.click();
  };

  return (
    <Card
      title="Explainable AI — Grad-CAM"
      subtitle="Regions that influenced the model's decision"
      icon={Eye}
      action={
        <div className="flex items-center gap-2">
          <button
            onClick={() => setEnlarged(true)}
            className="p-1.5 rounded-md hover:bg-white/5 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
            aria-label="Enlarge Grad-CAM"
          >
            <Maximize2 size={14} />
          </button>
          <button
            onClick={handleDownload}
            className="p-1.5 rounded-md hover:bg-white/5 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
            aria-label="Download Grad-CAM image"
          >
            <Download size={14} />
          </button>
        </div>
      }
    >
      <div className={preview ? "grid grid-cols-2 gap-3" : ""}>
        {preview && (
          <div>
            <p className="text-[10px] uppercase tracking-wide text-[var(--text-faint)] mb-1.5 font-mono">Original</p>
            <img src={preview} alt="Original upload" className="w-full rounded-lg border border-white/10 aspect-square object-cover" />
          </div>
        )}
        <div>
          <p className="text-[10px] uppercase tracking-wide text-[var(--text-faint)] mb-1.5 font-mono">Grad-CAM</p>
          <img
            src={heatmapSrc}
            alt="Grad-CAM heatmap overlay"
            className="w-full rounded-lg border border-white/10 aspect-square object-cover cursor-zoom-in"
            onClick={() => setEnlarged(true)}
          />
        </div>
      </div>
      <p className="text-[11px] text-[var(--text-faint)] mt-3 leading-relaxed">{gradcam.note}</p>

      {enlarged && (
        <div
          className="fixed inset-0 bg-black/85 z-50 flex items-center justify-center p-8"
          onClick={() => setEnlarged(false)}
          role="dialog"
          aria-modal="true"
        >
          <img src={heatmapSrc} alt="Grad-CAM heatmap enlarged" className="max-w-full max-h-full rounded-lg" />
        </div>
      )}
    </Card>
  );
}
