import Card from "../common/Card";
import StatusBadge from "../common/StatusBadge";
import ProbabilityBar from "../common/ProbabilityBar";
import { ScanSearch } from "lucide-react";

const TIER_LABEL = {
  HIGH_MODEL_CONFIDENCE: "High Model Confidence",
  MODERATE_MODEL_CONFIDENCE: "Moderate Model Confidence",
  LOWER_MODEL_CONFIDENCE: "Lower Model Confidence",
};

export default function DetectionResult({ result }) {
  const isFake = result.prediction === "FAKE";

  return (
    <Card title="AI Detection Result" subtitle="Model prediction & output probability" icon={ScanSearch}>
      <div className="flex items-center justify-between mb-5">
        <div>
          <p className="text-[11px] text-[var(--text-muted)] uppercase tracking-wide font-mono mb-1">Prediction</p>
          <p className={`font-display font-bold text-2xl ${isFake ? "text-[var(--fake)]" : "text-[var(--real)]"}`}>
            {isFake ? "AI-GENERATED" : "REAL PHOTOGRAPH"}
          </p>
        </div>
        <StatusBadge variant={isFake ? "fake" : "real"}>{result.prediction}</StatusBadge>
      </div>

      <div className="space-y-3 mb-5">
        <ProbabilityBar label="FAKE" value={result.fake_probability} variant="fake" />
        <ProbabilityBar label="REAL" value={result.real_probability} variant="real" />
      </div>

      <div className="flex items-center justify-between px-3.5 py-2.5 rounded-lg bg-black/[0.02] border border-black/[0.06]">
        <span className="text-xs text-[var(--text-muted)]">Confidence Tier</span>
        <StatusBadge variant="accent">{TIER_LABEL[result.model_confidence_tier]}</StatusBadge>
      </div>

      <p className="text-[11px] text-[var(--text-faint)] mt-4 leading-relaxed">
        This is the model's softmax output probability for each class, interpreted into a UI
        confidence tier by threshold — not a calibrated real-world certainty.
      </p>
    </Card>
  );
}
