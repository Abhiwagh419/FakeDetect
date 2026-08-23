import Card from "../common/Card";
import KeyValueRow from "../common/KeyValueRow";
import { Layers } from "lucide-react";

export default function ForensicsCard({ jpegForensics, ela }) {
  return (
    <Card title="Compression & Error Level Analysis" icon={Layers}>
      {jpegForensics.applicable ? (
        <>
          <KeyValueRow label="Progressive Encoding" value={jpegForensics.progressive_encoding ? "Yes" : "No"} />
          <KeyValueRow
            label="Quantization Tables"
            value={jpegForensics.quantization_tables_present ? "Present" : "Not detected"}
          />
          {jpegForensics.luma_quant_mean != null && (
            <KeyValueRow label="Luma Quant. Mean" value={jpegForensics.luma_quant_mean} mono />
          )}
        </>
      ) : (
        <p className="text-sm text-[var(--text-muted)] py-1">{jpegForensics.note}</p>
      )}

      {ela && (
        <div className="mt-4 pt-4 border-t border-white/[0.06]">
          <p className="text-[10px] uppercase tracking-wide text-[var(--text-faint)] mb-1.5 font-mono">
            ELA Visualization
          </p>
          <img
            src={`data:image/png;base64,${ela.image_b64}`}
            alt="Error Level Analysis visualization"
            className="w-full rounded-lg border border-white/10"
          />
        </div>
      )}

      <p className="text-[11px] text-[var(--text-faint)] mt-3 leading-relaxed">
        {ela?.note || jpegForensics.note}
      </p>
    </Card>
  );
}
