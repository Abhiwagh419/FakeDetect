import Card from "../common/Card";
import StatusBadge from "../common/StatusBadge";
import KeyValueRow from "../common/KeyValueRow";
import { FileSearch } from "lucide-react";

const AVAILABILITY_VARIANT = {
  AVAILABLE: "real",
  LIMITED: "amber",
  NOT_FOUND: "neutral",
};

export default function MetadataCard({ metadata }) {
  const entries = Object.entries(metadata.highlighted_fields || {});
  return (
    <Card
      title="Metadata (EXIF)"
      icon={FileSearch}
      action={<StatusBadge variant={AVAILABILITY_VARIANT[metadata.availability]}>{metadata.availability.replace("_", " ")}</StatusBadge>}
    >
      {entries.length === 0 ? (
        <p className="text-sm text-[var(--text-muted)] py-2">No EXIF metadata detected.</p>
      ) : (
        entries.map(([key, value]) => (
          <KeyValueRow key={key} label={key} value={Array.isArray(value) ? value.join(", ") : value} />
        ))
      )}
      <KeyValueRow label="GPS Data" value={metadata.gps_present ? "Present" : "Not present"} />
      <p className="text-[11px] text-[var(--text-faint)] mt-3 leading-relaxed">{metadata.note}</p>
    </Card>
  );
}
