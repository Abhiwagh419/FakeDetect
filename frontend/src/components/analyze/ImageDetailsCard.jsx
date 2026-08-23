import Card from "../common/Card";
import KeyValueRow from "../common/KeyValueRow";
import { Image as ImageIcon } from "lucide-react";

export default function ImageDetailsCard({ image, hash }) {
  return (
    <Card title="Image Information" icon={ImageIcon}>
      <KeyValueRow label="Filename" value={image.filename} />
      <KeyValueRow label="Format" value={image.format} />
      <KeyValueRow label="MIME Type" value={image.mime_type} mono />
      <KeyValueRow label="File Size" value={image.file_size_readable} />
      <KeyValueRow label="Resolution" value={`${image.width} × ${image.height}`} />
      <KeyValueRow label="Aspect Ratio" value={image.aspect_ratio_readable} />
      <KeyValueRow label="Megapixels" value={`${image.megapixels} MP`} />
      <KeyValueRow label="Color Mode" value={image.color_mode} />
      <KeyValueRow label="Channels" value={image.channels} />
      {image.progressive_jpeg !== null && image.progressive_jpeg !== undefined && (
        <KeyValueRow label="Progressive JPEG" value={image.progressive_jpeg ? "Yes" : "No"} />
      )}
      <KeyValueRow label="SHA-256" value={hash.sha256} mono />
    </Card>
  );
}
