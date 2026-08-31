import { useCallback, useRef, useState } from "react";
import { UploadCloud, X, ImageIcon } from "lucide-react";

const ALLOWED_EXT = ["jpg", "jpeg", "png", "webp"];
const MAX_MB = 15;

export default function UploadZone({ file, preview, onFileSelected, onClear, disabled, verdict }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState(null);

  const validateAndSet = useCallback(
    (selected) => {
      setLocalError(null);
      if (!selected) return;
      const ext = selected.name.split(".").pop()?.toLowerCase();
      if (!ALLOWED_EXT.includes(ext)) {
        setLocalError(`Unsupported format ".${ext}". Use JPG, PNG, or WEBP.`);
        return;
      }
      if (selected.size > MAX_MB * 1024 * 1024) {
        setLocalError(`File exceeds ${MAX_MB} MB.`);
        return;
      }
      onFileSelected(selected);
    },
    [onFileSelected]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const dropped = e.dataTransfer.files?.[0];
      validateAndSet(dropped);
    },
    [validateAndSet, disabled]
  );

  if (file && preview) {
    const lockClass = verdict === "REAL" ? "lock-real" : verdict === "FAKE" ? "lock-fake" : "is-idle";
    return (
      <div>
        <div className={`scan-frame border border-black/[0.08] bg-black/[0.03] ${lockClass}`}>
          <span className="corner-tl" />
          <span className="corner-br" />
          <img src={preview} alt="Selected upload preview" className="w-full max-h-[420px] object-contain" />
        </div>
        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)] font-mono">
            <ImageIcon size={14} />
            <span className="truncate max-w-[280px]">{file.name}</span>
            <span>· {(file.size / 1024).toFixed(0)} KB</span>
          </div>
          {!disabled && (
            <button
              onClick={onClear}
              className="flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-[var(--fake)] transition-colors"
            >
              <X size={14} /> Clear
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image for analysis"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-xl py-16 px-6 cursor-pointer transition-colors ${
          dragOver ? "border-[var(--accent)] bg-[var(--accent-dim)]" : "border-black/[0.08] hover:border-black/[0.16]"
        }`}
      >
        <div className="w-12 h-12 rounded-full bg-[var(--accent-dim)] flex items-center justify-center">
          <UploadCloud size={22} className="text-[var(--accent-strong)]" />
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-[var(--text)]">Drop an image, or click to browse</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">JPG, PNG, or WEBP · up to {MAX_MB} MB</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          className="hidden"
          onChange={(e) => validateAndSet(e.target.files?.[0])}
        />
      </div>
      {localError && <p className="text-xs text-[var(--fake)] mt-2">{localError}</p>}
    </div>
  );
}
