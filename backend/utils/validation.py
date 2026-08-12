import os
from config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_UPLOAD_BYTES

# Magic-byte signatures for quick file-type sanity checks.
_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",  # WEBP: RIFF....WEBP, checked further below
}


class ValidationError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def secure_filename(filename: str) -> str:
    """Minimal, dependency-free safe-filename sanitizer."""
    keep = "-_. "
    cleaned = "".join(c for c in filename if c.isalnum() or c in keep).strip()
    cleaned = cleaned.replace(" ", "_")
    return cleaned[-180:] if cleaned else "upload"


def validate_extension(filename: str) -> str:
    if "." not in filename:
        raise ValidationError("File has no extension.", 415)
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            415,
        )
    return ext


def validate_mimetype(mimetype: str):
    """Client-supplied MIME types (from the browser/HTTP client) are easily
    wrong or generic (e.g. 'application/octet-stream') and are not a
    trustworthy signal on their own. We only use this as a soft check, and
    rely on validate_signature() (real magic-byte inspection of the file
    contents) as the authoritative check."""
    generic_types = {"application/octet-stream", "", None}
    if mimetype in generic_types:
        return
    if mimetype not in ALLOWED_MIME_TYPES:
        raise ValidationError(f"Unsupported MIME type '{mimetype}'.", 415)


def validate_size(data: bytes):
    if len(data) == 0:
        raise ValidationError("Uploaded file is empty.", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"File exceeds the maximum allowed size ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            413,
        )


def validate_signature(data: bytes):
    """Best-effort magic-byte check. Not a substitute for full decoding,
    but catches obviously mislabeled/non-image files early."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ValidationError("File signature does not match a supported image format.", 415)
