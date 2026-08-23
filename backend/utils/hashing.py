import hashlib
import secrets
from datetime import datetime, timezone


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def generate_analysis_id() -> str:
    """FD-YYYYMMDD-XXXXXXXX (date-stamped + 8 random hex chars)."""
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    rand_part = secrets.token_hex(4).upper()
    return f"FD-{date_part}-{rand_part}"
