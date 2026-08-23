"""Orchestrates a full FakeDetect analysis: model prediction + Grad-CAM,
image info/statistics, EXIF metadata, hashing, and JPEG/ELA forensics.
Persists the original image + Grad-CAM as artifacts on disk (referenced by
path) so PDF reports can be regenerated later without re-uploading."""
import os
from datetime import datetime, timezone

from PIL import Image

from config import MODEL_INFO, ARTIFACTS_DIR
from services import model_service, image_analysis, metadata_service, forensic_service
from utils.hashing import sha256_bytes, md5_bytes, generate_analysis_id


def run_full_analysis(file_bytes: bytes, filename: str, normalize_domain: bool = False):
    analysis_id = generate_analysis_id()

    original_img = Image.open(_bytes_io(file_bytes))
    pil_format = original_img.format
    original_img.load()  # force decode before any lazy-loaded metadata is lost
    rgb_img = original_img.convert("RGB")

    # --- Model prediction + Grad-CAM (existing working pipeline) ---
    pred = model_service.predict(rgb_img, normalize_domain=normalize_domain)

    # --- Image info + statistics ---
    info = image_analysis.get_image_info(original_img, filename, len(file_bytes), pil_format)
    stats = image_analysis.compute_statistics(rgb_img)
    histogram = image_analysis.compute_histogram(rgb_img)

    # --- Metadata (EXIF) ---
    exif = metadata_service.extract_exif(original_img)

    # --- Hashing ---
    sha256 = sha256_bytes(file_bytes)
    md5 = md5_bytes(file_bytes)

    # --- JPEG / ELA forensics ---
    jpeg_info = forensic_service.jpeg_forensics(original_img, pil_format, file_bytes)
    ela = None
    if pil_format == "JPEG":
        ela = forensic_service.error_level_analysis(rgb_img)

    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Persist artifacts for later report regeneration ---
    artifact_dir = os.path.join(ARTIFACTS_DIR, analysis_id)
    os.makedirs(artifact_dir, exist_ok=True)
    original_path = os.path.join(artifact_dir, f"original.png")
    gradcam_path = os.path.join(artifact_dir, f"gradcam.png")
    rgb_img.save(original_path, format="PNG")
    pred["heatmap_img"].save(gradcam_path, format="PNG")

    fake_prob = pred["probabilities"].get("FAKE", 0.0)
    real_prob = pred["probabilities"].get("REAL", 0.0)

    result = {
        "analysis_id": analysis_id,
        "filename": filename,
        "timestamp": timestamp,
        "prediction": pred["prediction"],
        "confidence": pred["confidence"],
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "model_confidence_tier": _confidence_tier(pred["confidence"]),
        "domain_normalized": normalize_domain,
        "image": info,
        "statistics": stats,
        "histogram": histogram,
        "metadata": exif,
        "hash": {"sha256": sha256, "md5": md5},
        "jpeg_forensics": jpeg_info,
        "ela": ela,
        "gradcam": {
            "image_b64": pred["heatmap_b64"],
            "note": (
                "Grad-CAM highlights spatial regions that contributed more strongly "
                "to the CNN's decision. It is an interpretability aid and is not "
                "independent proof of image manipulation."
            ),
        },
        "model": {
            "architecture": MODEL_INFO["architecture"],
            "dataset": MODEL_INFO["dataset"],
            "test_accuracy": MODEL_INFO["test_accuracy"],
            "best_validation_accuracy": MODEL_INFO["best_val_accuracy"],
        },
        "artifacts": {
            "original_path": original_path,
            "gradcam_path": gradcam_path,
        },
    }
    return result


def _confidence_tier(confidence_pct: float) -> str:
    if confidence_pct >= 90:
        return "HIGH_MODEL_CONFIDENCE"
    if confidence_pct >= 70:
        return "MODERATE_MODEL_CONFIDENCE"
    return "LOWER_MODEL_CONFIDENCE"


def _bytes_io(data: bytes):
    import io
    return io.BytesIO(data)
