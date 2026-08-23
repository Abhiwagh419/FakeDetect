"""JPEG-specific forensic inspection and Error Level Analysis (ELA).

ELA is explicitly documented as a visualization aid, not proof of
manipulation, per the project's engineering principles.
"""
import io
import numpy as np
from PIL import Image, ImageChops

from services.model_service import image_to_base64


def jpeg_forensics(img: Image.Image, pil_format: str, original_bytes: bytes):
    info = {
        "applicable": pil_format == "JPEG",
        "format": pil_format,
    }
    if pil_format != "JPEG":
        info["note"] = "JPEG-specific forensic checks are not applicable to this format."
        return info

    info["progressive_encoding"] = bool(img.info.get("progressive", False))
    quantization = getattr(img, "quantization", None)
    if quantization:
        # Rough proxy for compression quality: mean of the luminance
        # quantization table (lower generally implies less aggressive compression).
        luma_table = quantization.get(0)
        info["quantization_tables_present"] = True
        info["luma_quant_mean"] = round(float(np.mean(luma_table)), 2) if luma_table else None
    else:
        info["quantization_tables_present"] = False

    info["file_size_bytes"] = len(original_bytes)
    info["note"] = (
        "These are technical characteristics of the JPEG encoding and are not, "
        "by themselves, indicators of AI generation or manipulation."
    )
    return info


def error_level_analysis(img: Image.Image, quality: int = 90):
    """Generates an ELA visualization: recompresses the image at a known
    JPEG quality and amplifies the per-pixel difference from the original.
    Returned as a base64 PNG plus an explanatory note."""
    rgb_img = img.convert("RGB")

    buf = io.BytesIO()
    rgb_img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    diff = ImageChops.difference(rgb_img, recompressed)
    diff_np = np.array(diff).astype(np.float32)

    max_diff = diff_np.max()
    scale = 255.0 / max_diff if max_diff > 0 else 1.0
    amplified = np.clip(diff_np * scale, 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(amplified)

    return {
        "image_b64": image_to_base64(ela_img),
        "recompression_quality": quality,
        "max_pixel_difference": float(max_diff),
        "note": (
            "Error Level Analysis visualization. ELA is a forensic visualization "
            "technique and may be affected by normal JPEG recompression and "
            "image-processing operations. It is not proof of manipulation."
        ),
    }
