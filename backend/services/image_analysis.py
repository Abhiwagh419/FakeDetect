"""Computes real image information and statistics from the uploaded file.

Every value here is derived directly from the actual image bytes/pixels —
no placeholder or hardcoded figures.
"""
import numpy as np
from PIL import Image


def get_image_info(img: Image.Image, filename: str, file_size: int, pil_format: str | None):
    width, height = img.size
    aspect = width / height if height else 0
    return {
        "filename": filename,
        "format": pil_format or "UNKNOWN",
        "mime_type": _format_to_mime(pil_format),
        "file_size_bytes": file_size,
        "file_size_readable": _human_size(file_size),
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect, 4),
        "aspect_ratio_readable": _aspect_ratio_readable(width, height),
        "megapixels": round((width * height) / 1_000_000, 2),
        "color_mode": img.mode,
        "channels": len(img.getbands()),
        "progressive_jpeg": bool(getattr(img, "info", {}).get("progressive", False))
        if pil_format == "JPEG" else None,
    }


def _format_to_mime(pil_format):
    mapping = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    return mapping.get(pil_format, "application/octet-stream")


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def _aspect_ratio_readable(width, height):
    from math import gcd
    g = gcd(width, height) or 1
    return f"{width // g}:{height // g}"


def compute_statistics(img: Image.Image):
    """Real, computed statistics from actual pixel data. Values labeled as
    estimates where they are heuristic (sharpness, noise), and as direct
    measurements where they are exact (brightness, contrast, histogram)."""
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)

    # Brightness: mean luminance, 0-255 -> percentage
    brightness_mean = float(gray.mean())
    brightness_pct = round(brightness_mean / 255 * 100, 1)

    # Contrast: standard deviation of luminance, normalized against a
    # theoretical max std of 127.5 for an 8-bit channel.
    contrast_std = float(gray.std())
    contrast_pct = round(min(contrast_std / 127.5 * 100, 100), 1)

    # Sharpness estimate: variance of Laplacian (classic focus-measure heuristic).
    sharpness_variance = _laplacian_variance(gray)
    # Normalize against an empirical reference ceiling for display purposes only.
    sharpness_pct = round(min(sharpness_variance / 1500 * 100, 100), 1)

    # Noise estimate: high-frequency residual after a simple 3x3 mean smoothing.
    noise_estimate = _noise_estimate(gray)
    noise_pct = round(min(noise_estimate / 20 * 100, 100), 1)

    avg_rgb = rgb.reshape(-1, 3).mean(axis=0)

    return {
        "brightness": {
            "mean_luminance_0_255": round(brightness_mean, 2),
            "display_percent": brightness_pct,
            "label": "Direct measurement (mean luminance)",
        },
        "contrast": {
            "std_luminance_0_255": round(contrast_std, 2),
            "display_percent": contrast_pct,
            "label": "Direct measurement (luminance standard deviation)",
        },
        "sharpness": {
            "laplacian_variance": round(sharpness_variance, 2),
            "display_percent": sharpness_pct,
            "label": "Heuristic estimate (Laplacian variance focus measure)",
        },
        "noise": {
            "residual_std": round(noise_estimate, 3),
            "display_percent": noise_pct,
            "label": "Heuristic estimate (high-frequency residual after smoothing)",
        },
        "average_rgb": {
            "r": round(float(avg_rgb[0]), 2),
            "g": round(float(avg_rgb[1]), 2),
            "b": round(float(avg_rgb[2]), 2),
        },
        "pixel_range": {
            "min": int(rgb.min()),
            "max": int(rgb.max()),
        },
    }


def _laplacian_variance(gray: np.ndarray) -> float:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    h, w = gray.shape
    lap = (
        padded[0:h, 1:w + 1] * kernel[0, 1]
        + padded[1:h + 1, 0:w] * kernel[1, 0]
        + padded[1:h + 1, 1:w + 1] * kernel[1, 1]
        + padded[1:h + 1, 2:w + 2] * kernel[1, 2]
        + padded[2:h + 2, 1:w + 1] * kernel[2, 1]
    )
    return float(lap.var())


def _noise_estimate(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    h, w = gray.shape
    smoothed = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            smoothed += padded[dy:dy + h, dx:dx + w]
    smoothed /= 9.0
    residual = gray - smoothed
    return float(residual.std())


def compute_histogram(img: Image.Image, bins: int = 32):
    """Downsampled RGB + luminance histograms suitable for charting."""
    rgb = np.array(img.convert("RGB"))
    gray = np.array(img.convert("L"))

    def hist_for(channel):
        counts, _ = np.histogram(channel, bins=bins, range=(0, 255))
        return counts.tolist()

    return {
        "bins": bins,
        "r": hist_for(rgb[..., 0]),
        "g": hist_for(rgb[..., 1]),
        "b": hist_for(rgb[..., 2]),
        "luminance": hist_for(gray),
        "dominant_colors": _dominant_colors(rgb),
    }


def _dominant_colors(rgb: np.ndarray, k: int = 5):
    """Cheap dominant-color extraction via coarse color quantization
    (no external clustering dependency required)."""
    quantized = (rgb // 32 * 32).reshape(-1, 3)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)[:k]
    total = counts.sum()
    result = []
    for idx in order:
        c = colors[idx]
        result.append({
            "hex": "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])),
            "percent": round(float(counts[idx]) / total * 100, 2),
        })
    return result
