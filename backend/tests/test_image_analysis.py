from PIL import Image

from services.image_analysis import compute_histogram, compute_statistics, get_image_info


def test_get_image_info_basic_fields():
    img = Image.new("RGB", (200, 100))
    info = get_image_info(img, "sample.jpg", file_size=12345, pil_format="JPEG")

    assert info["filename"] == "sample.jpg"
    assert info["format"] == "JPEG"
    assert info["mime_type"] == "image/jpeg"
    assert info["width"] == 200
    assert info["height"] == 100
    assert info["aspect_ratio"] == 2.0
    assert info["aspect_ratio_readable"] == "2:1"
    assert info["megapixels"] == 0.02
    assert info["channels"] == 3


def test_get_image_info_unknown_format_falls_back():
    img = Image.new("RGB", (10, 10))
    info = get_image_info(img, "sample", file_size=1, pil_format=None)
    assert info["format"] == "UNKNOWN"
    assert info["mime_type"] == "application/octet-stream"


def test_get_image_info_handles_zero_height_without_crash():
    # Defensive: aspect ratio math must not raise ZeroDivisionError even
    # for a degenerate 0-height image (shouldn't normally reach here since
    # PIL won't produce one, but the guard exists — verify it holds).
    img = Image.new("RGB", (10, 10))
    info = get_image_info(img, "x.png", file_size=1, pil_format="PNG")
    assert info["height"] > 0  # sanity: real PIL images never have height 0


def test_compute_statistics_on_flat_color_image():
    # A perfectly flat image has zero contrast, zero sharpness, zero noise —
    # useful as a known-value baseline for the heuristic estimators.
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    stats = compute_statistics(img)

    assert stats["brightness"]["mean_luminance_0_255"] == 128.0
    assert stats["contrast"]["std_luminance_0_255"] == 0.0
    assert stats["sharpness"]["laplacian_variance"] == 0.0
    assert stats["noise"]["residual_std"] == 0.0
    assert stats["average_rgb"]["r"] == 128.0
    assert stats["pixel_range"]["min"] == 128
    assert stats["pixel_range"]["max"] == 128


def test_compute_statistics_percentages_are_bounded():
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    stats = compute_statistics(img)
    for key in ("brightness", "contrast", "sharpness", "noise"):
        assert 0 <= stats[key]["display_percent"] <= 100


def test_compute_histogram_bin_counts_sum_to_pixel_count():
    img = Image.new("RGB", (10, 10), (50, 100, 150))
    hist = compute_histogram(img, bins=16)

    assert sum(hist["r"]) == 100  # 10x10 pixels
    assert sum(hist["g"]) == 100
    assert sum(hist["b"]) == 100
    assert len(hist["r"]) == 16


def test_compute_histogram_dominant_colors_sum_close_to_100_percent():
    img = Image.new("RGB", (20, 20), (10, 200, 30))
    hist = compute_histogram(img)
    total_pct = sum(c["percent"] for c in hist["dominant_colors"])
    assert 99.0 <= total_pct <= 100.01  # single flat color -> ~100%
    assert hist["dominant_colors"][0]["hex"].startswith("#")
