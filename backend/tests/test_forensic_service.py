from PIL import Image

from services.forensic_service import error_level_analysis, jpeg_forensics


def test_jpeg_forensics_not_applicable_for_png(png_bytes):
    img = Image.open(__import__("io").BytesIO(png_bytes))
    result = jpeg_forensics(img, "PNG", png_bytes)
    assert result["applicable"] is False
    assert "note" in result


def test_jpeg_forensics_applicable_for_jpeg(jpeg_bytes):
    img = Image.open(__import__("io").BytesIO(jpeg_bytes))
    result = jpeg_forensics(img, "JPEG", jpeg_bytes)
    assert result["applicable"] is True
    assert result["quantization_tables_present"] is True
    assert result["luma_quant_mean"] is not None
    assert result["file_size_bytes"] == len(jpeg_bytes)


def test_error_level_analysis_returns_valid_image_and_metadata(jpeg_bytes):
    img = Image.open(__import__("io").BytesIO(jpeg_bytes)).convert("RGB")
    result = error_level_analysis(img)

    assert result["recompression_quality"] == 90
    assert result["max_pixel_difference"] >= 0
    assert isinstance(result["image_b64"], str) and len(result["image_b64"]) > 0

    # Round-trip the base64 back into a real image to prove it's valid PNG,
    # not just a non-empty string.
    import base64
    import io as _io
    decoded = base64.b64decode(result["image_b64"])
    out_img = Image.open(_io.BytesIO(decoded))
    assert out_img.format == "PNG"


def test_ela_on_flat_image_has_low_max_difference():
    # A perfectly flat, already-heavily-compressed-friendly image should
    # show very little recompression error.
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    result = error_level_analysis(img, quality=90)
    assert result["max_pixel_difference"] < 50
