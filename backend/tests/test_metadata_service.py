from PIL import Image

from services.metadata_service import extract_exif


def test_image_with_no_exif_reports_not_found():
    img = Image.new("RGB", (10, 10))
    meta = extract_exif(img)
    assert meta["availability"] == "NOT_FOUND"
    assert meta["field_count"] == 0
    assert meta["gps_present"] is False
    assert meta["highlighted_fields"] == {}


def test_image_with_basic_exif_is_detected():
    img = Image.new("RGB", (10, 10))
    exif = img.getexif()
    exif[0x0110] = "Test Camera"  # Model
    exif[0x010F] = "Test Make"    # Make
    meta = extract_exif(img)

    assert meta["field_count"] >= 2
    assert meta["highlighted_fields"].get("Model") == "Test Camera"
    assert meta["highlighted_fields"].get("Make") == "Test Make"
    assert meta["availability"] in ("LIMITED", "AVAILABLE")


def test_gps_ifd_tags_are_named_not_numeric():
    """Regression test: GPS sub-IFD tags must resolve through GPSTAGS, not
    the main EXIF TAGS table, or they render as raw numeric keys like "2"
    instead of "GPSLatitude"."""
    img = Image.new("RGB", (10, 10))
    exif = img.getexif()
    gps_ifd = exif.get_ifd(0x8825)
    gps_ifd[1] = "N"                       # GPSLatitudeRef
    gps_ifd[2] = (19.0, 4.0, 30.0)          # GPSLatitude
    exif[0x8825] = gps_ifd

    meta = extract_exif(img)

    assert meta["gps_present"] is True
    assert "GPSLatitude" in meta["all_fields"]
    assert "GPSLatitudeRef" in meta["all_fields"]
    # The old buggy behavior would have stored these under "1" and "2".
    assert "1" not in meta["all_fields"]
    assert "2" not in meta["all_fields"]


def test_note_is_always_present():
    img = Image.new("RGB", (10, 10))
    meta = extract_exif(img)
    assert "not sufficient" in meta["note"].lower()
