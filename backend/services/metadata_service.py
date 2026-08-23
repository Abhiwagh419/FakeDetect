"""Extracts EXIF metadata that actually exists in the uploaded file.
Absence of metadata is reported as absence — never interpreted as a
detection signal by itself."""
from PIL import Image, ExifTags

_TAG_NAMES = {v: k for k, v in ExifTags.TAGS.items()}

_INTERESTING_FIELDS = [
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "Orientation", "FocalLength", "ISOSpeedRatings", "ExposureTime",
    "FNumber", "LensModel", "LensMake", "WhiteBalance", "Flash",
]


def extract_exif(img: Image.Image):
    raw = img.getexif()
    fields = {}

    if raw:
        for tag_id, value in raw.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            fields[tag_name] = _stringify(value)

    # Also check the more complete IFD-based EXIF (covers fields like
    # FocalLength/ISO/ExposureTime that live in the Exif sub-IFD, and GPS
    # fields that live in the GPS sub-IFD). The GPS IFD uses a separate
    # tag-numbering scheme (ExifTags.GPSTAGS), not the main ExifTags.TAGS
    # table, so it must be looked up separately or GPS fields render as
    # raw numeric keys (e.g. "2") instead of names (e.g. "GPSLatitude").
    try:
        exif_data = img.getexif()
        for ifd_id, tag_table in ((0x8769, ExifTags.TAGS), (0x8825, ExifTags.GPSTAGS)):
            try:
                ifd = exif_data.get_ifd(ifd_id)
            except Exception:
                ifd = {}
            for tag_id, value in ifd.items():
                tag_name = tag_table.get(tag_id, str(tag_id))
                fields[tag_name] = _stringify(value)
    except Exception:
        pass

    gps_present = any(k.lower().startswith("gps") for k in fields)

    highlighted = {f: fields[f] for f in _INTERESTING_FIELDS if f in fields}

    if not fields:
        availability = "NOT_FOUND"
    elif len(fields) <= 2:
        availability = "LIMITED"
    else:
        availability = "AVAILABLE"

    return {
        "availability": availability,
        "gps_present": gps_present,
        "field_count": len(fields),
        "highlighted_fields": highlighted,
        "all_fields": fields,
        "note": (
            "Metadata availability alone is not sufficient to determine "
            "image authenticity."
        ),
    }


def _stringify(value):
    if isinstance(value, bytes):
        try:
            return value.decode(errors="replace")
        except Exception:
            return repr(value)
    if isinstance(value, (tuple, list)):
        return [_stringify(v) for v in value]
    return str(value)
