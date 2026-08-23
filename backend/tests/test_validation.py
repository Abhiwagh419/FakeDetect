import pytest

from utils.validation import (
    ValidationError,
    secure_filename,
    validate_extension,
    validate_mimetype,
    validate_size,
    validate_signature,
)


class TestSecureFilename:
    def test_strips_path_separators_and_specials(self):
        assert secure_filename("../../etc/passwd") == "etcpasswd"

    def test_collapses_dot_dot_sequences(self):
        # Defensive: the sanitized name must never contain a literal ".."
        # even though this value isn't currently used to build a filesystem
        # path (see docstring in validation.py).
        result = secure_filename("../../etc/passwd")
        assert ".." not in result

    def test_replaces_spaces_with_underscore(self):
        assert secure_filename("my photo.jpg") == "my_photo.jpg"

    def test_empty_or_fully_stripped_falls_back(self):
        assert secure_filename("") == "upload"
        assert secure_filename("***???") == "upload"

    def test_truncates_very_long_names(self):
        name = ("a" * 300) + ".jpg"
        result = secure_filename(name)
        assert len(result) <= 180


class TestValidateExtension:
    @pytest.mark.parametrize("filename", ["photo.jpg", "photo.jpeg", "photo.PNG", "photo.webp"])
    def test_allowed_extensions_pass(self, filename):
        validate_extension(filename)  # should not raise

    def test_no_extension_rejected(self):
        with pytest.raises(ValidationError) as exc:
            validate_extension("noextension")
        assert exc.value.status_code == 415

    def test_disallowed_extension_rejected(self):
        with pytest.raises(ValidationError) as exc:
            validate_extension("malware.exe")
        assert exc.value.status_code == 415
        assert "exe" in exc.value.message


class TestValidateMimetype:
    def test_correct_mimetype_passes(self):
        validate_mimetype("image/jpeg")

    def test_generic_octet_stream_passes_as_soft_check(self):
        validate_mimetype("application/octet-stream")

    def test_empty_or_none_passes_as_soft_check(self):
        validate_mimetype("")
        validate_mimetype(None)

    def test_wrong_mimetype_rejected(self):
        with pytest.raises(ValidationError):
            validate_mimetype("application/pdf")


class TestValidateSize:
    def test_empty_file_rejected(self):
        with pytest.raises(ValidationError) as exc:
            validate_size(b"")
        assert exc.value.status_code == 400

    def test_oversized_file_rejected(self):
        from config import MAX_UPLOAD_BYTES
        with pytest.raises(ValidationError) as exc:
            validate_size(b"x" * (MAX_UPLOAD_BYTES + 1))
        assert exc.value.status_code == 413

    def test_normal_size_passes(self, png_bytes):
        validate_size(png_bytes)


class TestValidateSignature:
    def test_jpeg_signature_detected(self, jpeg_bytes):
        assert validate_signature(jpeg_bytes) == "jpeg"

    def test_png_signature_detected(self, png_bytes):
        assert validate_signature(png_bytes) == "png"

    def test_webp_signature_detected(self, webp_bytes):
        assert validate_signature(webp_bytes) == "webp"

    def test_spoofed_executable_rejected(self):
        """An .exe renamed to look like an upload must still fail on
        magic bytes, even if extension/MIME were spoofed upstream."""
        fake = b"MZ\x90\x00\x03\x00\x00\x00this is not an image"
        with pytest.raises(ValidationError) as exc:
            validate_signature(fake)
        assert exc.value.status_code == 415

    def test_truncated_file_rejected(self):
        with pytest.raises(ValidationError):
            validate_signature(b"\xff\xd8")  # too short to match full JPEG marker context safely
