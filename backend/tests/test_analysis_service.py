import os

import pytest

from services import analysis_service


@pytest.fixture()
def patched_artifacts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_service, "ARTIFACTS_DIR", str(tmp_path))
    return tmp_path


def test_full_analysis_on_png_has_all_expected_top_level_keys(png_bytes, patched_artifacts_dir):
    result = analysis_service.run_full_analysis(png_bytes, "sample.png")

    expected_keys = {
        "analysis_id", "filename", "timestamp", "prediction", "confidence",
        "fake_probability", "real_probability", "model_confidence_tier",
        "domain_normalized", "image", "statistics", "histogram", "metadata",
        "hash", "jpeg_forensics", "ela", "gradcam", "model", "artifacts",
    }
    assert expected_keys.issubset(result.keys())
    assert result["prediction"] in ("FAKE", "REAL")
    assert result["ela"] is None  # PNG has no ELA


def test_full_analysis_on_jpeg_includes_ela(jpeg_bytes, patched_artifacts_dir):
    result = analysis_service.run_full_analysis(jpeg_bytes, "sample.jpg")
    assert result["ela"] is not None
    assert result["jpeg_forensics"]["applicable"] is True


def test_fake_and_real_probabilities_sum_to_100(png_bytes, patched_artifacts_dir):
    result = analysis_service.run_full_analysis(png_bytes, "sample.png")
    total = result["fake_probability"] + result["real_probability"]
    assert abs(total - 100.0) < 0.5


def test_confidence_tier_matches_confidence_value(png_bytes, patched_artifacts_dir):
    result = analysis_service.run_full_analysis(png_bytes, "sample.png")
    conf = result["confidence"]
    tier = result["model_confidence_tier"]
    if conf >= 90:
        assert tier == "HIGH_MODEL_CONFIDENCE"
    elif conf >= 70:
        assert tier == "MODERATE_MODEL_CONFIDENCE"
    else:
        assert tier == "LOWER_MODEL_CONFIDENCE"


def test_artifacts_are_persisted_to_disk(png_bytes, patched_artifacts_dir):
    result = analysis_service.run_full_analysis(png_bytes, "sample.png")
    assert os.path.exists(result["artifacts"]["original_path"])
    assert os.path.exists(result["artifacts"]["gradcam_path"])


def test_each_analysis_gets_a_unique_id(png_bytes, patched_artifacts_dir):
    r1 = analysis_service.run_full_analysis(png_bytes, "a.png")
    r2 = analysis_service.run_full_analysis(png_bytes, "b.png")
    assert r1["analysis_id"] != r2["analysis_id"]


def test_hash_reflects_actual_uploaded_bytes(png_bytes, patched_artifacts_dir):
    from utils.hashing import sha256_bytes
    result = analysis_service.run_full_analysis(png_bytes, "sample.png")
    assert result["hash"]["sha256"] == sha256_bytes(png_bytes)
