"""Exercises the actual trained checkpoint. Slower than the other service
tests (loads a real ResNet18 + does real forward/backward passes) but this
is the core of the project — it deserves a real, unmocked test."""
import io

import pytest
from PIL import Image

from services import model_service


@pytest.fixture(scope="module")
def sample_image():
    return Image.new("RGB", (300, 300), (90, 130, 180))


def test_classes_loaded_from_checkpoint():
    assert set(model_service.CLASSES) == {"FAKE", "REAL"}


def test_predict_returns_well_formed_result(sample_image):
    result = model_service.predict(sample_image, normalize_domain=False)

    assert result["prediction"] in model_service.CLASSES
    assert 0.0 <= result["confidence"] <= 100.0
    assert set(result["probabilities"].keys()) == set(model_service.CLASSES)
    assert result["domain_normalized"] is False


def test_probabilities_sum_to_100(sample_image):
    result = model_service.predict(sample_image)
    total = sum(result["probabilities"].values())
    assert abs(total - 100.0) < 0.5  # rounding tolerance


def test_confidence_matches_predicted_class_probability(sample_image):
    result = model_service.predict(sample_image)
    assert result["confidence"] == result["probabilities"][result["prediction"]]


def test_heatmap_is_valid_base64_png(sample_image):
    result = model_service.predict(sample_image)
    import base64
    decoded = base64.b64decode(result["heatmap_b64"])
    img = Image.open(io.BytesIO(decoded))
    assert img.format == "PNG"
    assert img.size == (model_service.IMG_SIZE, model_service.IMG_SIZE) if hasattr(model_service, "IMG_SIZE") else True


def test_domain_normalize_flag_is_recorded(sample_image):
    result = model_service.predict(sample_image, normalize_domain=True)
    assert result["domain_normalized"] is True


def test_domain_normalize_downsamples_and_upsamples_correctly(sample_image):
    normalized = model_service.domain_normalize(sample_image)
    assert normalized.size == (model_service.IMG_SIZE, model_service.IMG_SIZE)


def test_predict_is_deterministic_in_eval_mode(sample_image):
    """Same input, same weights, eval mode (no dropout/batchnorm updates)
    -> repeated calls must agree, otherwise something is train()-mode or
    non-deterministic in a way that would make demo results flaky."""
    r1 = model_service.predict(sample_image)
    r2 = model_service.predict(sample_image)
    assert r1["prediction"] == r2["prediction"]
    assert abs(r1["confidence"] - r2["confidence"]) < 0.01
