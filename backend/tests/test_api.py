import io

import pytest


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert set(data["classes"]) == {"FAKE", "REAL"}


def test_model_info_endpoint(client):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["architecture"]
    assert "confusion_matrix" in data
    assert "classification_report" in data


def test_analyze_with_no_file_returns_400(client):
    res = client.post("/api/analyze", data={})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_analyze_with_bad_extension_returns_415(client):
    data = {"image": (io.BytesIO(b"not an image"), "malware.exe")}
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 415


def test_analyze_with_spoofed_signature_returns_415(client):
    """A file named photo.jpg whose bytes don't match a real JPEG magic
    number must still be rejected — extension alone is not trusted."""
    fake_bytes = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 100
    data = {"image": (io.BytesIO(fake_bytes), "photo.jpg")}
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 415


def test_analyze_with_valid_png_returns_full_result(client, png_bytes):
    data = {"image": (io.BytesIO(png_bytes), "sample.png")}
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200

    body = res.get_json()
    assert body["prediction"] in ("FAKE", "REAL")
    assert "artifacts" not in body  # internal disk paths must never leak to the client
    assert "analysis_id" in body
    assert "gradcam" in body and "image_b64" in body["gradcam"]


def test_analyze_with_valid_jpeg_includes_ela(client, jpeg_bytes):
    data = {"image": (io.BytesIO(jpeg_bytes), "sample.jpg")}
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["ela"] is not None


def test_analyze_then_appears_in_history(client, png_bytes):
    data = {"image": (io.BytesIO(png_bytes), "trackable.png")}
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    analysis_id = res.get_json()["analysis_id"]

    hist_res = client.get("/api/history")
    ids = [item["analysis_id"] for item in hist_res.get_json()]
    assert analysis_id in ids


def test_history_item_not_found_returns_404(client):
    res = client.get("/api/history/FD-00000000-DEADBEEF")
    assert res.status_code == 404


def test_delete_history_item(client, png_bytes):
    data = {"image": (io.BytesIO(png_bytes), "todelete.png")}
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    analysis_id = res.get_json()["analysis_id"]

    del_res = client.delete(f"/api/history/{analysis_id}")
    assert del_res.status_code == 200
    assert del_res.get_json()["deleted"] is True

    get_res = client.get(f"/api/history/{analysis_id}")
    assert get_res.status_code == 404


def test_delete_nonexistent_history_item_returns_404(client):
    res = client.delete("/api/history/FD-00000000-DEADBEEF")
    assert res.status_code == 404


def test_dashboard_endpoint_reflects_activity(client, png_bytes):
    before = client.get("/api/dashboard").get_json()["total_analyses"]

    data = {"image": (io.BytesIO(png_bytes), "dash.png")}
    client.post("/api/analyze", data=data, content_type="multipart/form-data")

    after = client.get("/api/dashboard").get_json()["total_analyses"]
    assert after == before + 1


def test_report_generation_from_fresh_upload_returns_pdf(client, large_jpeg_bytes):
    data = {"image": (io.BytesIO(large_jpeg_bytes), "report_test.jpg")}
    res = client.post("/api/report", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.mimetype == "application/pdf"
    assert res.data[:4] == b"%PDF"


def test_report_generation_from_existing_analysis_id(client, png_bytes):
    data = {"image": (io.BytesIO(png_bytes), "for_report.png")}
    analyze_res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    analysis_id = analyze_res.get_json()["analysis_id"]

    report_res = client.post("/api/report", data={"analysis_id": analysis_id})
    assert report_res.status_code == 200
    assert report_res.data[:4] == b"%PDF"


def test_report_for_unknown_analysis_id_returns_404(client):
    res = client.post("/api/report", data={"analysis_id": "FD-00000000-DEADBEEF"})
    assert res.status_code == 404


def test_get_report_by_id_regenerates_if_missing(client, png_bytes):
    data = {"image": (io.BytesIO(png_bytes), "regen.png")}
    analyze_res = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    analysis_id = analyze_res.get_json()["analysis_id"]

    res = client.get(f"/api/report/{analysis_id}")
    assert res.status_code == 200
    assert res.data[:4] == b"%PDF"


def test_legacy_predict_endpoint_still_works(client, png_bytes):
    """Backward-compat route referenced in the README — must keep working."""
    data = {"image": (io.BytesIO(png_bytes), "legacy.png")}
    res = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    body = res.get_json()
    assert body["prediction"] in ("FAKE", "REAL")
    assert "heatmap" in body
