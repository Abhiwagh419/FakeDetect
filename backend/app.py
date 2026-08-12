import os
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

from config import MAX_UPLOAD_BYTES, MODEL_INFO, REPORTS_DIR
from utils.validation import (
    ValidationError, validate_extension, validate_mimetype,
    validate_size, validate_signature, secure_filename,
)
from services import model_service, history_service, analysis_service, report_service

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app)

history_service.init_db()


# ---------------------------------------------------------------------------
# Error handlers — never leak raw tracebacks to the client
# ---------------------------------------------------------------------------

@app.errorhandler(ValidationError)
def handle_validation_error(err):
    return jsonify({"error": err.message}), err.status_code


@app.errorhandler(413)
def handle_too_large(err):
    return jsonify({"error": "File too large."}), 413


@app.errorhandler(Exception)
def handle_unexpected(err):
    traceback.print_exc()
    return jsonify({"error": "Internal analysis error. Please try again."}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _io(data: bytes):
    import io
    return io.BytesIO(data)


def _read_and_validate_upload():
    if "image" not in request.files:
        raise ValidationError("No image file provided.", 400)
    file = request.files["image"]
    if not file or file.filename == "":
        raise ValidationError("No image file provided.", 400)

    filename = secure_filename(file.filename)
    validate_extension(filename)
    validate_mimetype(file.mimetype)

    data = file.read()
    validate_size(data)
    validate_signature(data)

    try:
        img = Image.open(_io(data))
        img.load()  # fully decode now to catch truncated/corrupted data early
    except Exception:
        raise ValidationError("Invalid or corrupted image file.", 400)

    return data, filename


# ---------------------------------------------------------------------------
# Core / legacy-compatible endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "classes": model_service.CLASSES,
        "device": str(model_service.DEVICE),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Kept for backward compatibility with the original prototype."""
    data, filename = _read_and_validate_upload()
    normalize_domain = request.form.get("normalize_domain", "false").lower() == "true"
    original_img = Image.open(_io(data)).convert("RGB")
    result = model_service.predict(original_img, normalize_domain=normalize_domain)
    return jsonify({
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "heatmap": result["heatmap_b64"],
        "domain_normalized": result["domain_normalized"],
    })


# ---------------------------------------------------------------------------
# Unified analysis
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data, filename = _read_and_validate_upload()
    normalize_domain = request.form.get("normalize_domain", "false").lower() == "true"
    result = analysis_service.run_full_analysis(data, filename, normalize_domain=normalize_domain)
    history_service.save_analysis(result)

    response = {k: v for k, v in result.items() if k != "artifacts"}
    return jsonify(response)


@app.route("/api/report", methods=["POST"])
def create_report():
    """Accepts either a fresh image upload (runs analysis then reports),
    or an existing analysis_id (reuses stored analysis + artifacts)."""
    analysis_id = request.form.get("analysis_id")

    if analysis_id:
        analysis = history_service.get_analysis(analysis_id)
        if not analysis:
            raise ValidationError("Analysis ID not found.", 404)
        if not os.path.exists(analysis["artifacts"]["original_path"]):
            raise ValidationError("Stored artifacts for this analysis are no longer available.", 404)
    else:
        data, filename = _read_and_validate_upload()
        analysis = analysis_service.run_full_analysis(data, filename)
        history_service.save_analysis(analysis)

    pdf_path = report_service.generate_report(analysis)
    history_service.mark_report_available(analysis["analysis_id"])

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f'{analysis["analysis_id"]}.pdf',
    )


@app.route("/api/report/<analysis_id>", methods=["GET"])
def get_report(analysis_id):
    analysis = history_service.get_analysis(analysis_id)
    if not analysis:
        raise ValidationError("Analysis ID not found.", 404)

    pdf_path = os.path.join(REPORTS_DIR, f"{analysis_id}.pdf")
    if not os.path.exists(pdf_path):
        if not os.path.exists(analysis["artifacts"]["original_path"]):
            raise ValidationError("Stored artifacts for this analysis are no longer available.", 404)
        pdf_path = report_service.generate_report(analysis)
        history_service.mark_report_available(analysis_id)

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{analysis_id}.pdf",
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@app.route("/api/history", methods=["GET"])
def get_history():
    prediction_filter = request.args.get("prediction")
    search = request.args.get("search")
    limit = int(request.args.get("limit", 200))
    return jsonify(history_service.list_history(limit=limit, prediction_filter=prediction_filter, search=search))


@app.route("/api/history/<analysis_id>", methods=["GET"])
def get_history_item(analysis_id):
    analysis = history_service.get_analysis(analysis_id)
    if not analysis:
        raise ValidationError("Analysis ID not found.", 404)
    response = {k: v for k, v in analysis.items() if k != "artifacts"}
    return jsonify(response)


@app.route("/api/history/<analysis_id>", methods=["DELETE"])
def delete_history_item(analysis_id):
    deleted = history_service.delete_analysis(analysis_id)
    if not deleted:
        raise ValidationError("Analysis ID not found.", 404)
    return jsonify({"deleted": True, "analysis_id": analysis_id})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    return jsonify(history_service.dashboard_stats())


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        **MODEL_INFO,
        "classes": model_service.CLASSES,
        "device": str(model_service.DEVICE),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
