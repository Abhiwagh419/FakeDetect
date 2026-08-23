"""Generates a professional multi-page PDF forensic report using ReportLab.
Includes the actual uploaded image and the actual Grad-CAM image, scaled
to fit while preserving aspect ratio."""
import os
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, NextPageTemplate
)
from reportlab.lib.enums import TA_CENTER
from PIL import Image as PILImage

from config import MODEL_INFO, DISCLAIMER, REPORTS_DIR

INK = colors.HexColor("#0f172a")
ACCENT = colors.HexColor("#6366f1")
MUTED = colors.HexColor("#64748b")
FAKE_COLOR = colors.HexColor("#e11d48")
REAL_COLOR = colors.HexColor("#059669")
LINE = colors.HexColor("#e2e8f0")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("FDTitle", fontSize=26, leading=30, textColor=INK, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("FDSubtitle", fontSize=12, textColor=MUTED, fontName="Helvetica"))
styles.add(ParagraphStyle("FDH2", fontSize=15, leading=18, textColor=INK, fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=8))
styles.add(ParagraphStyle("FDBody", fontSize=9.5, leading=14, textColor=INK, fontName="Helvetica"))
styles.add(ParagraphStyle("FDSmall", fontSize=8, leading=11, textColor=MUTED, fontName="Helvetica"))
styles.add(ParagraphStyle("FDCenter", parent=styles["FDBody"], alignment=TA_CENTER))


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(20 * mm, 287 * mm, "FAKEDETECT")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(190 * mm, 287 * mm, "AI Image Forensics Report")
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 284 * mm, 190 * mm, 284 * mm)

    canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    canvas.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _fit_image(path, max_w_mm, max_h_mm):
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = min((max_w_mm * mm) / w, (max_h_mm * mm) / h)
    return RLImage(path, width=w * ratio, height=h * ratio)


def _kv_table(rows, col_widths=(45 * mm, 100 * mm)):
    data = [[Paragraph(f"<b>{k}</b>", styles["FDBody"]), Paragraph(str(v), styles["FDBody"])] for k, v in rows]
    t = Table(data, colWidths=list(col_widths))
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
    ]))
    return t


def generate_report(analysis: dict) -> str:
    """Builds the PDF and returns the output file path."""
    analysis_id = analysis["analysis_id"]
    out_path = os.path.join(REPORTS_DIR, f"{analysis_id}.pdf")

    doc = BaseDocTemplate(out_path, pagesize=A4,
                           leftMargin=20 * mm, rightMargin=20 * mm,
                           topMargin=25 * mm, bottomMargin=20 * mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=_header_footer)])

    story = []
    story += _page1_cover(analysis)
    story.append(PageBreak())
    story += _page2_image_info(analysis)
    story.append(PageBreak())
    story += _page3_detection(analysis)
    story.append(PageBreak())
    story += _page4_xai(analysis)
    story.append(PageBreak())
    story += _page5_forensics(analysis)
    story.append(PageBreak())
    story += _page6_technical(analysis)

    doc.build(story)
    return out_path


def _page1_cover(a):
    elems = [
        Spacer(1, 4 * mm),
        Paragraph("FAKEDETECT", styles["FDTitle"]),
        Paragraph("AI Image Forensics Report", styles["FDSubtitle"]),
        Spacer(1, 8 * mm),
    ]
    img_path = a["artifacts"]["original_path"]
    elems.append(_fit_image(img_path, 150, 90))
    elems.append(Spacer(1, 8 * mm))

    is_fake = a["prediction"] == "FAKE"
    badge_color = FAKE_COLOR if is_fake else REAL_COLOR
    verdict_text = "AI-GENERATED" if is_fake else "REAL PHOTOGRAPH"
    badge = Table([[Paragraph(f'<font color="white"><b>{verdict_text}</b></font>', styles["FDCenter"])]],
                   colWidths=[90 * mm])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elems.append(badge)
    elems.append(Spacer(1, 6 * mm))

    elems.append(_kv_table([
        ("Model Probability", f'{max(a["fake_probability"], a["real_probability"]):.2f}%'),
        ("Confidence Tier", a["model_confidence_tier"].replace("_", " ").title()),
        ("Analysis ID", a["analysis_id"]),
        ("Filename", a["filename"]),
        ("Generated", a["timestamp"]),
    ]))
    elems.append(Spacer(1, 6 * mm))
    elems.append(Paragraph(DISCLAIMER, styles["FDSmall"]))
    return elems


def _page2_image_info(a):
    info = a["image"]
    elems = [Paragraph("Image Information", styles["FDH2"])]
    elems.append(_fit_image(a["artifacts"]["original_path"], 90, 60))
    elems.append(Spacer(1, 4 * mm))
    elems.append(_kv_table([
        ("Filename", info["filename"]),
        ("Format", info["format"]),
        ("MIME Type", info["mime_type"]),
        ("File Size", info["file_size_readable"]),
        ("Resolution", f'{info["width"]} × {info["height"]}'),
        ("Aspect Ratio", info["aspect_ratio_readable"]),
        ("Megapixels", f'{info["megapixels"]} MP'),
        ("Color Mode", info["color_mode"]),
        ("Channels", info["channels"]),
        ("SHA-256", a["hash"]["sha256"]),
    ]))
    return elems


def _page3_detection(a):
    elems = [Paragraph("AI Detection", styles["FDH2"])]
    elems.append(_kv_table([
        ("Prediction", a["prediction"]),
        ("FAKE Probability", f'{a["fake_probability"]:.2f}%'),
        ("REAL Probability", f'{a["real_probability"]:.2f}%'),
        ("Confidence Tier", a["model_confidence_tier"].replace("_", " ").title()),
    ]))
    elems.append(Spacer(1, 6 * mm))
    elems.append(_prob_bar("FAKE", a["fake_probability"], FAKE_COLOR))
    elems.append(Spacer(1, 3 * mm))
    elems.append(_prob_bar("REAL", a["real_probability"], REAL_COLOR))
    elems.append(Spacer(1, 6 * mm))
    elems.append(Paragraph("Model Reference", styles["FDH2"]))
    elems.append(_kv_table([
        ("Architecture", MODEL_INFO["architecture"]),
        ("Dataset", MODEL_INFO["dataset"]),
        ("Best Validation Accuracy", f'{MODEL_INFO["best_val_accuracy"] * 100:.2f}%'),
        ("Test Accuracy", f'{MODEL_INFO["test_accuracy"] * 100:.2f}%'),
    ]))
    elems.append(Spacer(1, 4 * mm))
    elems.append(Paragraph(
        "This is the model's softmax output probability for each class, and a UI "
        "interpretation of that probability into a confidence tier — not a "
        "calibrated real-world certainty.", styles["FDSmall"]))
    return elems


def _prob_bar(label, pct, color):
    filled = max(1, int(pct / 100 * 60))
    bar = Table([[
        Paragraph(f"<b>{label}</b>", styles["FDBody"]),
        Table([[""]], colWidths=[filled * 1.2], rowHeights=[6],
              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), color)])),
        Paragraph(f"{pct:.2f}%", styles["FDBody"]),
    ]], colWidths=[25 * mm, 100 * mm, 20 * mm])
    bar.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return bar


def _page4_xai(a):
    elems = [Paragraph("Explainable AI — Grad-CAM", styles["FDH2"])]
    elems.append(Table([[
        _fit_image(a["artifacts"]["original_path"], 75, 75),
        _fit_image(a["artifacts"]["gradcam_path"], 75, 75),
    ]], colWidths=[85 * mm, 85 * mm]))
    elems.append(Spacer(1, 4 * mm))
    elems.append(Paragraph("Left: original image. Right: Grad-CAM overlay.", styles["FDSmall"]))
    elems.append(Spacer(1, 4 * mm))
    elems.append(Paragraph(a["gradcam"]["note"], styles["FDBody"]))
    return elems


def _page5_forensics(a):
    stats = a["statistics"]
    meta = a["metadata"]
    elems = [Paragraph("Image Forensics", styles["FDH2"])]

    elems.append(Paragraph("Metadata (EXIF)", styles["FDBody"]))
    if meta["highlighted_fields"]:
        rows = [(k, v) for k, v in meta["highlighted_fields"].items()]
    else:
        rows = [("Status", "No EXIF metadata detected.")]
    elems.append(_kv_table(rows))
    elems.append(Paragraph(meta["note"], styles["FDSmall"]))
    elems.append(Spacer(1, 4 * mm))

    elems.append(Paragraph("Image Statistics", styles["FDBody"]))
    elems.append(_kv_table([
        ("Brightness (mean luminance)", f'{stats["brightness"]["mean_luminance_0_255"]:.2f} / 255'),
        ("Contrast (luminance std dev)", f'{stats["contrast"]["std_luminance_0_255"]:.2f}'),
        ("Sharpness estimate", f'{stats["sharpness"]["laplacian_variance"]:.2f} (Laplacian variance)'),
        ("Noise estimate", f'{stats["noise"]["residual_std"]:.3f} (residual std dev)'),
        ("Average RGB", f'({stats["average_rgb"]["r"]:.0f}, {stats["average_rgb"]["g"]:.0f}, {stats["average_rgb"]["b"]:.0f})'),
    ]))
    elems.append(Spacer(1, 4 * mm))

    jpeg = a["jpeg_forensics"]
    elems.append(Paragraph("Compression", styles["FDBody"]))
    if jpeg.get("applicable"):
        elems.append(_kv_table([
            ("Progressive Encoding", jpeg.get("progressive_encoding")),
            ("Quantization Tables Present", jpeg.get("quantization_tables_present")),
        ]))
    else:
        elems.append(Paragraph(jpeg.get("note", ""), styles["FDSmall"]))

    if a.get("ela"):
        elems.append(Spacer(1, 4 * mm))
        elems.append(Paragraph("Error Level Analysis (ELA)", styles["FDBody"]))
        elems.append(Paragraph(a["ela"]["note"], styles["FDSmall"]))

    return elems


def _page6_technical(a):
    mi = MODEL_INFO
    cm = mi["confusion_matrix"]
    cr = mi["classification_report"]
    elems = [Paragraph("Technical Summary", styles["FDH2"])]
    elems.append(_kv_table([
        ("Model Architecture", mi["architecture"]),
        ("Dataset", mi["dataset"]),
        ("Training Samples", f'{mi["train_samples"]:,} ({mi["train_real"]:,} REAL / {mi["train_fake"]:,} FAKE)'),
        ("Test Samples", f'{mi["test_samples"]:,} ({mi["test_real"]:,} REAL / {mi["test_fake"]:,} FAKE)'),
        ("Epochs / Hardware", f'{mi["epochs"]} epochs, {mi["training_hardware"]}'),
        ("Best Validation Accuracy", f'{mi["best_val_accuracy"] * 100:.2f}%'),
        ("Test Accuracy", f'{mi["test_accuracy"] * 100:.2f}%'),
    ]))
    elems.append(Spacer(1, 4 * mm))

    elems.append(Paragraph("Classification Report", styles["FDBody"]))
    cr_rows = [["Class", "Precision", "Recall", "F1"]]
    for cls, m in cr.items():
        cr_rows.append([cls, f'{m["precision"]:.2f}', f'{m["recall"]:.2f}', f'{m["f1"]:.2f}'])
    cr_table = Table(cr_rows, colWidths=[40 * mm, 35 * mm, 35 * mm, 35 * mm])
    cr_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elems.append(cr_table)
    elems.append(Spacer(1, 4 * mm))

    elems.append(Paragraph("Confusion Matrix (Test Set)", styles["FDBody"]))
    labels = cm["labels"]
    cm_rows = [["Actual \\ Predicted"] + labels]
    for i, row in enumerate(cm["matrix"]):
        cm_rows.append([labels[i]] + [str(v) for v in row])
    cm_table = Table(cm_rows, colWidths=[45 * mm] + [35 * mm] * len(labels))
    cm_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elems.append(cm_table)
    elems.append(Spacer(1, 6 * mm))

    elems.append(Paragraph("Limitations & Disclaimer", styles["FDBody"]))
    elems.append(Paragraph(DISCLAIMER, styles["FDSmall"]))
    return elems
