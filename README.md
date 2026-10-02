# FakeDetect — AI Image Detection & Digital Image Forensics Platform

FakeDetect classifies an uploaded image as **AI-generated** or a **real
photograph** using a self-trained ResNet18 CNN, and surrounds that
prediction with real, computed digital-forensics context: EXIF metadata,
image statistics, cryptographic hashing, JPEG/ELA analysis, Grad-CAM
explainability, and a downloadable professional PDF report.

All inference and analysis happens locally. No third-party AI-detection API
is used — the model was trained from scratch (transfer learning) on
CIFAKE plus supplementary datasets covering human faces, nature/scenes, and
a mix of Stable Diffusion, MidJourney, and DALL-E generated images, and
does all the work on your machine.

---

## Features

- **AI-generated image detection** — ResNet18 binary classifier, softmax
  probability output for FAKE / REAL
- **Grad-CAM explainability** — visual heatmap of the regions that most
  influenced the model's decision
- **Image information** — format, MIME type, resolution, aspect ratio,
  megapixels, color mode, channels
- **Image statistics** — brightness, contrast, sharpness estimate, noise
  estimate, average RGB, all computed from actual pixel data (NumPy)
- **Color histogram** — RGB + luminance distribution, dominant colors
- **EXIF metadata extraction** — camera make/model, timestamps, GPS
  presence, and more, when present; absence is reported as absence, never
  interpreted as a detection signal
- **File integrity hashing** — SHA-256 and MD5 of the uploaded file
- **JPEG / ELA forensics** — progressive-encoding and quantization-table
  inspection; Error Level Analysis visualization for JPEGs
- **Professional PDF report** — 6-page ReportLab report containing the
  actual uploaded image, the actual Grad-CAM image, full forensic detail,
  and the model's technical/evaluation summary
- **Detection history** — SQLite-backed, searchable and filterable
- **Dashboard** — totals, FAKE/REAL distribution, activity over time
- **Model information page** — architecture, dataset, training procedure,
  confusion matrix, precision/recall/F1 (for academic evaluation)
- **Downloadable artifacts** — PDF report, Grad-CAM image, analysis JSON

---

## Architecture

```
frontend/ (React + Vite + Tailwind CSS)
   │  fetch() via src/services/api.js
   ▼
backend/ (Flask)
   ├── app.py                  routes only — thin controller layer
   ├── config.py                paths, constants, static model metrics
   ├── services/
   │   ├── model_service.py     loads best_model.pt, runs inference + Grad-CAM
   │   ├── analysis_service.py  orchestrates a full analysis, persists artifacts
   │   ├── image_analysis.py    image info + statistics + histogram (NumPy/Pillow)
   │   ├── metadata_service.py  EXIF extraction
   │   ├── forensic_service.py  JPEG inspection + Error Level Analysis
   │   ├── report_service.py    PDF report generation (ReportLab)
   │   └── history_service.py   SQLite persistence
   ├── utils/
   │   ├── hashing.py           SHA-256 / MD5 / analysis ID generation
   │   └── validation.py        extension/size/signature validation
   ├── model/best_model.pt      trained checkpoint (not modified by this project)
   ├── data/
   │   ├── database.sqlite      history database
   │   └── artifacts/<id>/      original + Grad-CAM PNGs, kept for report regeneration
   └── reports/                 generated PDF reports
```

The original single-file prototype (`app.py` with `/api/predict` and
`/api/health`) has been refactored into this service-oriented structure.
`/api/predict` and `/api/health` still work exactly as before — nothing
that was already working was broken in the process.

---

## Machine Learning Model

| | |
|---|---|
| Architecture | ResNet18 (ImageNet-pretrained; `layer3` + `layer4` + final classifier fine-tuned) |
| Task | Binary classification — FAKE (AI-generated) vs. REAL |
| Datasets | [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (objects/animals/vehicles) + [real-vs-AI-generated faces](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset) (FFHQ real faces vs. Stable Diffusion / StyleGAN / SFHQ / faceswap fakes) + [AI-vs-human-generated](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) (nature/scenes/portraits) + [AI-generated-images-vs-real-images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) (Stable Diffusion / MidJourney / DALL-E mix vs. Pexels/Unsplash/WikiArt real photos) — merged, so the model covers objects, people, nature/scenes, and modern-generator output |
| Training samples | 210,000 (105,000 REAL / 105,000 FAKE) |
| Test samples | 38,000 (19,000 REAL / 19,000 FAKE) |
| Epochs | 14, trained locally on an NVIDIA RTX 3050 Laptop GPU (6GB VRAM), ~7h32m total |
| Best validation accuracy | 96.35% (epoch 13 — the deployed checkpoint) |
| Test accuracy | 96.26% (epoch 14 evaluation pass) |

Training also uses resolution/sharpness and JPEG-compression augmentation
(see `train_local_v5.py`) so the model isn't biased toward flagging
genuinely sharp, high-resolution real photos as FAKE — a known failure
mode of CIFAKE-only models, whose native images are only 32×32px. Sampling
across the three supplementary datasets is deliberately capped below their
full available size rather than maximized, based on evidence from this
project's own iteration history that maximizing training volume does not
reliably improve real-world detection quality and can bias the model
toward unrepresentative visual styles (e.g. professional stock photography
as "REAL").

**Confusion matrix (test set):**

|              | Predicted FAKE | Predicted REAL |
|--------------|-----------------|-----------------|
| Actual FAKE  | 18,300          | 700             |
| Actual REAL  | 723             | 18,277          |

**Classification report:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| FAKE  | 0.96      | 0.96   | 0.96 |
| REAL  | 0.96      | 0.96   | 0.96 |

These figures are static, reported training results (from
`train_local_v5.py`) and are surfaced as-is in the app and PDF report —
they are not recomputed at runtime.

### Engineering principle: no invented metrics

Every number shown in the UI or PDF corresponds to an actual calculation
or a known model output. The app never claims that missing EXIF data or
an ELA visualization "proves" AI generation — these are documented as
supplementary, non-definitive signals throughout the UI, PDF report, and
JSON export.

---

## Setup

### Backend

```bash
cd backend
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

The backend starts on `http://localhost:5000`. Verify with:
```bash
curl http://localhost:5000/api/health
```

The trained checkpoint must be present at `backend/model/best_model.pt`
(already included in this repository — it is not retrained or modified by
this refactor).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend starts on `http://localhost:5173`.

### Environment variables

Copy `frontend/.env.example` to `frontend/.env` and adjust if the backend
runs somewhere other than `localhost:5000`:

```
VITE_API_URL=http://localhost:5000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend + model status |
| POST | `/api/predict` | Legacy prediction-only endpoint (kept for compatibility) |
| POST | `/api/analyze` | Full unified analysis — prediction, image info, statistics, EXIF, hash, JPEG/ELA forensics, Grad-CAM. Saves to history. |
| POST | `/api/report` | Generates a PDF report. Accepts either an `image` file (analyzes fresh) or an `analysis_id` (reuses a stored analysis). Returns the PDF as a download. |
| GET | `/api/report/<analysis_id>` | Re-downloads (or regenerates) the PDF for a stored analysis. |
| GET | `/api/history` | Lists stored analyses. Query params: `prediction` (FAKE/REAL), `search` (filename), `limit`. |
| GET | `/api/history/<analysis_id>` | Full stored analysis by ID. |
| DELETE | `/api/history/<analysis_id>` | Deletes a stored analysis. |
| GET | `/api/dashboard` | Aggregate stats: totals, FAKE/REAL split, average probability, recent analyses, activity timeline. |
| GET | `/api/model-info` | Static model architecture/dataset/metrics. |

**Error responses** use plain JSON (`{"error": "..."}`) with correct HTTP
status codes (400 invalid request, 404 not found, 413 file too large, 415
unsupported media type, 500 internal analysis error) and never expose raw
Python tracebacks.

**Example `/api/analyze` response shape** (trimmed):
```json
{
  "analysis_id": "FD-20260811-1A8BF041",
  "filename": "example.jpg",
  "prediction": "FAKE",
  "fake_probability": 99.61,
  "real_probability": 0.39,
  "model_confidence_tier": "HIGH_MODEL_CONFIDENCE",
  "image": { "format": "JPEG", "width": 1920, "height": 1080, "...": "..." },
  "statistics": { "brightness": { "...": "..." }, "...": "..." },
  "histogram": { "r": [...], "g": [...], "b": [...], "dominant_colors": [...] },
  "metadata": { "availability": "NOT_FOUND", "...": "..." },
  "hash": { "sha256": "...", "md5": "..." },
  "jpeg_forensics": { "...": "..." },
  "ela": { "image_b64": "...", "...": "..." },
  "gradcam": { "image_b64": "...", "note": "..." },
  "model": { "architecture": "ResNet18", "test_accuracy": 0.9626, "...": "..." }
}
```

---

## Database

SQLite, single table (`backend/data/database.sqlite`):

```sql
CREATE TABLE history (
    id TEXT PRIMARY KEY,        
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    prediction TEXT NOT NULL,
    fake_probability REAL NOT NULL,
    real_probability REAL NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    report_available INTEGER DEFAULT 0,
    analysis_json TEXT NOT NULL     
);
```

Raw images are **not** stored in the database. The original image and its
Grad-CAM overlay are saved as small PNGs under
`backend/data/artifacts/<analysis_id>/`, referenced by path, so PDF
reports can be regenerated later without re-uploading.

---

## Validation & Security

- Allowed extensions: `jpg`, `jpeg`, `png`, `webp`
- Maximum upload size: 15 MB
- Magic-byte file-signature verification (the authoritative check — client-
  supplied MIME types are treated as an untrustworthy, soft signal only,
  since they're easily generic or wrong)
- Full image decode (`Image.load()`) before analysis, so truncated/corrupted
  files fail cleanly with 400 rather than crashing mid-analysis
- Secure filename sanitization
- No raw Python tracebacks are ever returned to the client

---

## Frontend

React + Vite + Tailwind CSS v4, `react-router-dom` for navigation,
`recharts` for charts, `lucide-react` for icons.

```
frontend/src/
  services/api.js       centralized API client (VITE_API_URL configurable)
  components/
    layout/AppShell.jsx  responsive sidebar/drawer navigation
    common/               Card, KeyValueRow, ProbabilityBar, StatusBadge, Spinner
    analyze/               UploadZone, DetectionResult, ImageDetailsCard,
                            StatisticsCard, HistogramCard, MetadataCard,
                            GradCamCard, ForensicsCard, ForensicSummaryCard
  pages/
    Dashboard.jsx  Analyze.jsx  History.jsx  Reports.jsx  Model.jsx
```

Design: dark, glass-panel forensic aesthetic with a cyan accent, Space
Grotesk/Inter/JetBrains Mono type system, and a signature scan-frame
corner-bracket motif (with an animated scan sweep during analysis) that
ties the visual language to the "image forensics" theme. The sidebar
collapses to a slide-out drawer with a hamburger toggle below the `lg`
breakpoint.

---

## Testing performed

Verified against a live running backend + frontend:

- `/api/health`, `/api/predict` (legacy), `/api/analyze` on JPEG, PNG, and WEBP
- PDF report generation — all 6 pages rendered to images and visually
  inspected (cover, image info, detection, Grad-CAM, forensics, technical
  summary)
- History create/list/search/filter/view/delete
- Dashboard aggregate stats and charts
- Error paths: no file, unsupported extension, bad file signature,
  truncated/corrupted image, oversized file (>15 MB), multiple consecutive
  analyses
- Full UI flow via Playwright: file upload → run detection → all result
  cards render → generate PDF report → downloads a valid multi-page PDF;
  history detail modal; mobile drawer navigation
- Desktop (1440px) and mobile (390px) viewports
- Real-world verification against images generated with ChatGPT's image
  tool: 5/5 AI-generated face images correctly classified FAKE (91–97%
  confidence); 2/2 AI-generated nature/landscape images still
  misclassified REAL (95%+ confidence) — see Known limitations below

## Known limitations

- The classifier is trained on the datasets listed above; it may not
  generalize to generative models or image pipelines outside that
  training distribution.
- **Verified working:** real-world testing confirms reliable detection of
  AI-generated **face** images from modern proprietary generators (e.g.
  ChatGPT's image tool) — every test case attempted was correctly
  classified FAKE with 91–97% confidence.
- **Known open limitation:** AI-generated **nature/landscape** images from
  the same modern generators may still be misclassified as REAL — this
  was observed directly in testing and is not yet resolved. Closing it
  would likely require a dataset that combines both nature/landscape
  content and modern-generator coverage together, which none of the
  current training sources provide in combination.
- Sharpness and noise statistics are heuristic estimates (Laplacian
  variance / high-frequency residual), not calibrated, standardized image-
  quality scores.
- ELA and JPEG-encoding characteristics are forensic visualization aids,
  not proof of manipulation.
- EXIF absence is common for images that have simply been re-saved,
  screenshotted, or stripped by a messaging app — it is not, by itself,
  evidence of AI generation.
- The Flask dev server is used for local/academic evaluation; a production
  deployment should sit behind a WSGI server (gunicorn/waitress) and a
  reverse proxy.

## Future improvements

- Batch analysis of multiple images at once
- User accounts / multi-tenant history
- Configurable confidence-tier thresholds
- Additional forensic signals (e.g. PRNU, copy-move detection)
- Async/background report generation for very large images
- A combined nature-and-modern-generator dataset to close the remaining
  nature/landscape detection gap documented above
