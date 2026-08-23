# FakeDetect backend test suite

85 tests covering validation, hashing, image analysis, metadata (EXIF),
JPEG/ELA forensics, the trained model's inference + Grad-CAM, the SQLite
history layer, and every Flask API route end-to-end (via Flask's test
client, hitting the real endpoints — not mocked).

These tests load the **real** trained checkpoint
(`backend/model/best_model.pt`) and write to a **temporary** SQLite DB and
artifacts directory per test run — they never touch or modify
`backend/data/` or `backend/reports/`.

## Run

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

## With coverage

```bash
pytest --cov=services --cov=utils --cov-report=term-missing
```

Current coverage on `services/` + `utils/`: **98%** statements.

## Layout

| File | Covers |
|---|---|
| `test_validation.py` | Upload security: extension, MIME, size, magic-byte signature checks, filename sanitization |
| `test_hashing.py` | SHA-256/MD5 hashing, analysis ID generation |
| `test_image_analysis.py` | Image info, brightness/contrast/sharpness/noise stats, histograms |
| `test_metadata_service.py` | EXIF extraction, incl. a regression test for GPS-tag naming |
| `test_forensic_service.py` | JPEG quantization inspection, Error Level Analysis |
| `test_model_service.py` | Real ResNet18 inference + Grad-CAM on the trained checkpoint |
| `test_analysis_service.py` | Full analysis orchestration pipeline |
| `test_history_service.py` | SQLite persistence: save/list/filter/delete/dashboard stats |
| `test_api.py` | Every Flask route, incl. error paths (bad upload, spoofed file, 404s) |

`test_model_service.py` and the model-backed parts of `test_api.py` /
`test_analysis_service.py` are the slowest (real forward+backward passes
through ResNet18 on CPU) — expect the full suite to take ~10-15s.
