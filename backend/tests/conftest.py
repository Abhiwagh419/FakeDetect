"""Shared fixtures for the FakeDetect backend test suite.

Runs against the real trained checkpoint (backend/model/best_model.pt) and
a real, temporary SQLite database — no mocking of the model or the DB.
This is deliberate: for a forensics tool, "the model loads and produces a
valid probability distribution" and "a saved analysis round-trips through
SQLite correctly" are exactly the things worth testing for real.
"""
import io
import os
import sys

import pytest
from PIL import Image

# Make `backend/` importable the same way app.py expects (config, services,
# utils are imported as top-level packages, not backend.config etc.)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _image_bytes(size=(64, 64), color=(120, 140, 160), fmt="PNG", **save_kwargs):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


@pytest.fixture(scope="session")
def png_bytes():
    return _image_bytes(fmt="PNG")


@pytest.fixture(scope="session")
def jpeg_bytes():
    return _image_bytes(fmt="JPEG", quality=85)


@pytest.fixture(scope="session")
def webp_bytes():
    return _image_bytes(fmt="WEBP")


@pytest.fixture(scope="session")
def large_jpeg_bytes():
    """A bigger, more photo-realistic-sized JPEG for report/PDF tests."""
    return _image_bytes(size=(600, 400), color=(80, 120, 200), fmt="JPEG", quality=90)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """Flask app wired to a temporary DB/artifacts/reports dir per test,
    so tests never touch or pollute the developer's real backend/data/."""
    import config as cfg

    monkeypatch.setattr(cfg, "DB_PATH", str(tmp_path / "database.sqlite"))
    monkeypatch.setattr(cfg, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(cfg, "REPORTS_DIR", str(tmp_path / "reports"))
    os.makedirs(cfg.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)

    # app.py and its services read these constants at import/call time via
    # `from config import ...`, so patch the *consuming* modules too.
    import services.history_service as history_service
    import services.analysis_service as analysis_service
    import services.report_service as report_service

    monkeypatch.setattr(history_service, "DB_PATH", str(tmp_path / "database.sqlite"))
    monkeypatch.setattr(analysis_service, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(report_service, "REPORTS_DIR", str(tmp_path / "reports"))

    import importlib
    import app as flask_app_module
    importlib.reload(flask_app_module)  # re-run init_db() against the patched DB_PATH

    flask_app_module.app.config.update(TESTING=True)
    return flask_app_module.app


@pytest.fixture()
def client(app):
    return app.test_client()
