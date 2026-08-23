import pytest

from services import analysis_service, history_service


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.sqlite")
    monkeypatch.setattr(history_service, "DB_PATH", db_path)
    monkeypatch.setattr(analysis_service, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    history_service.init_db()
    return db_path


def _make_analysis(png_bytes, filename="sample.png"):
    return analysis_service.run_full_analysis(png_bytes, filename)


def test_save_and_retrieve_round_trips_full_analysis(db, png_bytes):
    analysis = _make_analysis(png_bytes)
    history_service.save_analysis(analysis)

    fetched = history_service.get_analysis(analysis["analysis_id"])
    assert fetched is not None
    assert fetched["analysis_id"] == analysis["analysis_id"]
    assert fetched["prediction"] == analysis["prediction"]


def test_get_nonexistent_analysis_returns_none(db):
    assert history_service.get_analysis("FD-00000000-DEADBEEF") is None


def test_list_history_returns_saved_items(db, png_bytes):
    a = _make_analysis(png_bytes, "one.png")
    history_service.save_analysis(a)
    items = history_service.list_history()
    assert len(items) == 1
    assert items[0]["analysis_id"] == a["analysis_id"]


def test_list_history_filters_by_prediction(db, png_bytes):
    a = _make_analysis(png_bytes, "one.png")
    history_service.save_analysis(a)

    matching = history_service.list_history(prediction_filter=a["prediction"])
    other = "REAL" if a["prediction"] == "FAKE" else "FAKE"
    non_matching = history_service.list_history(prediction_filter=other)

    assert len(matching) == 1
    assert len(non_matching) == 0


def test_list_history_filters_by_search(db, png_bytes):
    a = _make_analysis(png_bytes, "vacation_photo.png")
    history_service.save_analysis(a)

    assert len(history_service.list_history(search="vacation")) == 1
    assert len(history_service.list_history(search="nonexistent")) == 0


def test_delete_analysis_removes_it(db, png_bytes):
    a = _make_analysis(png_bytes)
    history_service.save_analysis(a)

    deleted = history_service.delete_analysis(a["analysis_id"])
    assert deleted is True
    assert history_service.get_analysis(a["analysis_id"]) is None


def test_delete_nonexistent_returns_false(db):
    assert history_service.delete_analysis("FD-00000000-DEADBEEF") is False


def test_mark_report_available_updates_flag(db, png_bytes):
    a = _make_analysis(png_bytes)
    history_service.save_analysis(a)
    history_service.mark_report_available(a["analysis_id"])

    items = history_service.list_history()
    assert items[0]["report_available"] is True


def test_dashboard_stats_on_empty_db(db):
    stats = history_service.dashboard_stats()
    assert stats["total_analyses"] == 0
    assert stats["fake_detected"] == 0
    assert stats["real_detected"] == 0
    assert stats["average_model_probability"] == 0


def test_dashboard_stats_counts_match_saved_items(db, png_bytes):
    a1 = _make_analysis(png_bytes, "one.png")
    a2 = _make_analysis(png_bytes, "two.png")
    history_service.save_analysis(a1)
    history_service.save_analysis(a2)

    stats = history_service.dashboard_stats()
    assert stats["total_analyses"] == 2
    assert stats["fake_detected"] + stats["real_detected"] == 2
    assert len(stats["recent"]) == 2
