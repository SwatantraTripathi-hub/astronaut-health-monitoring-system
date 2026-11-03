"""Tests for the PhysioNet catalog and on-demand record download."""

from pathlib import Path

import pytest

from src.ingest import datasets


def test_catalog_lists_only_records_the_project_uses():
    assert datasets.catalog_choices() == [
        "mitdb/100",
        "mitdb/201",
        "mitdb/202",
        "bidmc/bidmc01",
    ]


def test_record_choices_include_catalog_and_extra_local_records(tmp_path):
    bidmc = tmp_path / "bidmc"
    bidmc.mkdir()
    (bidmc / "bidmc02.hea").touch()
    (bidmc / "bidmc02n.hea").touch()

    choices = datasets.record_choices(tmp_path)

    assert choices[:4] == datasets.catalog_choices()
    assert "bidmc/bidmc02" in choices
    assert "bidmc/bidmc02n" not in choices


def test_ensure_record_skips_download_when_required_files_exist(
    tmp_path, monkeypatch
):
    record = datasets.get_catalog_record("mitdb/100")
    base = tmp_path / "mitdb"
    base.mkdir()
    for name in record.required:
        (base / name).touch()

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be attempted")

    monkeypatch.setattr(datasets.wfdb, "dl_files", fail_download)

    path = datasets.ensure_record(tmp_path, "mitdb/100")

    assert path == base / "100"
    assert datasets.is_record_available(tmp_path, record)


def test_ensure_record_downloads_missing_catalog_files(tmp_path, monkeypatch):
    record = datasets.get_catalog_record("mitdb/100")
    calls = []

    def fake_download(kind, directory, files, **kwargs):
        calls.append((kind, Path(directory), list(files)))
        for name in files:
            (Path(directory) / name).touch()

    monkeypatch.setattr(datasets.wfdb, "dl_files", fake_download)

    path = datasets.ensure_record(tmp_path, "mitdb/100")

    assert path == tmp_path / "mitdb" / "100"
    assert calls == [("mitdb", tmp_path / "mitdb", list(record.files))]
    assert datasets.is_record_available(tmp_path, record)


def test_ensure_record_rejects_unknown_missing_record(tmp_path):
    with pytest.raises(datasets.DatasetUnavailableError):
        datasets.ensure_record(tmp_path, "mitdb/999")


def test_ensure_record_wraps_download_failures(tmp_path, monkeypatch):
    def fail_download(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(datasets.wfdb, "dl_files", fail_download)

    with pytest.raises(datasets.DatasetUnavailableError) as excinfo:
        datasets.ensure_record(tmp_path, "mitdb/100")

    assert "PhysioNet" in str(excinfo.value)
    assert not (tmp_path / "mitdb" / "100.hea").exists()
