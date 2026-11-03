import os

import numpy as np
import pytest
import wfdb

from src.ingest.wfdb_loader import (
    NON_BEAT_SYMBOLS,
    LoadedRecord,
    RecordLoadError,
    SignalNotFound,
    beat_only,
    get_signal,
    load_beat_annotations,
    load_record,
    normalize_name,
)


def test_normalize_name():
    assert normalize_name("RESP,") == "RESP"
    assert normalize_name("  RESP,  ") == "RESP"


def test_loader_simulated(tmp_path):
    base = tmp_path / "simrec"
    sig_name = ["ECG", "RESP,"]
    p_signal = np.column_stack(
        [np.sin(np.linspace(0, 1, 100)), np.ones(100)]).astype(float)
    units = ["mV", "pm"]
    wfdb.wrsamp(base.name, fs=125, units=units, sig_name=sig_name,
                p_signal=p_signal, fmt=["16", "16"], write_dir=str(tmp_path))
    rec_path = str(tmp_path / base.name)
    rec = load_record(rec_path, data_type="simulated", source_label="sim")
    assert "RESP" in rec.signal_names
    assert rec.data_type == "simulated"
    s = get_signal(rec, "resp")
    assert isinstance(s, np.ndarray)
    assert len(s) == 100


def test_missing_file_raises():
    with pytest.raises(RecordLoadError):
        load_record("no_such_record_12345")


def test_loader_rejects_unsafe_path_before_wfdb_access():
    with pytest.raises(RecordLoadError, match="NUL"):
        load_record("record\x00name")


def test_annotation_loader_rejects_unsafe_extension():
    with pytest.raises(RecordLoadError, match="extension"):
        load_beat_annotations("record", ext="../atr")


def test_get_signal_missing_raises():
    from pathlib import Path

    d = tmp_path if False else None
    # use pytest tmp_path via fixture? but test defined without it; easier create own
    import tempfile

    d = Path(tempfile.mkdtemp())
    base = d / "s"
    wfdb.wrsamp(str(base.name), fs=100, units=["mV"], sig_name=[
                "ECG"], p_signal=np.ones((50, 1)).astype(float), fmt=["16"], write_dir=str(d))
    rec = load_record(str(d / base.name))
    with pytest.raises(SignalNotFound):
        get_signal(rec, "NOPE")


@pytest.mark.realdata
def test_realdata_mitdb_100():
    if not os.path.exists("data/raw/mitdb/100.hea"):
        pytest.skip("real data missing")
    rec = load_record("data/raw/mitdb/100")
    assert rec.fs == 360
    assert rec.sig_len == 650000
    assert set(rec.signal_names) == {"MLII", "V5"}


@pytest.mark.realdata
def test_realdata_bidmc01():
    if not os.path.exists("data/raw/bidmc/bidmc01.hea"):
        pytest.skip("real data missing")
    rec = load_record("data/raw/bidmc/bidmc01")
    assert rec.fs == 125
    assert rec.sig_len == 60001
    assert set(rec.signal_names) == {"RESP", "PLETH", "V", "AVR", "II"}


def test_beat_only_drops_non_beat_symbols():
    samples = np.arange(6)
    symbols = ["N", "~", "|", "x", "+", "N"]
    out_samples, out_symbols = beat_only(samples, symbols)
    assert out_symbols == ["N", "N"]
    assert list(out_samples) == [0, 5]


@pytest.mark.realdata
def test_realdata_201_beat_only_leaves_only_beat_symbols():
    if not os.path.exists("data/raw/mitdb/201.hea"):
        pytest.skip("real data missing")
    samples, symbols, _ = load_beat_annotations("data/raw/mitdb/201")
    print("symbols in file:", sorted(set(symbols)))
    print("non-beat symbols:", sorted(NON_BEAT_SYMBOLS))
    _, kept = beat_only(samples, symbols)
    allowed = set(symbols) - set(NON_BEAT_SYMBOLS)
    assert set(kept) <= allowed
    assert len(kept) > 0


@pytest.mark.realdata
def test_realdata_bidmc01n():
    if not os.path.exists("data/raw/bidmc/bidmc01n.hea"):
        pytest.skip("real data missing")
    rec = load_record("data/raw/bidmc/bidmc01n")
    assert rec.fs == 1
    assert "SpO2" in rec.signal_names
