import os

import numpy as np
import pytest
import wfdb

from src.validate.validator import Issue, ValidationReport, nan_fraction, validate_record


def test_nan_fraction_helper():
    assert nan_fraction([1.0, float('nan'), 2.0]) == pytest.approx(1 / 3)


def test_valid_format16(tmp_path):
    base = tmp_path / "v16"
    p_signal = np.column_stack([np.ones(100), np.zeros(100)]).astype(float)
    wfdb.wrsamp(base.name, fs=250, units=["mV", "mV"], sig_name=["ECG", "II"], p_signal=p_signal, fmt=["16", "16"], write_dir=str(tmp_path))
    rep = validate_record(str(tmp_path / base.name))
    assert rep.status == "valid"


def test_valid_format212(tmp_path):
    base = tmp_path / "v212"
    p_signal = np.column_stack([np.ones(120), np.zeros(120)]).astype(float)
    wfdb.wrsamp(base.name, fs=360, units=["mV", "mV"], sig_name=["MLII", "V1"], p_signal=p_signal, fmt=["212", "212"], write_dir=str(tmp_path))
    rep = validate_record(str(tmp_path / base.name))
    assert rep.status == "valid"


def test_truncate_dat_mismatch(tmp_path):
    base = tmp_path / "bad"
    p_signal = np.column_stack([np.ones(100)]).astype(float)
    wfdb.wrsamp(base.name, fs=250, units=["mV"], sig_name=["ECG"], p_signal=p_signal, fmt=["16"], write_dir=str(tmp_path))
    p = tmp_path / (base.name + ".dat")
    data = p.read_bytes()
    p.write_bytes(data[:-100])
    rep = validate_record(str(tmp_path / base.name))
    assert rep.status == "invalid"
    codes = [i.code for i in rep.issues]
    assert "SIZE_MISMATCH" in codes


def test_delete_dat_file_missing(tmp_path):
    base = tmp_path / "bad2"
    p_signal = np.ones((50, 1)).astype(float)
    wfdb.wrsamp(base.name, fs=250, units=["mV"], sig_name=["ECG"], p_signal=p_signal, fmt=["16"], write_dir=str(tmp_path))
    (tmp_path / (base.name + ".dat")).unlink()
    rep = validate_record(str(tmp_path / base.name))
    assert rep.status == "invalid"
    codes = [i.code for i in rep.issues]
    assert "FILE_MISSING" in codes


def test_garbage_hea_no_exception(tmp_path):
    base = tmp_path / "bad3"
    (tmp_path / (base.name + ".hea")).write_text("not a header\n")
    (tmp_path / (base.name + ".dat")).write_bytes(b"1234")
    rep = validate_record(str(tmp_path / base.name))
    assert isinstance(rep, ValidationReport)
    assert rep.status == "invalid"
    codes = [i.code for i in rep.issues]
    assert "HEADER_UNREADABLE" in codes


def test_bad_fs(tmp_path):
    base = tmp_path / "bad4"
    p_signal = np.ones((50, 1)).astype(float)
    # write with bad fs by editing header? easier: write normal then change
    wfdb.wrsamp(base.name, fs=250, units=["mV"], sig_name=["ECG"], p_signal=p_signal, fmt=["16"], write_dir=str(tmp_path))
    hea = (tmp_path / (base.name + ".hea")).read_text()
    # replace fs
    hea2 = hea.replace("250", "0", 1)
    (tmp_path / (base.name + ".hea")).write_text(hea2)
    rep = validate_record(str(tmp_path / base.name))
    assert rep.status == "invalid"
    codes = [i.code for i in rep.issues]
    assert "BAD_FS" in codes


def test_duplicate_names(tmp_path):
    base = tmp_path / "bad5"
    p_signal = np.column_stack([np.ones(50), np.ones(50)]).astype(float)
    wfdb.wrsamp(base.name, fs=250, units=["mV", "mV"], sig_name=["ECG,", "ECG"], p_signal=p_signal, fmt=["16", "16"], write_dir=str(tmp_path))
    rep = validate_record(str(tmp_path / base.name))
    assert rep.status == "invalid"
    codes = [i.code for i in rep.issues]
    assert "DUPLICATE_SIGNAL_NAME" in codes


def test_unknown_format_skipped(tmp_path):
    base = tmp_path / "bad6"
    # create with fmt 16 then change header format to 80?
    p_signal = np.ones((50, 1)).astype(float)
    wfdb.wrsamp(base.name, fs=250, units=["mV"], sig_name=["ECG"], p_signal=p_signal, fmt=["16"], write_dir=str(tmp_path))
    hea = (tmp_path / (base.name + ".hea")).read_text()
    hea2 = hea.replace("16", "80", 1)
    (tmp_path / (base.name + ".hea")).write_text(hea2)
    rep = validate_record(str(tmp_path / base.name))
    codes = [i.code for i in rep.issues]
    assert "SIZE_CHECK_SKIPPED" in codes
    assert "size" in rep.not_checked


def _write_record(tmp_path, name, p_signal, sig_name, comments=None):
    wfdb.wrsamp(
        name,
        fs=1,
        units=["%"] * len(sig_name),
        sig_name=sig_name,
        p_signal=p_signal,
        fmt=["16"] * len(sig_name),
        comments=comments,
        write_dir=str(tmp_path),
    )
    return str(tmp_path / name)


def test_metadata_word_nan_elsewhere_does_not_trigger(tmp_path):
    sig = np.ones((20, 1)).astype(float)
    path = _write_record(tmp_path, "meta_ok", sig, ["PULSE"], comments=["<age>: 88 <sex>: M", "derived from nan test"])
    rep = validate_record(path)
    codes = [i.code for i in rep.issues]
    assert "METADATA_MISSING" not in codes


def test_metadata_missing_age_nan_and_sex_question_triggers(tmp_path):
    sig = np.ones((20, 1)).astype(float)
    path = _write_record(tmp_path, "meta_bad", sig, ["PULSE"], comments=["<age>: NaN <sex>: ?"])
    rep = validate_record(path)
    hits = [i for i in rep.issues if i.code == "METADATA_MISSING"]
    assert len(hits) >= 1


def test_nan_present_reports_every_signal(tmp_path):
    a = np.ones(50)
    a[3] = np.nan
    b = np.ones(50)
    b[7] = np.nan
    b[8] = np.nan
    path = _write_record(tmp_path, "nan_rec", np.column_stack([np.ones(50), a, b]), ["HR,", "PULSE,", "SpO2,"])
    rep = validate_record(path)
    nan_issues = [i for i in rep.issues if i.code == "NAN_PRESENT"]
    assert sorted(i.field for i in nan_issues) == ["PULSE", "SpO2"]
    assert all(i.severity == "warning" for i in nan_issues)
    by_field = {i.field: i for i in nan_issues}
    for field in ("PULSE", "SpO2"):
        assert "fraction" in by_field[field].message


@pytest.mark.realdata
def test_realdata_bidmc01n_nan_per_signal():
    if not os.path.exists("data/raw/bidmc/bidmc01n.hea"):
        pytest.skip("real data missing")
    import re

    rep = validate_record("data/raw/bidmc/bidmc01n")
    assert rep.status == "valid"
    nan_issues = [i for i in rep.issues if i.code == "NAN_PRESENT"]
    by_field = {i.field: i for i in nan_issues}
    assert set(by_field) == {"PULSE", "SpO2"}
    for field in ("PULSE", "SpO2"):
        frac = float(re.search(r"fraction=([0-9.eE+-]+)", by_field[field].message).group(1))
        assert frac == pytest.approx(13 / 481, abs=1e-6)


@pytest.mark.realdata
def test_realdata_validate_all():
    cases = ["data/raw/mitdb/100", "data/raw/mitdb/201", "data/raw/bidmc/bidmc01", "data/raw/bidmc/bidmc01n"]
    for c in cases:
        hea = c + ".hea"
        if not os.path.exists(hea):
            pytest.skip("real data missing")
        rep = validate_record(c)
        assert rep.status == "valid", c
