import os

import numpy as np
import pytest

from src.features.rr import RmssdResult, nn_interval_mask, rmssd_nn, rr_intervals
from src.ingest.wfdb_loader import beat_only, load_beat_annotations


def test_rmssd_example():
    fs = 1000
    # beat times 0.0, 0.8, 1.7, 2.3, 3.3, 4.1, 5.0 seconds
    beat_times_s = [0.0, 0.8, 1.7, 2.3, 3.3, 4.1, 5.0]
    beat_samples = (np.array(beat_times_s) * fs).astype(int)
    symbols = ["N", "N", "N", "A", "N", "N", "N"]
    res = rmssd_nn(beat_samples, symbols, fs, min_differences=1)
    assert res.status == "evaluated"
    assert res.value_ms == pytest.approx(100, abs=1e-6)
    assert res.n_differences == 2


def test_all_a_not_evaluated():
    fs = 1000
    beat_times_s = [0.0, 0.8, 1.7, 2.3]
    beat_samples = (np.array(beat_times_s) * fs).astype(int)
    symbols = ["A", "A", "A", "A"]
    res = rmssd_nn(beat_samples, symbols, fs, min_differences=1)
    assert res.status == "not_evaluated"
    assert res.value_ms is None


def test_min_differences_too_high():
    fs = 1000
    beat_times_s = [0.0, 0.8, 1.7, 2.3, 3.3]
    beat_samples = (np.array(beat_times_s) * fs).astype(int)
    symbols = ["N", "N", "N", "N", "N"]
    res = rmssd_nn(beat_samples, symbols, fs, min_differences=10)
    assert res.status == "not_evaluated"
    assert res.value_ms is None


def test_empty_input_not_evaluated():
    res = rmssd_nn([], [], 360, min_differences=1)
    assert res.status == "not_evaluated"
    assert res.value_ms is None


@pytest.mark.realdata
def test_realdata_mitdb100_rmssd():
    if not os.path.exists("data/raw/mitdb/100.hea"):
        pytest.skip("real data missing")
    import wfdb

    rec = wfdb.rdrecord("data/raw/mitdb/100")
    ann = wfdb.rdann("data/raw/mitdb/100", "atr")
    fs = rec.fs
    n = 10 * fs
    beat_idx = [i for i, sym in enumerate(ann.symbol) if sym != "+" and ann.sample[i] < n]
    beat_samples = np.array([ann.sample[i] for i in beat_idx])
    syms = [ann.symbol[i] for i in beat_idx]
    # also apply beat_only logic as per loader style
    beat_samples_b, syms_b = beat_only(beat_samples, syms) if False else (beat_samples, syms)  # keep as in plot logic (they already filtered +)
    res = rmssd_nn(beat_samples, syms, fs, min_differences=1)
    assert res.status == "evaluated"
    assert res.value_ms == pytest.approx(19.91, abs=0.01)
    assert res.n_differences == 8


@pytest.mark.realdata
def test_realdata_mitdb201_rmssd():
    if not os.path.exists("data/raw/mitdb/201.hea"):
        pytest.skip("real data missing")
    import wfdb

    rec = wfdb.rdrecord("data/raw/mitdb/201")
    ann = wfdb.rdann("data/raw/mitdb/201", "atr")
    fs = rec.fs
    n = 10 * fs
    beat_idx = [i for i, sym in enumerate(ann.symbol) if sym != "+" and ann.sample[i] < n]
    beat_samples = np.array([ann.sample[i] for i in beat_idx])
    syms = [ann.symbol[i] for i in beat_idx]
    res = rmssd_nn(beat_samples, syms, fs, min_differences=1)
    assert res.status == "evaluated"
    assert res.value_ms == pytest.approx(159.54, abs=0.01)
    assert res.n_differences == 12
