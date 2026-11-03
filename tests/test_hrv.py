import os

import numpy as np
import pytest
import wfdb

from src.features.hrv import ELIGIBLE_RHYTHM_LABELS, HrvWindowResult, rmssd_gated
from src.features.rr import rmssd_nn
from src.ingest.rhythm import RhythmSegment, rhythm_segments
from src.ingest.wfdb_loader import beat_only, load_beat_annotations

FS = 1000
# beat times 0.0, 0.8, 1.7, 2.3, 3.3, 4.1, 5.0 seconds
BEAT_TIMES_S = [0.0, 0.8, 1.7, 2.3, 3.3, 4.1, 5.0]
SYMBOLS = ["N", "N", "N", "A", "N", "N", "N"]


def _beats():
    return [int(round(t * FS)) for t in BEAT_TIMES_S]


def test_eligible_labels_is_conservative():
    assert ELIGIBLE_RHYTHM_LABELS == {"(N"}


def test_gated_rmssd_simulated_n_rhythm():
    res = rmssd_gated(
        _beats(),
        SYMBOLS,
        FS,
        [RhythmSegment(start_sample=0, end_sample=10 * FS, label="(N")],
        min_differences=1,
        min_coverage=0.1,
    )
    assert isinstance(res, HrvWindowResult)
    assert res.status == "evaluated"
    assert res.value_ms == pytest.approx(100, abs=1e-6)
    assert res.n_used == 2
    assert res.n_possible == 5


def test_gated_rmssd_low_coverage_not_evaluated():
    # 2 used of 5 possible = 0.4 coverage, below the required 0.5
    res = rmssd_gated(
        _beats(),
        SYMBOLS,
        FS,
        [RhythmSegment(start_sample=0, end_sample=10 * FS, label="(N")],
        min_differences=1,
        min_coverage=0.5,
    )
    assert res.status == "not_evaluated"
    assert res.value_ms is None
    assert res.n_used == 2
    assert res.n_possible == 5
    assert "coverage" in res.reason


def test_gated_rmssd_ineligible_rhythm_label_not_evaluated():
    res = rmssd_gated(
        _beats(),
        SYMBOLS,
        FS,
        [RhythmSegment(start_sample=0, end_sample=10 * FS, label="(AFIB")],
        min_differences=1,
        min_coverage=0.5,
    )
    assert res.status == "not_evaluated"
    assert res.value_ms is None
    assert res.n_used == 0
    assert "(AFIB" in res.reason


def test_gated_rmssd_no_segments_not_evaluated():
    res = rmssd_gated(_beats(), SYMBOLS, FS, [], min_differences=1, min_coverage=0.5)
    assert res.status == "not_evaluated"
    assert res.value_ms is None


def test_gated_rmssd_min_differences_not_met():
    res = rmssd_gated(
        _beats(),
        SYMBOLS,
        FS,
        [RhythmSegment(start_sample=0, end_sample=10 * FS, label="(N")],
        min_differences=3,
        min_coverage=0.1,
    )
    assert res.status == "not_evaluated"
    assert res.value_ms is None
    assert res.n_used == 2


def _first_ten_seconds(path):
    samples, symbols, aux_notes = load_beat_annotations(path)
    sig_len = wfdb.rdheader(path).sig_len
    fs = wfdb.rdheader(path).fs
    n = int(10 * fs)
    idx = [i for i in range(len(samples)) if samples[i] < n]
    beat_samples, beat_symbols = beat_only(samples[idx], [symbols[i] for i in idx])
    segments = rhythm_segments(samples, symbols, aux_notes, sig_len)
    return beat_samples, beat_symbols, fs, segments


@pytest.mark.realdata
def test_realdata_100_gated_rmssd():
    if not os.path.exists("data/raw/mitdb/100.hea"):
        pytest.skip("real data missing")
    beat_samples, symbols, fs, segments = _first_ten_seconds("data/raw/mitdb/100")
    res = rmssd_gated(beat_samples, symbols, fs, segments, min_differences=1, min_coverage=0.5)
    assert res.status == "evaluated"
    assert res.value_ms == pytest.approx(19.91, abs=0.01)
    assert res.n_used == 8
    assert res.n_possible == 11


@pytest.mark.realdata
def test_realdata_201_gated_rmssd_is_not_evaluated():
    if not os.path.exists("data/raw/mitdb/201.hea"):
        pytest.skip("real data missing")
    beat_samples, symbols, fs, segments = _first_ten_seconds("data/raw/mitdb/201")
    res = rmssd_gated(beat_samples, symbols, fs, segments, min_differences=1, min_coverage=0.5)
    assert res.status == "not_evaluated"
    assert res.value_ms is None
    assert "(AFIB" in res.reason
    # Contrast kept on purpose: the ungated rmssd_nn (existing, untouched test)
    # still returns 159.54 ms for record 201 even though that stretch is annotated "(AFIB".
    assert rmssd_nn(beat_samples, symbols, fs, min_differences=1).value_ms == pytest.approx(159.54, abs=0.01)
