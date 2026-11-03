"""Tests for the prototype R-peak detector. SIMULATED signals only (plus matching)."""

import numpy as np
import pytest

from src.features.rpeaks import MatchResult, detect_rpeaks, match_beats

FS = 250
PARAMS = {
    "bandpass_low_hz": 5,
    "bandpass_high_hz": 15,
    "integration_ms": 150,
    "refractory_ms": 200,
    "match_tolerance_ms": 150,
    "threshold_fraction": 0.3,
    "threshold_window_s": 1.0,
}


def simulated_ecg(peak_times_s, fs=FS, duration_s=30.0, width_s=0.008):
    """SIMULATED narrow spikes at known times; not a physiological ECG model."""
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    signal = np.zeros(n)
    for pt in peak_times_s:
        signal += np.exp(-0.5 * ((t - pt) / width_s) ** 2)
    return signal


def test_simulated_pulse_train_detected_within_tolerance():
    true_times = [0.5 + 0.8 * i for i in range(36)]
    signal = simulated_ecg(true_times)
    detected = detect_rpeaks(signal, FS, PARAMS)
    assert len(detected) > 0
    result = match_beats(detected, [round(t * FS) for t in true_times], FS, PARAMS["match_tolerance_ms"])
    assert isinstance(result, MatchResult)
    assert result.tp == len(true_times)
    assert result.fp == 0
    assert result.fn == 0
    assert result.sensitivity == pytest.approx(1.0)
    assert result.ppv == pytest.approx(1.0)


def test_match_beats_hand_made_example():
    result = match_beats([100, 200, 300], [100, 300], 1000, 50)
    assert (result.tp, result.fp, result.fn) == (2, 1, 0)
    assert result.sensitivity == pytest.approx(1.0)
    assert result.ppv == pytest.approx(2 / 3)


def test_match_beats_empty_denominators_are_none():
    result = match_beats([], [], 1000, 50)
    assert result.tp == 0 and result.fp == 0 and result.fn == 0
    assert result.sensitivity is None
    assert result.ppv is None


def test_match_beats_outside_tolerance_is_missed():
    result = match_beats([500], [100], 1000, 50)
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)


def test_no_peaks_on_flat_signal():
    detected = detect_rpeaks(np.zeros(FS * 5), FS, PARAMS)
    assert len(detected) == 0
