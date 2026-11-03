"""Experimental R-peak detector and a beat-matching helper.

Pipeline (documented on purpose, in this order):
bandpass -> derivative -> square -> moving-average integration -> local maxima
with a refractory period -> refine each candidate to the maximum |filtered|.

Every number that shapes the detector comes from configs/params.yaml; nothing is
tuned to a specific record.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import butter, filtfilt, find_peaks

# Refinement is +/- 50 ms around the candidate: the detector's energy peak can
# sit slightly off the R wave, and 50 ms is short compared with an RR interval.
REFINE_HALF_WINDOW_S = 0.05


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    sensitivity: float | None
    ppv: float | None


def _bandpass(signal, fs, low_hz, high_hz):
    nyquist = fs / 2.0
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 0.99)
    if not 0 < low < high < 1:
        raise ValueError(f"invalid bandpass {low_hz}-{high_hz} Hz for fs={fs}")
    b, a = butter(2, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_rpeaks(signal, fs, params):
    """Return R-peak sample indices (ints) for one signal.

    `params` is the `rpeaks` mapping from configs/params.yaml.
    """
    x = np.asarray(signal, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x) if x.size else x
    if x.size < 2:
        return np.array([], dtype=int)

    fs = float(fs)
    filtered = _bandpass(
        x, fs, params["bandpass_low_hz"], params["bandpass_high_hz"])

    derivative = np.diff(filtered, prepend=filtered[0])
    energy = derivative ** 2

    integration = max(1, int(round(params["integration_ms"] * fs / 1000.0)))
    kernel = np.ones(integration) / integration
    integrated = np.convolve(energy, kernel, mode="same")

    if not np.any(integrated > 0):
        return np.array([], dtype=int)

    # The threshold follows a running maximum, not a single global maximum:
    # in record 100 the R-wave energy varies several-fold over the record, so a
    # global fraction hid most beats. Both the window and the fraction are
    # design parameters from configs/params.yaml.
    run_window = max(1, int(round(params["threshold_window_s"] * fs))) | 1
    running_max = maximum_filter(integrated, size=run_window, mode="nearest")
    threshold = params["threshold_fraction"] * running_max
    distance = max(1, int(round(params["refractory_ms"] * fs / 1000.0)))
    candidates, _ = find_peaks(integrated, height=threshold, distance=distance)

    half = max(1, int(round(REFINE_HALF_WINDOW_S * fs)))
    refined = []
    for candidate in candidates:
        lo = max(0, int(candidate) - half)
        hi = min(filtered.size, int(candidate) + half + 1)
        if hi <= lo:
            continue
        refined.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))

    if not refined:
        return np.array([], dtype=int)
    peaks = np.unique(np.array(refined, dtype=int))

    # Enforce the refractory period again after refinement.
    kept = [int(peaks[0])]
    for peak in peaks[1:]:
        if peak - kept[-1] >= distance:
            kept.append(int(peak))
        elif abs(filtered[peak]) > abs(filtered[kept[-1]]):
            kept[-1] = int(peak)
    return np.array(kept, dtype=int)


def match_beats(detected, reference, fs, tol_ms):
    """Greedy one-to-one matching of detected beats to reference beats.

    sensitivity = TP / (TP + FN), ppv = TP / (TP + FP); None if the
    denominator is 0, so "nothing to measure" never looks like a score.
    """
    detected = np.sort(np.asarray(detected, dtype=float))
    reference = np.sort(np.asarray(reference, dtype=float))
    tolerance = float(tol_ms) * float(fs) / 1000.0

    used = np.zeros(reference.size, dtype=bool)
    tp = 0
    fp = 0
    for beat in detected:
        if reference.size == 0:
            fp += 1
            continue
        distances = np.abs(reference - beat)
        distances[used] = np.inf
        idx = int(np.argmin(distances))
        if np.isfinite(distances[idx]) and distances[idx] <= tolerance:
            used[idx] = True
            tp += 1
        else:
            fp += 1
    fn = int(reference.size - used.sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    return MatchResult(tp=tp, fp=fp, fn=fn, sensitivity=sensitivity, ppv=ppv)
