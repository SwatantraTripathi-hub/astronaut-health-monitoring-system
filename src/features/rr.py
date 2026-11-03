from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RmssdResult:
    status: str  # "evaluated","not_evaluated"
    value_ms: float | None
    n_differences: int
    reason: str


def rr_intervals(beat_samples, fs):
    beat_samples = np.array(beat_samples)
    if len(beat_samples) < 2:
        return np.array([])
    beat_t = beat_samples.astype(float) / float(fs)
    rr = np.diff(beat_t)
    return rr


def nn_interval_mask(symbols) -> List[bool]:
    symbols = list(symbols)
    mask = []
    for a, b in zip(symbols[:-1], symbols[1:]):
        if a == "N" and b == "N":
            mask.append(True)
        else:
            mask.append(False)
    return mask


def rmssd_nn(beat_samples, symbols, fs, *, min_differences):
    if min_differences is None:
        return RmssdResult(status="not_evaluated", value_ms=None, n_differences=0, reason="min_differences required")
    try:
        min_d = int(min_differences)
    except Exception:
        return RmssdResult(status="not_evaluated", value_ms=None, n_differences=0, reason="invalid min_differences")

    rr = rr_intervals(beat_samples, fs)
    if len(rr) == 0:
        return RmssdResult(status="not_evaluated", value_ms=None, n_differences=0, reason="insufficient beats")
    nn_mask = nn_interval_mask(symbols)
    diffs = np.diff(rr)
    if len(diffs) == 0:
        return RmssdResult(status="not_evaluated", value_ms=None, n_differences=0, reason="insufficient intervals")
    valid = []
    for i in range(len(diffs)):
        if i < len(nn_mask) and i + 1 < len(nn_mask):
            if nn_mask[i] and nn_mask[i + 1]:
                valid.append(True)
            else:
                valid.append(False)
        else:
            valid.append(False)
    valid_arr = np.array(valid, dtype=bool)
    n_differences = int(valid_arr.sum())
    if n_differences < min_d:
        return RmssdResult(status="not_evaluated", value_ms=None, n_differences=n_differences, reason="fewer than min_differences NN-paired differences")
    rmssd = float(np.sqrt(np.mean(diffs[valid_arr] ** 2))) * 1000.0
    return RmssdResult(status="evaluated", value_ms=rmssd, n_differences=n_differences, reason="")
