"""Plausibility cross-check: ECG-derived heart rate vs the monitor's HR numerics.

This compares two signals recorded on the same BIDMC record. It is a
plausibility check on the prototype detector, not an accuracy validation, and
it says nothing about a person's health.
"""

from dataclasses import dataclass, field

import numpy as np

from src.features.rpeaks import detect_rpeaks
from src.ingest.wfdb_loader import get_signal, load_record


@dataclass
class CrosscheckResult:
    status: str  # "evaluated", "not_evaluated"
    median_abs_diff_bpm: float | None
    n_compared: int
    n_skipped: int
    window_s: float
    reason: str = ""
    skipped_reasons: dict = field(default_factory=dict)


def numeric_window_mean(values, fs, start_s, end_s, min_valid_fraction):
    """Mean of non-NaN samples whose time lies in [start_s, end_s).

    Returns (mean|None, n_valid, n_total). NaN counts as missing, never as 0,
    and a window is refused when its valid fraction is below the parameter.
    """
    values = np.asarray(values, dtype=float)
    times = np.arange(values.size) / float(fs)
    mask = (times >= start_s) & (times < end_s)
    window_values = values[mask]
    n_total = int(window_values.size)
    finite = window_values[np.isfinite(window_values)]
    n_valid = int(finite.size)
    if n_total == 0 or (n_valid / n_total) < float(min_valid_fraction):
        return None, n_valid, n_total
    return float(np.mean(finite)), n_valid, n_total


def crosscheck_record(base_path, params, *, waveform_lead="II", numeric_lead="HR"):
    """Compare detector HR against monitor HR numerics on one BIDMC record."""
    params = params
    window_s = float(params["crosscheck"]["window_s"])
    min_valid = float(params["numerics"]["min_valid_fraction"])

    waveform = load_record(base_path)
    numerics = load_record(f"{base_path}n")
    ecg = get_signal(waveform, waveform_lead)
    monitor_hr = get_signal(numerics, numeric_lead)

    detected = detect_rpeaks(ecg, waveform.fs, params["rpeaks"])
    beat_times = detected.astype(float) / float(waveform.fs)

    waveform_duration = waveform.sig_len / float(waveform.fs)
    numerics_duration = numerics.sig_len / float(numerics.fs)
    # Only windows fully inside both recordings are used: the waveform is
    # 60001 samples at 125 Hz (480.008 s) and the numerics 481 samples at 1 Hz,
    # so the last window must end at or before 480 s.
    limit = min(waveform_duration, numerics_duration)

    skipped = {}
    differences = []
    start = 0.0
    while start + window_s <= limit + 1e-9:
        end = start + window_s
        inside = beat_times[(beat_times >= start) & (beat_times < end)]
        if inside.size < 2:
            skipped["fewer than two detected beats"] = skipped.get("fewer than two detected beats", 0) + 1
            start += window_s
            continue
        waveform_hr = 60.0 / float(np.mean(np.diff(inside)))
        monitor_mean, n_valid, n_total = numeric_window_mean(
            monitor_hr, numerics.fs, start, end, min_valid
        )
        if monitor_mean is None:
            key = f"monitor HR has fewer than {min_valid:.0%} valid samples"
            skipped[key] = skipped.get(key, 0) + 1
        else:
            differences.append(abs(waveform_hr - monitor_mean))
        start += window_s

    n_compared = len(differences)
    n_skipped = int(sum(skipped.values()))
    if n_compared == 0:
        return CrosscheckResult(
            status="not_evaluated",
            median_abs_diff_bpm=None,
            n_compared=0,
            n_skipped=n_skipped,
            window_s=window_s,
            reason="no window had both a detector HR and a usable monitor HR",
            skipped_reasons=skipped,
        )
    return CrosscheckResult(
        status="evaluated",
        median_abs_diff_bpm=float(np.median(differences)),
        n_compared=n_compared,
        n_skipped=n_skipped,
        window_s=window_s,
        skipped_reasons=skipped,
    )


def crosscheck_all(raw_dir, params, *, waveform_lead="II", numeric_lead="HR"):
    """Run the cross-check on every BIDMC record pair that exists locally."""
    from pathlib import Path

    base = Path(raw_dir)
    results = {}
    for hea in sorted(base.glob("*.hea")):
        stem = hea.stem
        if stem.endswith("n"):
            continue
        if not (base / f"{stem}n.hea").exists():
            continue
        results[stem] = crosscheck_record(
            str(base / stem), params, waveform_lead=waveform_lead, numeric_lead=numeric_lead
        )
    return results
