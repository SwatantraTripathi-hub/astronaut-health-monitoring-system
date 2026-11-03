"""RMSSD with a rhythm-label eligibility gate.

The gate is a conservative rule of this prototype: it does not say the rhythm is
normal, only that the label is one the prototype is willing to compute on.
"""

from dataclasses import dataclass

import numpy as np

from src.features.rr import nn_interval_mask, rr_intervals
from src.ingest.rhythm import rhythm_label_at

ELIGIBLE_RHYTHM_LABELS = {"(N"}  # conservative eligibility gate, not an interpretation.
# Only the label seen on record 100 is eligible. Owner must confirm the meaning in the MIT-BIH docs.


@dataclass
class HrvWindowResult:
    status: str  # "evaluated","not_evaluated"
    value_ms: float | None
    n_used: int
    n_possible: int
    reason: str


def _beat_labels(beat_samples, segments):
    return [rhythm_label_at(segments, int(b)) for b in beat_samples]


def _blocking_labels(labels):
    seen = []
    for label in labels:
        if label is None or label in ELIGIBLE_RHYTHM_LABELS:
            continue
        if label not in seen:
            seen.append(label)
    return seen


def rmssd_gated(beat_samples, symbols, fs, segments, *, min_differences, min_coverage):
    """RMSSD over NN intervals that also sit inside an eligible rhythm segment.

    An interval counts only if both of its beats are "N" and both lie in a
    segment whose label is in ELIGIBLE_RHYTHM_LABELS. A difference counts only
    if both of its intervals count.

    min_differences and min_coverage are design parameters, no clinical source.
    """
    if min_differences is None or min_coverage is None:
        return HrvWindowResult(
            status="not_evaluated", value_ms=None, n_used=0, n_possible=0,
            reason="min_differences and min_coverage are required keyword arguments",
        )
    try:
        min_d = int(min_differences)
        min_c = float(min_coverage)
    except Exception:
        return HrvWindowResult(
            status="not_evaluated", value_ms=None, n_used=0, n_possible=0,
            reason="invalid min_differences or min_coverage",
        )

    beats = list(beat_samples)
    symbols = list(symbols)
    if len(beats) < 2:
        return HrvWindowResult(
            status="not_evaluated", value_ms=None, n_used=0, n_possible=0,
            reason="insufficient beats",
        )
    if not segments:
        return HrvWindowResult(
            status="not_evaluated", value_ms=None, n_used=0, n_possible=0,
            reason="no rhythm segments available, rhythm eligibility unknown",
        )

    rr = rr_intervals(beats, fs)
    diffs = np.diff(rr)
    n_possible = int(len(diffs))
    if n_possible == 0:
        return HrvWindowResult(
            status="not_evaluated", value_ms=None, n_used=0, n_possible=0,
            reason="insufficient intervals",
        )

    labels = _beat_labels(beats, segments)
    blocking = _blocking_labels(labels)
    nn_mask = nn_interval_mask(symbols)

    interval_ok = []
    for i in range(len(rr)):
        ok = i < len(nn_mask) and nn_mask[i]
        ok = ok and labels[i] in ELIGIBLE_RHYTHM_LABELS
        ok = ok and labels[i + 1] in ELIGIBLE_RHYTHM_LABELS
        interval_ok.append(bool(ok))

    diff_ok = np.array(
        [interval_ok[i] and interval_ok[i + 1] for i in range(n_possible)],
        dtype=bool,
    )
    n_used = int(diff_ok.sum())

    if n_used < min_d:
        if blocking:
            reason = (
                f"rhythm label '{blocking[0]}' is not in the eligibility gate "
                f"(configured condition, not a diagnosis)"
            )
        else:
            reason = (
                f"fewer than min_differences usable NN-paired differences "
                f"({n_used} used of {n_possible} possible)"
            )
        return HrvWindowResult(
            status="not_evaluated", value_ms=None, n_used=n_used,
            n_possible=n_possible, reason=reason,
        )

    coverage = n_used / n_possible
    if coverage < min_c:
        reason = (
            f"coverage {n_used}/{n_possible} = {coverage:.2f} is below "
            f"min_coverage {min_c} (design parameter, no clinical source)"
        )
        if blocking:
            reason = reason + f"; rhythm label '{blocking[0]}' is also outside the eligibility gate"
        return HrvWindowResult(
            status="not_evaluated", value_ms=None, n_used=n_used,
            n_possible=n_possible, reason=reason,
        )

    value = float(np.sqrt(np.mean(diffs[diff_ok] ** 2))) * 1000.0
    return HrvWindowResult(
        status="evaluated", value_ms=value, n_used=n_used,
        n_possible=n_possible, reason="",
    )
