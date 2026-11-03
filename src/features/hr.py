"""Heart rate from beat times, per window.

Windows are half-open [start_s, end_s). HR is 60 / mean RR of the intervals
whose two beats both lie inside the window. Fewer than two beats means
"not_evaluated"; nothing is ever reported as normal or as zero.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HrWindow:
    start_s: float
    end_s: float
    hr_bpm: float | None
    n_beats: int
    status: str  # "evaluated", "not_evaluated"
    reason: str


def hr_windows(beat_samples, fs, duration_s, window_s, step_s):
    beats = np.asarray(beat_samples, dtype=float)
    fs = float(fs)
    duration_s = float(duration_s)
    window_s = float(window_s)
    step_s = float(step_s)
    if fs <= 0 or window_s <= 0 or step_s <= 0 or duration_s <= 0:
        return []

    windows = []
    start = 0.0
    while start < duration_s - 1e-9:
        end = min(start + window_s, duration_s)
        times = beats / fs
        inside = np.sort(times[(times >= start) & (times < end - 1e-12)])
        n_beats = int(inside.size)
        if n_beats < 2:
            windows.append(
                HrWindow(
                    start_s=start,
                    end_s=end,
                    hr_bpm=None,
                    n_beats=n_beats,
                    status="not_evaluated",
                    reason="fewer than two beats in window",
                )
            )
        else:
            rr = np.diff(inside)
            mean_rr = float(np.mean(rr))
            if mean_rr <= 0:
                windows.append(
                    HrWindow(
                        start_s=start,
                        end_s=end,
                        hr_bpm=None,
                        n_beats=n_beats,
                        status="not_evaluated",
                        reason="non-positive mean RR",
                    )
                )
            else:
                windows.append(
                    HrWindow(
                        start_s=start,
                        end_s=end,
                        hr_bpm=60.0 / mean_rr,
                        n_beats=n_beats,
                        status="evaluated",
                        reason="",
                    )
                )
        start = start + step_s
    return windows
