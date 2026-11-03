"""Rhythm annotations -> segments, so an RMSSD value can be tied to an annotated rhythm.

Labels are copied from the dataset annotation file exactly as written (e.g. "(AFIB").
This module does not interpret them.
"""

from dataclasses import dataclass
from typing import List


def clean_note(note) -> str:
    """Strip the NUL terminator wfdb adds and surrounding whitespace."""
    if note is None:
        return ""
    return str(note).replace("\x00", "").strip()


@dataclass
class RhythmSegment:
    start_sample: int
    end_sample: int
    label: str


def rhythm_segments(samples, symbols, aux_notes, sig_len) -> List[RhythmSegment]:
    """One segment per "+" (rhythm change) annotation.

    A segment runs from its "+" sample to the next "+" sample, or to sig_len
    for the last one. No "+" at all gives an empty list.
    """
    starts = []
    for i, sym in enumerate(symbols):
        if sym != "+":
            continue
        sample = int(samples[i])
        note = aux_notes[i] if i < len(aux_notes) else ""
        starts.append((sample, clean_note(note)))

    segments = []
    for j, (sample, label) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else int(sig_len)
        segments.append(RhythmSegment(start_sample=sample, end_sample=end, label=label))
    return segments


def rhythm_label_at(segments, sample):
    """Label of the segment containing sample, or None if it is not covered."""
    sample = int(sample)
    for seg in segments:
        if seg.start_sample <= sample < seg.end_sample:
            return seg.label
    return None
