import os

import pytest
import wfdb

from src.ingest.rhythm import RhythmSegment, clean_note, rhythm_label_at, rhythm_segments
from src.ingest.wfdb_loader import load_beat_annotations


def test_clean_note_strips_nul_and_whitespace():
    assert clean_note("(AFIB\x00") == "(AFIB"
    assert clean_note("  (N\x00  ") == "(N"
    assert clean_note(None) == ""


def test_segments_from_simulated_annotations():
    # parallel annotation arrays, like wfdb.rdann returns
    samples = [10, 200, 300, 800, 900]
    symbols = ["+", "N", "+", "N", "+"]
    aux_notes = ["(N\x00", "", "(AFIB\x00", "", "(N\x00"]
    segs = rhythm_segments(samples, symbols, aux_notes, sig_len=1000)
    assert segs == [
        RhythmSegment(start_sample=10, end_sample=300, label="(N"),
        RhythmSegment(start_sample=300, end_sample=900, label="(AFIB"),
        RhythmSegment(start_sample=900, end_sample=1000, label="(N"),
    ]


def test_label_at_sample_boundaries():
    segs = [
        RhythmSegment(start_sample=10, end_sample=300, label="(N"),
        RhythmSegment(start_sample=300, end_sample=900, label="(AFIB"),
    ]
    assert rhythm_label_at(segs, 10) == "(N"
    assert rhythm_label_at(segs, 299) == "(N"
    assert rhythm_label_at(segs, 300) == "(AFIB"
    assert rhythm_label_at(segs, 899) == "(AFIB"


def test_label_before_first_segment_is_none():
    segs = [RhythmSegment(start_sample=10, end_sample=100, label="(N")]
    assert rhythm_label_at(segs, 0) is None
    assert rhythm_label_at(segs, 9) is None


def test_no_rhythm_markers_gives_empty_list():
    assert rhythm_segments([], [], [], sig_len=1000) == []
    assert rhythm_segments([5, 9], ["N", "N"], ["", ""], sig_len=1000) == []


@pytest.mark.realdata
def test_realdata_201_first_rhythm_segment():
    if not os.path.exists("data/raw/mitdb/201.hea"):
        pytest.skip("real data missing")
    samples, symbols, aux_notes = load_beat_annotations("data/raw/mitdb/201")
    sig_len = wfdb.rdheader("data/raw/mitdb/201").sig_len
    segs = rhythm_segments(samples, symbols, aux_notes, sig_len)
    assert len(segs) > 0
    assert segs[0].label == "(AFIB"
    assert segs[0].start_sample == 60


@pytest.mark.realdata
def test_realdata_100_first_rhythm_segment():
    if not os.path.exists("data/raw/mitdb/100.hea"):
        pytest.skip("real data missing")
    samples, symbols, aux_notes = load_beat_annotations("data/raw/mitdb/100")
    sig_len = wfdb.rdheader("data/raw/mitdb/100").sig_len
    segs = rhythm_segments(samples, symbols, aux_notes, sig_len)
    assert len(segs) == 1
    assert segs[0].label == "(N"
    assert segs[0].start_sample == 18
