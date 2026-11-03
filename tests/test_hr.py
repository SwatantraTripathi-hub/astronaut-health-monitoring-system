import pytest

from src.features.hr import HrWindow, hr_windows

FS = 1000


def _beats(spacing_s, n, start_s=0.0):
    return [int(round((start_s + i * spacing_s) * FS)) for i in range(n)]


def test_constant_rr_gives_expected_bpm():
    # 0.8 s between beats -> 75 beats per minute
    beats = _beats(0.8, 13)
    windows = hr_windows(beats, FS, duration_s=10.0, window_s=10.0, step_s=10.0)
    assert len(windows) == 1
    win = windows[0]
    assert isinstance(win, HrWindow)
    assert win.status == "evaluated"
    assert win.hr_bpm == pytest.approx(75.0, abs=1e-6)
    assert win.n_beats == 13
    assert win.start_s == 0.0
    assert win.end_s == 10.0


def test_window_with_one_beat_is_not_evaluated():
    windows = hr_windows([int(0.5 * FS)], FS, duration_s=10.0, window_s=10.0, step_s=10.0)
    assert len(windows) == 1
    assert windows[0].status == "not_evaluated"
    assert windows[0].hr_bpm is None
    assert windows[0].n_beats == 1
    assert windows[0].reason


def test_windows_step_and_clip_to_duration():
    beats = _beats(0.8, 40)
    windows = hr_windows(beats, FS, duration_s=25.0, window_s=10.0, step_s=10.0)
    assert [w.start_s for w in windows] == [0.0, 10.0, 20.0]
    assert windows[-1].end_s == 25.0


def test_only_beats_inside_the_window_count():
    beats = _beats(1.0, 30)
    windows = hr_windows(beats, FS, duration_s=20.0, window_s=10.0, step_s=10.0)
    assert [w.n_beats for w in windows] == [10, 10]
    for win in windows:
        assert win.hr_bpm == pytest.approx(60.0, abs=1e-6)


def test_empty_beats_not_evaluated():
    windows = hr_windows([], FS, duration_s=10.0, window_s=10.0, step_s=10.0)
    assert windows and all(w.status == "not_evaluated" for w in windows)
    assert all(w.hr_bpm is None for w in windows)
