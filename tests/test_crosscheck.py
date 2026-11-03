import os

import numpy as np
import pytest

from src.config import load_params
from src.features.crosscheck import CrosscheckResult, crosscheck_record, numeric_window_mean


def test_numeric_window_mean_ignores_nan():
    values = np.array([60.0, np.nan, 64.0, 62.0])
    mean, n_valid, n_total = numeric_window_mean(values, 1.0, 0.0, 4.0, 0.5)
    assert n_total == 4
    assert n_valid == 3
    assert mean == pytest.approx((60.0 + 64.0 + 62.0) / 3)


def test_numeric_window_mean_below_valid_fraction_is_none():
    values = np.array([60.0, np.nan, np.nan, 60.0])
    mean, n_valid, n_total = numeric_window_mean(values, 1.0, 0.0, 4.0, 0.75)
    assert (mean, n_valid, n_total) == (None, 2, 4)


@pytest.mark.realdata
def test_realdata_bidmc01_crosscheck_runs():
    if not os.path.exists("data/raw/bidmc/bidmc01.hea"):
        pytest.skip("real data missing")
    result = crosscheck_record("data/raw/bidmc/bidmc01", load_params())
    assert isinstance(result, CrosscheckResult)
    assert result.status == "evaluated"
    assert result.median_abs_diff_bpm is not None
    assert result.median_abs_diff_bpm >= 0
    assert result.n_compared > 0
