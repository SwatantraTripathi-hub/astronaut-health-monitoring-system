import re

import pytest

from src.config import REQUIRED_KEYS, ConfigError, load_params


def test_load_params_returns_all_required_keys():
    params = load_params()
    for dotted in REQUIRED_KEYS:
        node = params
        for part in dotted.split("."):
            assert part in node, f"missing key {dotted}"
            node = node[part]
        assert node is not None


def test_required_keys_cover_phase_b_values():
    assert set(REQUIRED_KEYS) >= {
        "hrv.window_s",
        "hrv.min_differences",
        "hrv.min_coverage",
        "hr.window_s",
        "hr.step_s",
        "rpeaks.bandpass_low_hz",
        "rpeaks.bandpass_high_hz",
        "rpeaks.integration_ms",
        "rpeaks.refractory_ms",
        "rpeaks.match_tolerance_ms",
        "rpeaks.threshold_fraction",
        "rpeaks.threshold_window_s",
        "crosscheck.window_s",
        "numerics.min_valid_fraction",
    }


def test_missing_key_raises_config_error(tmp_path):
    text = "hrv:\n  window_s: 300  # design parameter, no clinical source\n"
    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_params(path)


def test_every_value_line_is_labelled_a_design_parameter():
    path = __import__("pathlib").Path("configs/params.yaml")
    lines = path.read_text(encoding="utf-8").splitlines()
    value_lines = [
        line for line in lines
        if re.match(r"^\s+[a-z_]+:\s*\S", line) and not line.strip().startswith("#")
    ]
    assert value_lines, "params.yaml has no values"
    for line in value_lines:
        assert "design parameter, no clinical source" in line, line
