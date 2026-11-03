import json
import os

import pytest

from src.pipeline import (
    LIMITATIONS,
    _rule_results,
    analyze,
    make_simulated_fixture,
)

TOP_KEYS = {
    "dataset",
    "data_quality",
    "rhythm_context",
    "indicators",
    "rule_results",
    "model_results",
    "limitations",
}
DATASET_KEYS = {"name", "record", "version_label", "data_type", "fs", "duration_s"}
INDICATOR_KEYS = {
    "name",
    "unit",
    "source",
    "method",
    "status",
    "n_windows_total",
    "n_windows_evaluated",
    "windows",
}
WINDOW_KEYS = {"start_s", "end_s", "value", "status", "reason"}
STATUSES = {"evaluated", "not_evaluated", "data_issue"}


def _has(record):
    return os.path.exists(f"data/raw/mitdb/{record}.hea")


# ---------------------------------------------------------------- key contract


def test_fixture_top_level_keys_are_exact():
    fixture = make_simulated_fixture()
    assert set(fixture) == TOP_KEYS
    assert set(fixture["dataset"]) == DATASET_KEYS
    assert set(fixture["data_quality"]) == {"status", "issues", "not_checked"}
    assert set(fixture["rhythm_context"]) == {"source", "segments"}
    assert fixture["model_results"] == {"status": "not_run", "warnings": []}
    assert isinstance(fixture["limitations"], list) and fixture["limitations"]


def test_fixture_indicator_and_window_keys_are_exact():
    fixture = make_simulated_fixture()
    assert fixture["indicators"]
    for indicator in fixture["indicators"]:
        assert set(indicator) == INDICATOR_KEYS
        assert indicator["status"] in STATUSES
        for window in indicator["windows"]:
            assert set(window) == WINDOW_KEYS
            assert window["status"] in {"evaluated", "not_evaluated"}


def test_fixture_is_labelled_simulated():
    fixture = make_simulated_fixture()
    assert fixture["dataset"]["data_type"] == "simulated"
    text = json.dumps(fixture).upper()
    assert "SIMULATED" in text
    assert "NORMAL" not in text


def test_fixture_shows_every_dashboard_state():
    fixture = make_simulated_fixture()
    statuses = {indicator["status"] for indicator in fixture["indicators"]}
    assert "evaluated" in statuses
    assert "not_evaluated" in statuses
    assert "data_issue" in statuses
    rule_statuses = {result["status"] for result in fixture["rule_results"]}
    assert "flag" in rule_statuses


def test_fixture_is_json_safe_without_nan():
    json.dumps(make_simulated_fixture(), allow_nan=False)


def _hr_indicator(source, value):
    return {
        "name": "HR",
        "unit": "bpm",
        "source": source,
        "method": "test fixture",
        "status": "evaluated",
        "n_windows_total": 1,
        "n_windows_evaluated": 1,
        "windows": [
            {
                "start_s": 0.0,
                "end_s": 10.0,
                "value": value,
                "status": "evaluated",
                "reason": "",
            }
        ],
    }


def test_rule_flag_does_not_propagate_across_hr_sources():
    indicators = [
        _hr_indicator("annotation file beats", 80.0),
        _hr_indicator("computed from ECG by prototype detector, exploratory", 400.0),
    ]
    rule = {
        "id": "hr_sensor_sanity",
        "indicator": "HR",
        "comparison": "outside_range",
        "value": [15, 300],
        "unit": "bpm",
        "source": "none; engineering choice",
        "status": "plausibility_check",
        "message": "possible sensor/data error, not a health finding",
    }
    result = _rule_results(indicators, [rule], "valid")[0]
    assert result["status"] == "flag"
    assert result["n_windows_evaluated"] == 2
    assert result["n_windows_flagged"] == 1
    assert result["flagged_indicator_sources"] == [
        "computed from ECG by prototype detector, exploratory"
    ]
    assert "annotation file beats" not in result["flagged_indicator_sources"]


def test_fixture_rule_flag_only_marks_the_flagged_source():
    fixture = make_simulated_fixture()
    by_id = {rule["rule_id"]: rule for rule in fixture["rule_results"]}
    sanity = by_id["hr_sensor_sanity"]
    assert sanity["status"] == "flag"
    assert sanity["flagged_indicator_sources"] == ["SIMULATED monitor numerics"]
    demo = by_id["hr_high_demo"]
    assert demo["status"] == "flag"
    assert demo["flagged_indicator_sources"] == ["SIMULATED monitor numerics"]


def test_invalid_bidmc_keeps_all_six_indicator_descriptors(tmp_path):
    (tmp_path / "case.hea").write_text("not a wfdb header", encoding="utf-8")
    (tmp_path / "casen.hea").write_text("not a wfdb header", encoding="utf-8")
    result = analyze(
        str(tmp_path / "case"),
        data_type="actual_research_data",
        source_label="deliberately invalid BIDMC-shaped record",
    )
    assert result["data_quality"]["status"] == "invalid"
    assert len(result["indicators"]) == 6
    assert all(indicator["status"] == "data_issue" for indicator in result["indicators"])
    assert all(indicator["windows"] == [] for indicator in result["indicators"])
    assert [indicator["name"] for indicator in result["indicators"]] == [
        "HR",
        "HR",
        "PULSE",
        "SpO2",
        "RESP",
        "RESP",
    ]
    sources = {indicator["source"] for indicator in result["indicators"]}
    assert "computed from ECG lead II by prototype detector, exploratory" in sources
    assert "computed from RESP waveform, exploratory" in sources
    assert all(rule["status"] == "not_evaluated" for rule in result["rule_results"])
    json.dumps(result, allow_nan=False)


def test_limitations_cover_the_required_points():
    text = " ".join(LIMITATIONS).lower()
    assert "research prototype" in text
    assert "not a diagnosis" in text
    assert "not astronaut data" in text
    assert "no flag does not mean healthy" in text
    assert "design parameter" in text
    assert "dataset annotations" in text


# ---------------------------------------------------------------- invalid input


def test_invalid_input_does_not_raise_and_is_not_evaluated():
    result = analyze(
        "data/raw/mitdb/does_not_exist",
        data_type="actual_research_data",
        source_label="missing file",
    )
    assert set(result) == TOP_KEYS
    assert result["data_quality"]["status"] == "invalid"
    assert all(indicator["status"] == "data_issue" for indicator in result["indicators"])
    assert all(indicator["windows"] == [] for indicator in result["indicators"])
    assert result["rule_results"]
    assert all(rule["status"] == "not_evaluated" for rule in result["rule_results"])
    json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------- MIT-BIH


@pytest.mark.realdata
def test_realdata_100_keys_and_indicators():
    if not _has("100"):
        pytest.skip("real data missing")
    result = analyze(
        "data/raw/mitdb/100",
        data_type="actual_research_data",
        source_label="MIT-BIH Arrhythmia Database, local copy",
    )
    assert set(result) == TOP_KEYS
    assert set(result["dataset"]) == DATASET_KEYS
    assert result["dataset"]["data_type"] == "actual_research_data"
    assert result["data_quality"]["status"] == "valid"

    names = [indicator["name"] for indicator in result["indicators"]]
    assert names.count("HR") == 2
    assert "RMSSD" in names
    for indicator in result["indicators"]:
        assert set(indicator) == INDICATOR_KEYS
        assert indicator["status"] in STATUSES

    sources = {indicator["source"] for indicator in result["indicators"]}
    assert "annotation file beats" in sources
    assert "computed from ECG by prototype detector, exploratory" in sources


@pytest.mark.realdata
def test_realdata_100_rhythm_context_comes_from_the_annotation_file():
    if not _has("100"):
        pytest.skip("real data missing")
    result = analyze(
        "data/raw/mitdb/100", data_type="actual_research_data", source_label="x"
    )
    assert set(result["rhythm_context"]) == {"source", "segments"}
    assert result["rhythm_context"]["segments"]
    for segment in result["rhythm_context"]["segments"]:
        assert set(segment) == {"start_s", "end_s", "label"}
    assert result["rhythm_context"]["segments"][0]["label"] == "(N"


@pytest.mark.realdata
def test_realdata_201_first_rmssd_window_is_not_evaluated_for_afib():
    if not _has("201"):
        pytest.skip("real data missing")
    result = analyze(
        "data/raw/mitdb/201",
        data_type="actual_research_data",
        source_label="MIT-BIH Arrhythmia Database, local copy",
    )
    rmssd = next(i for i in result["indicators"] if i["name"] == "RMSSD")
    first = rmssd["windows"][0]
    assert first["start_s"] == 0
    assert first["end_s"] == 300
    assert first["status"] == "not_evaluated"
    assert first["value"] is None
    assert "(AFIB" in first["reason"]


@pytest.mark.realdata
def test_realdata_100_first_rmssd_window_status_and_reason():
    if not _has("100"):
        pytest.skip("real data missing")
    result = analyze(
        "data/raw/mitdb/100", data_type="actual_research_data", source_label="x"
    )
    rmssd = next(i for i in result["indicators"] if i["name"] == "RMSSD")
    first = rmssd["windows"][0]
    assert first["start_s"] == 0
    assert first["end_s"] == 300
    print("record 100 first RMSSD window:", first["status"], first["reason"])
    assert first["status"] in {"evaluated", "not_evaluated"}
    if first["status"] == "not_evaluated":
        assert first["reason"]
    else:
        assert first["value"] is not None
        assert first["value"] > 0


@pytest.mark.realdata
@pytest.mark.parametrize("record", ["100", "201"])
def test_realdata_json_is_safe_without_nan(record):
    if not _has(record):
        pytest.skip("real data missing")
    result = analyze(
        f"data/raw/mitdb/{record}",
        data_type="actual_research_data",
        source_label="MIT-BIH Arrhythmia Database, local copy",
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.realdata
def test_realdata_rules_use_the_engine_and_flag_separately():
    if not _has("100"):
        pytest.skip("real data missing")
    result = analyze(
        "data/raw/mitdb/100", data_type="actual_research_data", source_label="x"
    )
    rule_ids = {rule["rule_id"] for rule in result["rule_results"]}
    assert rule_ids == {
        "hr_sensor_sanity",
        "spo2_sensor_sanity",
        "resp_sensor_sanity",
        "hr_high_demo",
        "spo2_low_demo",
    }
    for rule in result["rule_results"]:
        assert rule["status"] in {"flag", "no_flag", "not_evaluated"}
        assert "n_windows_evaluated" in rule
        assert "n_windows_flagged" in rule
        assert len(rule["flagged_window_starts"]) <= 20
    # No SpO2 or RESP indicator exists for MIT-BIH, so those rules stay not evaluated.
    by_id = {rule["rule_id"]: rule for rule in result["rule_results"]}
    assert by_id["spo2_sensor_sanity"]["status"] == "not_evaluated"
    assert by_id["resp_sensor_sanity"]["status"] == "not_evaluated"


# ---------------------------------------------------------------- BIDMC

BIDMC = "data/raw/bidmc/bidmc01"
MONITOR_SOURCE = "monitor numerics (1 Hz), numeric, monitor-derived"


def _has_bidmc():
    return os.path.exists(BIDMC + ".hea") and os.path.exists(BIDMC + "n.hea")


def _bidmc_result():
    return analyze(
        BIDMC, data_type="actual_research_data", source_label="BIDMC local copy"
    )


def _by_source(result, name, source):
    return [
        indicator
        for indicator in result["indicators"]
        if indicator["name"] == name and indicator["source"] == source
    ]


@pytest.mark.realdata
def test_realdata_bidmc01_keeps_each_source_separate():
    if not _has_bidmc():
        pytest.skip("real data missing")
    result = _bidmc_result()
    assert set(result) == TOP_KEYS
    assert result["data_quality"]["status"] == "valid"

    for name in ("HR", "PULSE", "SpO2", "RESP"):
        monitor = _by_source(result, name, MONITOR_SOURCE)
        assert len(monitor) == 1, f"{name} monitor indicator missing or duplicated"
    ecg = _by_source(
        result, "HR", "computed from ECG lead II by prototype detector, exploratory"
    )
    assert len(ecg) == 1
    resp_wave = [
        indicator
        for indicator in result["indicators"]
        if indicator["name"] == "RESP" and indicator["source"] != MONITOR_SOURCE
    ]
    assert len(resp_wave) == 1

    for indicator in result["indicators"]:
        assert set(indicator) == INDICATOR_KEYS
        assert indicator["status"] in STATUSES

    units = {}
    for indicator in result["indicators"]:
        if indicator["source"] == MONITOR_SOURCE:
            units[indicator["name"]] = indicator["unit"]
    assert units == {"HR": "bpm", "PULSE": "bpm", "SpO2": "%", "RESP": "pm"}
    assert ecg[0]["unit"] == "bpm"


@pytest.mark.realdata
def test_realdata_bidmc01_monitor_values_are_never_merged():
    if not _has_bidmc():
        pytest.skip("real data missing")
    result = _bidmc_result()
    ecg = _by_source(
        result, "HR", "computed from ECG lead II by prototype detector, exploratory"
    )[0]
    monitor = _by_source(result, "HR", MONITOR_SOURCE)[0]
    assert ecg is not monitor
    assert ecg["n_windows_evaluated"] > 0
    assert monitor["n_windows_evaluated"] > 0


@pytest.mark.realdata
def test_realdata_bidmc01_spo2_reports_missing_values_without_nan():
    if not _has_bidmc():
        pytest.skip("real data missing")
    result = _bidmc_result()
    spo2 = _by_source(result, "SpO2", MONITOR_SOURCE)[0]
    assert spo2["n_windows_evaluated"] > 0
    reasons = [window["reason"] for window in spo2["windows"]]
    assert any("missing" in reason for reason in reasons)
    assert any(
        window["status"] == "evaluated" and "missing (NaN)" in window["reason"]
        for window in spo2["windows"]
    )
    assert all(window["value"] is None or window["value"] == window["value"] for window in spo2["windows"])
    json.dumps(result, allow_nan=False)


@pytest.mark.realdata
def test_realdata_bidmc01_waveform_resp_is_not_evaluated():
    if not _has_bidmc():
        pytest.skip("real data missing")
    result = _bidmc_result()
    resp_wave = [
        indicator
        for indicator in result["indicators"]
        if indicator["name"] == "RESP" and indicator["source"] != MONITOR_SOURCE
    ][0]
    assert resp_wave["status"] == "not_evaluated"
    assert resp_wave["windows"]
    assert all(window["value"] is None for window in resp_wave["windows"])
    assert "validated" in resp_wave["windows"][0]["reason"].lower()


@pytest.mark.realdata
def test_realdata_bidmc01_rhythm_context_is_not_available():
    if not _has_bidmc():
        pytest.skip("real data missing")
    result = _bidmc_result()
    assert result["rhythm_context"] == {"source": "not available", "segments": []}


@pytest.mark.realdata
def test_realdata_bidmc01_rules_run_on_monitor_channels():
    if not _has_bidmc():
        pytest.skip("real data missing")
    result = _bidmc_result()
    by_id = {rule["rule_id"]: rule for rule in result["rule_results"]}
    for rule_id in ("spo2_sensor_sanity", "spo2_low_demo", "resp_sensor_sanity"):
        assert by_id[rule_id]["status"] in {"flag", "no_flag"}
        assert by_id[rule_id]["n_windows_evaluated"] > 0
    assert by_id["hr_sensor_sanity"]["n_windows_evaluated"] > 0
