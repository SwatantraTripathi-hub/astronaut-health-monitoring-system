import math
from pathlib import Path

import pytest
import yaml

from src.rules.engine import (
    ALLOWED_COMPARISONS,
    ALLOWED_STATUSES,
    DISCLAIMER,
    REQUIRED_FIELDS,
    RuleError,
    RuleResult,
    evaluate,
    load_rules,
    validate_rule,
)

# Values that section 0 rule 3 of docs/FREEBUFF_PROMPT.md forbids as rule values.
FORBIDDEN_VALUES = {40, 120, 20, 95, 90, 8, 25, 500, 6}


def _rule(**overrides):
    rule = {
        "id": "demo_hr_high",
        "indicator": "HR",
        "comparison": "gt",
        "value": 133,
        "unit": "bpm",
        "source": "none; demo only",
        "status": "illustrative_not_clinical",
        "message": "ILLUSTRATIVE, NOT CLINICAL: demo rule, arbitrary value",
    }
    rule.update(overrides)
    return rule


def test_allowed_statuses_and_comparisons_are_exact():
    assert ALLOWED_STATUSES == {"plausibility_check", "illustrative_not_clinical"}
    assert ALLOWED_COMPARISONS == {"lt", "gt", "lte", "gte", "outside_range"}


def test_shipped_rules_load_and_only_expected_rules_exist():
    rules = load_rules()
    ids = [rule["id"] for rule in rules]
    assert len(ids) == 5
    assert len(set(ids)) == len(ids)
    assert set(ids) == {
        "hr_sensor_sanity",
        "spo2_sensor_sanity",
        "resp_sensor_sanity",
        "hr_high_demo",
        "spo2_low_demo",
    }
    for rule in rules:
        for field in REQUIRED_FIELDS:
            assert field in rule, f"{rule.get('id')} missing {field}"
        assert rule["status"] in ALLOWED_STATUSES


def test_shipped_rule_sources_and_messages_by_status():
    rules = {rule["id"]: rule for rule in load_rules()}
    plausibility = [r for r in rules.values() if r["status"] == "plausibility_check"]
    illustrative = [r for r in rules.values() if r["status"] == "illustrative_not_clinical"]
    assert len(plausibility) == 3
    assert len(illustrative) == 2
    for rule in plausibility:
        assert rule["source"] == "none; engineering choice"
        assert "possible sensor/data error, not a health finding" in rule["message"]
    for rule in illustrative:
        assert rule["source"] == "none; demo only"
        assert rule["message"].startswith("ILLUSTRATIVE, NOT CLINICAL:")
    assert {rules[i]["indicator"] for i in ("hr_sensor_sanity", "hr_high_demo")} == {"HR"}
    assert {rules[i]["indicator"] for i in ("spo2_sensor_sanity", "spo2_low_demo")} == {"SpO2"}
    assert rules["resp_sensor_sanity"]["indicator"] == "RESP"


def test_no_forbidden_number_used_as_a_rule_value():
    for rule in load_rules():
        raw = rule["value"]
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            assert value not in FORBIDDEN_VALUES, f"{rule['id']} uses forbidden value {value}"


def test_every_shipped_rule_is_labelled_illustrative_or_data_quality_only():
    for rule in load_rules():
        assert rule["status"] in {"plausibility_check", "illustrative_not_clinical"}


def test_clinical_status_rejected_at_load(tmp_path):
    text = yaml.safe_dump([_rule(id="bad", status="clinical")], sort_keys=False)
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RuleError) as excinfo:
        load_rules(path)
    assert "clinical" in str(excinfo.value)


def test_malformed_rule_missing_field_rejected(tmp_path):
    broken = _rule()
    del broken["unit"]
    with pytest.raises(RuleError) as excinfo:
        validate_rule(broken)
    assert "unit" in str(excinfo.value)

    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump([broken], sort_keys=False), encoding="utf-8")
    with pytest.raises(RuleError):
        load_rules(path)


def test_unknown_comparison_rejected():
    with pytest.raises(RuleError):
        validate_rule(_rule(comparison="approx"))


def test_illustrative_hr_high_flags_above_demo_value():
    rule = _rule()
    result = evaluate(rule, 150, "bpm")
    assert isinstance(result, RuleResult)
    assert result.status == "flag"
    assert result.rule_id == "demo_hr_high"
    assert result.source == "none; demo only"
    assert DISCLAIMER in result.reason
    assert "ILLUSTRATIVE, NOT CLINICAL:" in result.reason


def test_illustrative_hr_high_no_flag_below_demo_value():
    result = evaluate(_rule(), 60, "bpm")
    assert result.status == "no_flag"
    assert DISCLAIMER in result.reason


def test_plausibility_hr_outside_engineering_bounds_flags():
    rule = next(r for r in load_rules() if r["id"] == "hr_sensor_sanity")
    high = evaluate(rule, 999, "bpm")
    low = evaluate(rule, 2, "bpm")
    ok = evaluate(rule, 75, "bpm")
    assert high.status == "flag"
    assert low.status == "flag"
    assert ok.status == "no_flag"
    assert "possible sensor/data error, not a health finding" in high.reason


def test_plausibility_spo2_outside_zero_to_hundred_flags():
    rule = next(r for r in load_rules() if r["id"] == "spo2_sensor_sanity")
    assert evaluate(rule, 105, "%").status == "flag"
    assert evaluate(rule, -1, "%").status == "flag"
    assert evaluate(rule, 97, "%").status == "no_flag"


def test_plausibility_resp_below_zero_flags():
    rule = next(r for r in load_rules() if r["id"] == "resp_sensor_sanity")
    assert evaluate(rule, -1, "pm").status == "flag"
    assert evaluate(rule, 12, "pm").status == "no_flag"


def test_unit_mismatch_is_not_evaluated():
    result = evaluate(_rule(), 150, "%")
    assert result.status == "not_evaluated"
    assert result.status != "no_flag"
    assert result.reason
    assert DISCLAIMER in result.reason


def test_missing_value_is_not_evaluated():
    for value in (None, math.nan):
        result = evaluate(_rule(), value, "bpm")
        assert result.status == "not_evaluated"
        assert result.reason
        assert DISCLAIMER in result.reason


def test_every_status_carries_the_same_disclaimer():
    rule = _rule()
    for value, unit in ((150, "bpm"), (60, "bpm"), (None, "bpm"), (150, "%")):
        assert DISCLAIMER in evaluate(rule, value, unit).reason
