"""Self-audit of the project's own hard rules (CONTEXT.md and prompt section 0)."""

import json
import re
from pathlib import Path

import pytest

from src.config import load_params
from src.pipeline import LIMITATIONS, analyze, make_simulated_fixture
from src.rules.engine import ALLOWED_STATUSES, load_rules

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src", "app", "config")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json",
                 ".txt", ".md", ".html", ".css", ".js"}

BANNED_PHRASES = [
    "diagnosis",
    "NASA-approved",
    "100% accurate",
    "guarantees safety",
    "real-time astronaut monitoring",
]

FORBIDDEN_RULE_VALUES = {40, 120, 20, 95, 90, 8, 25, 500, 6}

TOP_KEYS = {
    "dataset",
    "data_quality",
    "rhythm_context",
    "indicators",
    "rule_results",
    "model_results",
    "limitations",
}


def _text_files():
    for dirname in SCAN_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_rule_is_marked_clinical():
    rules = load_rules()
    assert rules
    for rule in rules:
        assert rule["status"] in ALLOWED_STATUSES
        assert rule["status"] != "clinical"
    text = (ROOT / "configs" / "rules.yaml").read_text(encoding="utf-8").lower()
    assert "status: clinical" not in text


def test_banned_phrases_do_not_appear_outside_a_negation():
    for path in _text_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if "not " in line or "NOT " in line:
                continue
            lowered = line.lower()
            for phrase in BANNED_PHRASES:
                assert phrase.lower() not in lowered, (
                    f"{path.relative_to(ROOT)}:{lineno} contains {phrase!r}"
                )


def test_every_pipeline_output_is_json_safe():
    outputs = [make_simulated_fixture()]
    for record in ("100", "201"):
        header = ROOT / "data" / "raw" / "mitdb" / f"{record}.hea"
        if header.exists():
            outputs.append(
                analyze(
                    str(header.with_suffix("")),
                    data_type="actual_research_data",
                    source_label="hard-rule audit",
                )
            )
    bidmc = ROOT / "data" / "raw" / "bidmc" / "bidmc01.hea"
    if bidmc.exists():
        outputs.append(
            analyze(
                str(bidmc.with_suffix("")),
                data_type="actual_research_data",
                source_label="hard-rule audit",
            )
        )
    outputs.append(
        analyze(
            str(ROOT / "data" / "raw" / "mitdb" / "no_such_record"),
            data_type="actual_research_data",
            source_label="hard-rule audit",
        )
    )

    for output in outputs:
        assert set(output) == TOP_KEYS
        serialized = json.dumps(output, allow_nan=False)
        # A numeric NaN/Infinity would be written as a bare JSON token; the word
        # "NaN" may still appear inside text such as "1 missing (NaN)".
        assert re.search(r":\s*(NaN|Infinity|-Infinity)", serialized) is None
        json.loads(serialized)


def test_simulated_fixture_is_labelled_simulated():
    fixture = make_simulated_fixture()
    assert fixture["dataset"]["data_type"] == "simulated"
    assert "SIMULATED" in json.dumps(fixture)
    assert any("SIMULATED" in line for line in fixture["limitations"])
    assert any("SIMULATED" in issue["message"]
               for issue in fixture["data_quality"]["issues"])


def test_params_values_all_carry_the_design_parameter_comment():
    text = (ROOT / "configs" / "params.yaml").read_text(encoding="utf-8")
    value_lines = [
        line
        for line in text.splitlines()
        if re.match(r"^\s+[a-z_]+:\s*\S", line) and not line.strip().startswith("#")
    ]
    assert value_lines, "params.yaml has no values"
    for line in value_lines:
        assert "design parameter, no clinical source" in line, line


def test_no_rule_uses_a_forbidden_value():
    for rule in load_rules():
        values = rule["value"] if isinstance(
            rule["value"], list) else [rule["value"]]
        for value in values:
            assert value not in FORBIDDEN_RULE_VALUES, f"{rule['id']} uses {value}"


def test_rule_messages_carry_their_required_label():
    for rule in load_rules():
        message = rule.get("message", "")
        if rule["status"] == "illustrative_not_clinical":
            assert message.startswith("ILLUSTRATIVE, NOT CLINICAL:")
        else:
            assert "possible sensor/data error, not a health finding" in message


def test_limitations_keep_the_required_negations():
    text = " ".join(LIMITATIONS).lower()
    for phrase in (
        "research prototype",
        "not a diagnosis",
        "not astronaut data",
        "no flag does not mean healthy",
        "design parameter",
        "dataset annotations",
    ):
        assert phrase in text


def test_no_combined_health_score_field_anywhere():
    fixture = make_simulated_fixture()
    text = json.dumps(fixture).lower()
    for forbidden in ("health_score", "overall_score", "combined_score", "risk_score"):
        assert forbidden not in text
    assert fixture["model_results"] == {"status": "not_run", "warnings": []}
