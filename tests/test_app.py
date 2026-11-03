"""Headless tests for the Streamlit dashboard (app/app.py)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")

TEXT_KINDS = (
    "title",
    "header",
    "subheader",
    "markdown",
    "caption",
    "text",
    "warning",
    "error",
    "info",
    "success",
    "code",
)


def _texts(at):
    out = []
    for kind in TEXT_KINDS:
        for element in at.get(kind):
            value = getattr(element, "value", None)
            if value is None:
                value = getattr(element, "body", "")
            if value:
                out.append(str(value))
    for element in at.dataframe:
        try:
            out.append(element.value.to_string())
        except Exception:
            pass
    for element in at.table:
        try:
            out.append(element.value.to_string())
        except Exception:
            out.append(str(element.value))
    return out


def _all_text(at):
    return "\n".join(_texts(at))


def _run(needle=None):
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    if needle is not None:
        boxes = at.sidebar.selectbox
        assert boxes, "the sidebar has no record selector"
        options = [str(o) for o in boxes[0].options if needle in str(o)]
        assert options, f"no sidebar option containing {needle!r}: {list(boxes[0].options)}"
        at.sidebar.selectbox[0].set_value(options[0]).run()
        assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_dashboard_runs_on_the_simulated_fixture_with_both_banners():
    at = _run()
    text = _all_text(at)
    assert "Research prototype. Not diagnostic. Not NASA-endorsed." in text
    assert "SIMULATED DATA" in text
    assert "simulated" in text.lower()
    assert "SIMULATED" in text


def test_fixture_shows_all_four_card_states():
    text = _all_text(_run())
    for state in ("Evaluated", "Rule flag", "Not evaluated", "Data issue"):
        assert state in text, f"dashboard never shows {state!r}"


def test_fixture_rule_flag_stays_on_its_own_source():
    at = _run()
    cards = [
        str(text)
        for text in _texts(at)
        if "indicator-card" in str(text) and "<h3>HR " in str(text)
    ]
    annotation_cards = [
        card for card in cards if "SIMULATED annotation file beats" in card
    ]
    monitor_cards = [
        card for card in cards if "SIMULATED monitor numerics" in card
    ]
    assert len(annotation_cards) == 1
    assert len(monitor_cards) == 1
    assert "status-evaluated" in annotation_cards[0]
    assert "status-rule-flag" not in annotation_cards[0]
    assert "status-rule-flag" in monitor_cards[0]


def test_fixture_labels_flag_types_differently():
    text = _all_text(_run())
    assert "possible sensor/data error" in text
    assert "ILLUSTRATIVE, NOT CLINICAL" in text


def test_fixture_shows_model_panel_and_no_combined_score():
    text = _all_text(_run())
    assert "No anomaly model run" in text
    lowered = text.lower()
    for forbidden in ("health score", "overall score", "combined score", "risk score"):
        assert forbidden not in lowered


def test_no_diagnosis_wording_outside_not_a_diagnosis():
    at = _run()
    for text in _texts(at):
        lowered = text.lower()
        index = lowered.find("diagnos")
        while index != -1:
            context = lowered[max(0, index - 30): index + 40]
            assert "not a diagnosis" in context or "not diagnostic" in context, text
            index = lowered.find("diagnos", index + 1)


def test_sidebar_offers_the_fixture_and_local_records():
    at = _run()
    options = [str(o) for o in at.sidebar.selectbox[0].options]
    assert any("SIMULATED" in option for option in options)
    assert any("mitdb/100" in option for option in options)
    assert any("mitdb/201" in option for option in options)
    assert any("mitdb/202" in option for option in options)
    assert any("bidmc/bidmc01" in option for option in options)
    if os.path.exists("data/raw/mitdb/100.hea"):
        assert any("mitdb/100" in option for option in options)
    if os.path.exists("data/raw/bidmc/bidmc01.hea"):
        assert any("bidmc/bidmc01" in option for option in options)


def test_tables_render_without_the_dataframe_element():
    if not os.path.exists("data/raw/mitdb/100.hea"):
        pytest.skip("real data missing")
    at = _run("mitdb/100")
    assert not at.dataframe
    assert at.table


@pytest.mark.realdata
def test_dashboard_runs_on_record_100():
    if not os.path.exists("data/raw/mitdb/100.hea"):
        pytest.skip("real data missing")
    text = _all_text(_run("mitdb/100"))
    assert "Research prototype. Not diagnostic. Not NASA-endorsed." in text
    assert "actual_research_data" in text
    assert "SIMULATED DATA" not in text
    assert "annotation file beats" in text


@pytest.mark.realdata
def test_dashboard_shows_201_afib_and_rmssd_not_evaluated():
    if not os.path.exists("data/raw/mitdb/201.hea"):
        pytest.skip("real data missing")
    at = _run("mitdb/201")
    text = _all_text(at)
    assert "(AFIB" in text
    cards = [t for t in _texts(at) if "RMSSD" in t]
    assert cards, "no RMSSD card"
    assert any("Not evaluated" in card for card in cards)


@pytest.mark.realdata
def test_dashboard_runs_on_bidmc_and_says_rhythm_is_not_available():
    if not os.path.exists("data/raw/bidmc/bidmc01.hea"):
        pytest.skip("real data missing")
    text = _all_text(_run("bidmc/bidmc01"))
    assert "monitor numerics" in text
    lowered = text.lower()
    assert "not available" in lowered
    assert "SIMULATED DATA" not in text
