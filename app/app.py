import sys
import json
from pathlib import Path
from html import escape

# Add project root to Python path BEFORE importing project modules.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline import analyze, make_simulated_fixture
from src.ingest.wfdb_loader import (
    beat_only,
    get_signal,
    load_beat_annotations,
    load_record,
)


BANNER = (
    "Research prototype. Not diagnostic. Not NASA-endorsed. "
    "Not validated for clinical use or operational astronaut care."
)

STATUS_TEXT = {
    "evaluated": "Evaluated",
    "not_evaluated": "Not evaluated",
    "data_issue": "Data issue",
}

st.set_page_config(
    page_title="Astronaut Health Monitoring (research prototype)", layout="wide")


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --mission-bg: #071018;
            --mission-panel: #0c1822;
            --mission-panel-alt: #101f2b;
            --mission-line: #203745;
            --mission-cyan: #62d9e8;
            --mission-text: #f1f6f7;
            --mission-muted: #9badb4;
            --mission-green: #63d39a;
            --mission-amber: #f2bb68;
            --mission-red: #ed827e;
        }
        [data-testid="stAppViewContainer"] { background: var(--mission-bg); }
        [data-testid="stSidebar"] { background: #09151e; border-right: 1px solid var(--mission-line); }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        .mission-eyebrow { color: var(--mission-cyan); font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .35rem; }
            .mission-eyebrow { color: #62d9e8 !important; display: block; font-size: .72rem; font-weight: 700; letter-spacing: .16em; line-height: 1.4; margin: 0 0 .45rem; opacity: 1 !important; text-transform: uppercase; visibility: visible !important; }
        .mission-subtitle { color: var(--mission-muted); font-size: 1rem; margin-top: -.5rem; }
        .mission-banner, .mission-panel, .indicator-card, .evidence-panel { background: var(--mission-panel); border: 1px solid var(--mission-line); border-radius: 4px; padding: 1rem 1.15rem; }
        .mission-banner { border-left: 4px solid var(--mission-cyan); margin: 1.2rem 0 1.5rem; }
        .mission-banner.simulated { border-left-color: var(--mission-amber); }
        .banner-label { color: var(--mission-cyan); font-size: .72rem; font-weight: 700; letter-spacing: .14em; }
        .simulated .banner-label { color: var(--mission-amber); }
        .banner-copy { color: var(--mission-text); margin-top: .35rem; }
        .section-kicker { color: var(--mission-cyan); font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin: .4rem 0 .45rem; }
        .summary-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .65rem; margin: .8rem 0 1.5rem; }
        .summary-cell { background: var(--mission-panel-alt); border: 1px solid var(--mission-line); padding: .8rem; min-height: 76px; }
        .summary-label { color: var(--mission-muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
        .summary-value { color: var(--mission-text); font-size: 1.15rem; font-weight: 700; margin-top: .35rem; }
        .indicator-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: .8rem; }
        .indicator-card { min-height: 245px; }
        .indicator-card.status-evaluated { border-top: 3px solid var(--mission-green); }
        .indicator-card.status-rule-flag { border-top: 3px solid var(--mission-amber); }
        .indicator-card.status-not-evaluated { border-top: 3px solid #71838b; }
        .indicator-card.status-data-issue { border-top: 3px solid var(--mission-red); }
        .indicator-card h3 { color: var(--mission-text); font-size: 1.05rem; margin: 0 0 .65rem; }
        .status-badge { border: 1px solid currentColor; border-radius: 2px; display: inline-block; font-size: .67rem; font-weight: 700; letter-spacing: .08em; padding: .2rem .4rem; text-transform: uppercase; }
        .status-evaluated .status-badge { color: var(--mission-green); }
        .status-rule-flag .status-badge { color: var(--mission-amber); }
        .status-not-evaluated .status-badge { color: #a9b5b9; }
        .status-data-issue .status-badge { color: var(--mission-red); }
        .metric-value { color: var(--mission-cyan); font-size: 1.35rem; font-weight: 700; margin: .5rem 0 .7rem; }
        .card-meta { color: var(--mission-muted); font-size: .83rem; line-height: 1.55; }
        .card-meta strong, .evidence-label { color: var(--mission-text); }
        .card-limit, .card-reason { border-left: 2px solid var(--mission-line); color: var(--mission-muted); font-size: .78rem; margin-top: .7rem; padding-left: .65rem; }
        .card-reason { border-left-color: var(--mission-amber); color: #d6c3a7; }
        .rule-label { color: var(--mission-amber); font-size: .78rem; margin-top: .65rem; }
        .evidence-panel { margin: .8rem 0; }
        .evidence-label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
        @media (max-width: 800px) { .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .block-container { padding: 1rem .8rem 2rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_indicator_card(indicator, status_text, flagged):
    status_class = status_text.lower().replace(" ", "-")
    name = escape(str(indicator["name"]))
    value = escape(value_text(indicator))
    source = escape(str(indicator["source"]))
    method = escape(str(indicator["method"]))
    unit = escape(str(indicator["unit"]))
    limit = escape(indicator_limit(indicator))
    reason = escape(first_reason(indicator))
    labels = []
    if any(rule["rule_status"] == "plausibility_check" for rule in flagged):
        labels.append(
            "possible sensor/data error - data quality, not a health finding")
    if any(rule["rule_status"] == "illustrative_not_clinical" for rule in flagged):
        labels.append(
            "ILLUSTRATIVE, NOT CLINICAL demo condition - see the rule table")
    label_html = "".join(
        f'<div class="rule-label"><strong>Rule label:</strong> {escape(label)}</div>' for label in labels)
    reason_html = f'<div class="card-reason"><strong>Reason:</strong> {reason}</div>' if reason else ""
    st.markdown(
        f"""
        <article class="indicator-card status-{status_class}">
            <h3>{name} <span class="status-badge">{escape(status_text)}</span></h3>
            <div class="metric-value">{value}</div>
            <div class="card-meta"><strong>Unit:</strong> {unit}<br>
            <strong>Window count:</strong> {indicator['n_windows_evaluated']} of {indicator['n_windows_total']} evaluated<br>
            <strong>Source:</strong> {source}<br>
            <strong>Method:</strong> {method}</div>
            <div class="card-limit"><strong>Limitation:</strong> {limit}</div>
            {label_html}{reason_html}
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, kicker=None):
    if kicker:
        st.markdown(
            f'<div class="section-kicker">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.subheader(title)


@st.cache_data(show_spinner=False)
def cached_analysis(path_str, source_label):
    return analyze(path_str, data_type="actual_research_data", source_label=source_label)


def record_options():
    options = ["SIMULATED fixture (sim-001)"]
    for kind in ("mitdb", "bidmc"):
        base = ROOT / "data" / "raw" / kind
        if not base.exists():
            continue
        for header in sorted(base.glob("*.hea")):
            if kind == "bidmc" and header.stem.endswith("n"):
                continue
            options.append(f"{kind}/{header.stem}")
    return options


def card_status(indicator, rule_results):
    """A card says Rule flag when any rule for that indicator flagged overall."""
    if any(
        rule["indicator"] == indicator["name"] and rule["status"] == "flag"
        and indicator["source"] in rule.get("flagged_indicator_sources", [])
        for rule in rule_results
    ):
        return "Rule flag"
    return STATUS_TEXT.get(indicator["status"], "Data issue")


def indicator_limit(indicator):
    source = indicator["source"].lower()
    if "detector" in source:
        return "Exploratory prototype method, not validated (see docs/rpeak_eval.md)."
    if "monitor numerics" in source:
        return "Monitor-derived number; never merged with ECG-derived values."
    if "waveform" in source:
        return "Not evaluated because this prototype has no validated method for it."
    return "Single dataset source; see the limitations section."


def value_text(indicator):
    evaluated = [
        window
        for window in indicator["windows"]
        if window["status"] == "evaluated" and window["value"] is not None
    ]
    if not evaluated:
        return "no value reported"
    window = evaluated[0]
    return (
        f"{window['value']:.2f} {indicator['unit']} "
        f"in window {window['start_s']:g}-{window['end_s']:g} s"
    )


def first_reason(indicator):
    for window in indicator["windows"]:
        if window["status"] != "evaluated" and window["reason"]:
            return window["reason"]
    return ""


@st.cache_data(show_spinner=False)
def signal_frame(choice, start_s, length_s):
    """Raw signal for display only. No indicator is computed here."""
    if choice.startswith("SIMULATED"):
        fs = 125.0
        n = int(length_s * fs)
        time_s = np.arange(n) / fs + start_s
        local = np.arange(n) / fs
        signal = np.sin(2 * np.pi * 1.2 * local) + 0.15 * \
            np.sin(2 * np.pi * 12 * local)
        marker = np.full(n, np.nan)
        for beat_s in np.arange(0.2, length_s, 0.8):
            index = int(beat_s * fs)
            if 0 <= index < n:
                marker[index] = signal[index] + 0.6
        frame = pd.DataFrame(
            {"time_s": time_s, "signal": signal, "beat marker": marker})
        return frame, "SIMULATED signal and invented markers; no person was monitored."

    kind, record = choice.split("/", 1)
    path = ROOT / "data" / "raw" / kind / record
    loaded = load_record(str(path))
    channel = "MLII" if "MLII" in loaded.signal_names else (
        "II" if "II" in loaded.signal_names else loaded.signal_names[0]
    )
    values = get_signal(loaded, channel)
    fs = float(loaded.fs)
    first = int(round(start_s * fs))
    last = min(values.size, first + int(round(length_s * fs)))
    local_signal = values[first:last]
    time_s = (np.arange(first, last) / fs)

    if kind == "mitdb":
        samples, symbols, _aux = load_beat_annotations(str(path))
        beat_samples, _beat_symbols = beat_only(samples, symbols)
        marker = np.full(local_signal.size, np.nan)
        for sample in beat_samples:
            index = int(sample) - first
            if 0 <= index < local_signal.size:
                marker[index] = local_signal[index] + 0.5
        frame = pd.DataFrame(
            {"time_s": time_s, "signal": local_signal, "beat marker": marker}
        )
        return frame, f"Channel {channel}. Beat markers come from the dataset annotation file."

    frame = pd.DataFrame({"time_s": time_s, "signal": local_signal})
    return frame, (
        f"Channel {channel}. BIDMC has no beat annotations, so no beat markers are drawn."
    )


inject_styles()
st.markdown(
    '<div class="mission-eyebrow" style="color:#62d9e8;display:block;opacity:1;visibility:visible;">'
    'Telemetry console / flight health research</div>',
    unsafe_allow_html=True,
)
st.title("Astronaut Health Monitoring System")
st.markdown('<div class="mission-subtitle">Mission health telemetry review | Research prototype interface</div>', unsafe_allow_html=True)
st.warning(BANNER)

options = record_options()
choice = st.sidebar.selectbox("Record", options)

if choice.startswith("SIMULATED"):
    result = make_simulated_fixture()
    source_label = "hand-built SIMULATED fixture"
else:
    kind, record = choice.split("/", 1)
    source_label = (
        "MIT-BIH Arrhythmia Database, local copy"
        if kind == "mitdb"
        else "BIDMC PPG and Respiration Dataset, local copy"
    )
    with st.spinner("Analysing record..."):
        result = cached_analysis(
            str(ROOT / "data" / "raw" / kind / record), source_label)

dataset = result["dataset"]
st.sidebar.caption(f"{dataset['name']} - record {dataset['record']}")
st.sidebar.caption(dataset["version_label"])

if dataset["data_type"] == "simulated":
    st.markdown(
        '<div class="mission-banner simulated"><div class="banner-label">SIMULATED DATA</div>'
        '<div class="banner-copy">Hand-built fixture for interface and pipeline checks. No person was monitored.</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="mission-banner"><div class="banner-label">ACTUAL RESEARCH DATA</div>'
        f'<div class="banner-copy">Data type: {escape(str(dataset["data_type"]))} - terrestrial/hospital data, not astronaut data.</div></div>',
        unsafe_allow_html=True,
    )

data_quality = result["data_quality"]
rule_results = result["rule_results"]
indicators = result["indicators"]
evaluated_count = sum(indicator["status"] ==
                      "evaluated" for indicator in indicators)
flag_count = sum(rule["status"] == "flag" for rule in rule_results)
not_evaluated_count = sum(
    indicator["status"] == "not_evaluated" for indicator in indicators)
issue_count = len(data_quality["issues"])

render_section_title("Telemetry summary", "Live record overview")
st.markdown(
    f"""
    <div class="summary-grid">
        <div class="summary-cell"><div class="summary-label">Record ID</div><div class="summary-value">{escape(str(dataset['record']))}</div></div>
        <div class="summary-cell"><div class="summary-label">Evaluated indicators</div><div class="summary-value">{evaluated_count}</div></div>
        <div class="summary-cell"><div class="summary-label">Rule flags</div><div class="summary-value">{flag_count}</div></div>
        <div class="summary-cell"><div class="summary-label">Not evaluated</div><div class="summary-value">{not_evaluated_count}</div></div>
        <div class="summary-cell"><div class="summary-label">Data issues</div><div class="summary-value">{issue_count}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_section_title("Data quality", "Validation evidence")
st.markdown(
    f'<div class="evidence-panel"><div class="evidence-label">Status</div>'
    f'<div class="summary-value">{escape(str(data_quality["status"]))}</div></div>',
    unsafe_allow_html=True,
)
if data_quality["issues"]:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "code": issue["code"],
                    "severity": issue["severity"],
                    "message": issue["message"],
                    "field": issue["field"] or "",
                }
                for issue in data_quality["issues"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("No data-quality issues were reported by the validator.")
if data_quality["not_checked"]:
    st.caption("Not checked: " + ", ".join(data_quality["not_checked"]))

rhythm = result["rhythm_context"]
render_section_title("Rhythm context", "Annotation provenance")
st.caption("Labels are shown exactly as written in the dataset annotation file.")
st.markdown(
    f'<div class="mission-panel"><div class="card-meta"><strong>Rhythm label source:</strong> {escape(str(rhythm["source"]))}<br>'
    f'<strong>Availability:</strong> {"Available" if rhythm["segments"] else "Not available"}</div></div>',
    unsafe_allow_html=True,
)
if rhythm["segments"]:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "start (s)": segment["start_s"],
                    "end (s)": segment["end_s"],
                    "label": segment["label"],
                }
                for segment in rhythm["segments"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("No rhythm annotations are available for this record.")

render_section_title("Indicators", "Per-source telemetry status")
st.markdown('<div class="indicator-grid">', unsafe_allow_html=True)
for indicator in indicators:
    status_text = card_status(indicator, rule_results)
    flagged = [
        rule for rule in rule_results
        if rule["indicator"] == indicator["name"]
        and rule["status"] == "flag"
        and indicator["source"] in rule.get("flagged_indicator_sources", [])
    ]
    render_indicator_card(indicator, status_text, flagged)
st.markdown('</div>', unsafe_allow_html=True)

render_section_title("Evaluation evidence", "Rules and eligibility")
st.markdown(
    '<div class="evidence-panel">Rule labels, reasons, eligibility gates, and data-quality evidence are shown below without interpretation.</div>',
    unsafe_allow_html=True,
)
st.subheader("Rule results")
st.dataframe(
    pd.DataFrame(
        [
            {
                "rule id": rule["rule_id"],
                "source": rule["source"],
                "status": rule["status"],
                "reason": rule["reason"],
            }
            for rule in rule_results
        ]
    ),
    width="stretch",
    hide_index=True,
)

render_section_title("Model", "Optional analysis")
st.info("No anomaly model run")

render_section_title("Signal / waveform", "Raw signal context")
duration_s = float(dataset["duration_s"] or 60.0)
length_s = 10.0
max_start = max(0.0, duration_s - length_s)
if max_start > 0:
    start_s = st.slider("Window start (s)", 0.0, max_start, 0.0, step=1.0)
else:
    start_s = 0.0
frame, chart_caption = signal_frame(choice, float(start_s), length_s)
if "beat marker" in frame.columns:
    st.line_chart(frame, x="time_s", y=["signal", "beat marker"], height=260)
else:
    st.line_chart(frame, x="time_s", y=["signal"], height=260)
st.caption(chart_caption)

render_section_title("Limitations", "Scientific boundaries")
with st.expander("Limitations"):
    for line in result["limitations"]:
        st.markdown(f"- {line}")

st.download_button(
    "Download JSON",
    data=json.dumps(result, indent=2, allow_nan=False),
    file_name=f"{dataset['record']}_analysis.json",
    mime="application/json",
)
