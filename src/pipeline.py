"""One record in, one analysis dict out.

The output keeps three things apart on purpose: data quality, rule results and
model results. Nothing here produces a combined health score.
"""

from pathlib import Path

import numpy as np
import wfdb

from src.config import load_params
from src.features.crosscheck import numeric_window_mean
from src.features.hr import hr_windows
from src.features.hrv import rmssd_gated
from src.features.rpeaks import detect_rpeaks
from src.ingest.rhythm import rhythm_segments
from src.ingest.wfdb_loader import (
    beat_only,
    get_signal,
    load_beat_annotations,
    load_record,
    normalize_name,
)
from src.rules.engine import DISCLAIMER, evaluate, load_rules
from src.validate.validator import validate_record

LIMITATIONS = [
    "This is a research prototype, not a diagnosis, not a medical device and not endorsed by NASA.",
    "Data is terrestrial/hospital data (MIT-BIH, BIDMC), not astronaut data.",
    "No flag does not mean healthy: every rule is a configured condition, not a finding.",
    "Window lengths, minimum counts and rhythm eligibility labels are design parameters, no clinical source.",
    "Rhythm labels are dataset annotations, not output of this software.",
    "Indicators are computed by an early prototype and have not been validated for clinical or operational use.",
]

DATASET_NAMES = {
    "mitdb": "MIT-BIH Arrhythmia Database",
    "bidmc": "BIDMC PPG and Respiration Dataset",
    "simulated": "Simulated fixture",
}

VERSION_LABEL = "PhysioNet local copy; version not recorded"

BIDMC_MONITOR_SOURCE = "monitor numerics (1 Hz), numeric, monitor-derived"
BIDMC_ECG_SOURCE = "computed from ECG lead II by prototype detector, exploratory"
BIDMC_WAVE_RESP_SOURCE = "computed from RESP waveform, exploratory"
MONITOR_CHANNELS = [
    ("HR", "bpm"),
    ("PULSE", "bpm"),
    ("SpO2", "%"),
    ("RESP", "pm"),
]

MITDB_INDICATORS = [
    ("HR", "bpm", "annotation file beats",
     "60 / mean RR of annotation beats inside each window"),
    (
        "HR",
        "bpm",
        "computed from ECG by prototype detector, exploratory",
        "60 / mean RR of detected R-peaks inside each window; see docs/rpeak_eval.md",
    ),
    (
        "RMSSD",
        "ms",
        "annotation file beats",
        "RMSSD over NN intervals inside eligible rhythm labels; beats = beat_only(annotations)",
    ),
]


# ----------------------------------------------------------------- small helpers


def _issue_dict(issue):
    return {
        "code": issue.code,
        "severity": issue.severity,
        "message": issue.message,
        "field": issue.field,
    }


def _window(start_s, end_s, value, status, reason):
    return {
        "start_s": start_s,
        "end_s": end_s,
        "value": value,
        "status": status,
        "reason": reason,
    }


def _indicator(name, unit, source, method, windows, status=None):
    if status is None:
        status = "evaluated" if any(
            w["status"] == "evaluated" for w in windows) else "not_evaluated"
    return {
        "name": name,
        "unit": unit,
        "source": source,
        "method": method,
        "status": status,
        "n_windows_total": len(windows),
        "n_windows_evaluated": sum(1 for w in windows if w["status"] == "evaluated"),
        "windows": windows,
    }


def _header_info(path):
    try:
        header = wfdb.rdheader(path)
    except Exception:
        return None
    names = [normalize_name(n) for n in header.sig_name]
    units = {}
    for i, name in enumerate(names):
        units[name] = header.units[i] if i < len(header.units) else ""
    return {
        "fs": float(header.fs) if header.fs else None,
        "sig_len": int(header.sig_len),
        "record_name": header.record_name,
        "sig_names": names,
        "units": units,
    }


def _record_type(path, info):
    p = str(path)
    if Path(p + ".atr").exists():
        return "mitdb"
    if Path(p + "n.hea").exists():
        return "bidmc"
    if info is not None:
        names = set(info["sig_names"])
        if "MLII" in names:
            return "mitdb"
        if names & {"PLETH", "II", "AVR"}:
            return "bidmc"
    return "unknown"


def _dataset(path, info, record_type, data_type, source_label):
    if info is not None:
        record = info["record_name"]
        fs = info["fs"]
        duration = (info["sig_len"] / info["fs"]) if info["fs"] else None
    else:
        record = Path(str(path)).name
        fs = None
        duration = None
    version = VERSION_LABEL if record_type in {
        "mitdb", "bidmc"} else "not a PhysioNet record"
    if source_label:
        version = f"{version}; {source_label}"
    return {
        "name": DATASET_NAMES.get(record_type, record_type),
        "record": record,
        "version_label": version,
        "data_type": data_type,
        "fs": fs,
        "duration_s": duration,
    }


# ----------------------------------------------------------------- MIT-BIH


def _rmssd_windows(beat_samples, symbols, fs, duration_s, segments, params):
    beats = np.asarray(beat_samples, dtype=float)
    times = beats / float(fs)
    window_s = float(params["hrv"]["window_s"])
    windows = []
    start = 0.0
    while start < duration_s - 1e-9:
        end = min(start + window_s, duration_s)
        mask = (times >= start) & (times < end)
        window_symbols = [symbols[i] for i in np.flatnonzero(mask)]
        result = rmssd_gated(
            beats[mask],
            window_symbols,
            fs,
            segments,
            min_differences=params["hrv"]["min_differences"],
            min_coverage=params["hrv"]["min_coverage"],
        )
        windows.append(
            _window(
                start,
                end,
                result.value_ms if result.status == "evaluated" else None,
                result.status,
                result.reason,
            )
        )
        start += window_s
    return windows


def _hr_indicator_windows(beat_samples, fs, duration_s, params):
    windows = hr_windows(
        beat_samples,
        fs,
        duration_s,
        params["hr"]["window_s"],
        params["hr"]["step_s"],
    )
    return [_window(w.start_s, w.end_s, w.hr_bpm, w.status, w.reason) for w in windows]


def _mitdb_indicators(path, params):
    record = load_record(path)
    fs = record.fs
    duration = record.duration_s

    samples, symbols, aux_notes = load_beat_annotations(
        path, sig_len=record.sig_len)
    beat_samples, beat_symbols = beat_only(samples, symbols)
    segments = rhythm_segments(samples, symbols, aux_notes, record.sig_len)

    indicators = [
        _indicator(
            "HR",
            "bpm",
            "annotation file beats",
            "60 / mean RR of annotation beats inside each window",
            _hr_indicator_windows(beat_samples, fs, duration, params),
        )
    ]

    channel = "MLII" if "MLII" in record.signal_names else record.signal_names[0]
    detected = detect_rpeaks(get_signal(record, channel), fs, params["rpeaks"])
    indicators.append(
        _indicator(
            "HR",
            "bpm",
            "computed from ECG by prototype detector, exploratory",
            f"60 / mean RR of R-peaks detected on channel {channel}; see docs/rpeak_eval.md",
            _hr_indicator_windows(detected, fs, duration, params),
        )
    )

    indicators.append(
        _indicator(
            "RMSSD",
            "ms",
            "annotation file beats",
            "RMSSD over NN intervals inside eligible rhythm labels; beats = beat_only(annotations)",
            _rmssd_windows(beat_samples, beat_symbols, fs,
                           duration, segments, params),
        )
    )
    return indicators


# ----------------------------------------------------------------- BIDMC


def _numeric_windows(values, fs, duration_s, params):
    window_s = float(params["hr"]["window_s"])
    step_s = float(params["hr"]["step_s"])
    min_valid = float(params["numerics"]["min_valid_fraction"])
    windows = []
    start = 0.0
    while start < duration_s - 1e-9:
        end = min(start + window_s, duration_s)
        mean, n_valid, n_total = numeric_window_mean(
            values, fs, start, end, min_valid
        )
        missing = n_total - n_valid
        if mean is None:
            windows.append(
                _window(
                    start,
                    end,
                    None,
                    "not_evaluated",
                    f"{n_valid} of {n_total} samples valid (missing {missing}), below "
                    f"min_valid_fraction {min_valid} (design parameter, no clinical source)",
                )
            )
        else:
            windows.append(
                _window(
                    start,
                    end,
                    mean,
                    "evaluated",
                    f"mean of {n_valid} of {n_total} samples; {missing} missing (NaN)",
                )
            )
        start += step_s
    return windows


def _bidmc_indicators(path, params):
    base = str(path)[:-1] if str(path).endswith("n") else str(path)
    waveform = load_record(base)
    numerics = load_record(base + "n")

    channel = "II" if "II" in waveform.signal_names else waveform.signal_names[0]
    detected = detect_rpeaks(get_signal(
        waveform, channel), waveform.fs, params["rpeaks"])
    indicators = [
        _indicator(
            "HR",
            "bpm",
            BIDMC_ECG_SOURCE,
            f"60 / mean RR of R-peaks detected on ECG lead {channel}; see docs/crosscheck.md",
            _hr_indicator_windows(
                detected, waveform.fs, waveform.duration_s, params
            ),
        )
    ]

    for name, fallback_unit in MONITOR_CHANNELS:
        values = get_signal(numerics, name)
        unit = numerics.units.get(name) or fallback_unit
        indicators.append(
            _indicator(
                name,
                unit,
                BIDMC_MONITOR_SOURCE,
                "mean of non-missing 1 Hz samples inside each window; NaN samples are "
                "missing, never counted as zero",
                _numeric_windows(values, numerics.fs,
                                 numerics.duration_s, params),
            )
        )

    resp_unit = waveform.units.get("RESP") or "pm"
    indicators.append(
        _indicator(
            "RESP",
            resp_unit,
            BIDMC_WAVE_RESP_SOURCE,
            "no validated breath detector exists in this prototype",
            [
                _window(
                    0.0,
                    waveform.duration_s,
                    None,
                    "not_evaluated",
                    "method not validated: this prototype has no validated breath detector "
                    "for the RESP waveform",
                )
            ],
        )
    )
    return indicators


# ----------------------------------------------------------------- shared pieces


def _rhythm_context(path, record_type, info, params):
    if record_type != "mitdb" or info is None:
        return {"source": "not available", "segments": []}
    try:
        samples, symbols, aux_notes = load_beat_annotations(path)
        segments = rhythm_segments(
            samples, symbols, aux_notes, info["sig_len"])
    except Exception:
        return {"source": "not available", "segments": []}
    fs = float(info["fs"])
    return {
        "source": "dataset annotation file; labels exactly as written",
        "segments": [
            {
                "start_s": segment.start_sample / fs,
                "end_s": segment.end_sample / fs,
                "label": segment.label,
            }
            for segment in segments
        ],
    }


def _data_issue_indicators(record_type):
    if record_type == "mitdb":
        return [
            _indicator(name, unit, source, method, [], status="data_issue")
            for name, unit, source, method in MITDB_INDICATORS
        ]
    if record_type == "bidmc":
        descriptors = [
            (
                "HR",
                "bpm",
                BIDMC_ECG_SOURCE,
                "60 / mean RR of R-peaks detected on ECG lead II; see docs/crosscheck.md",
            )
        ]
        descriptors.extend(
            (
                name,
                unit,
                BIDMC_MONITOR_SOURCE,
                "mean of non-missing 1 Hz samples inside each window",
            )
            for name, unit in MONITOR_CHANNELS
        )
        descriptors.append(
            (
                "RESP",
                "pm",
                BIDMC_WAVE_RESP_SOURCE,
                "no validated breath detector exists in this prototype",
            )
        )
        return [
            _indicator(name, unit, source, method, [], status="data_issue")
            for name, unit, source, method in descriptors
        ]
    return []


def _build_indicators(path, record_type, params):
    if record_type == "mitdb":
        return _mitdb_indicators(path, params)
    if record_type == "bidmc":
        return _bidmc_indicators(path, params)
    return []


def _rule_results(indicators, rules, data_quality_status):
    """Aggregate each rule over indicator windows, keeping source identity.

    Indicators are identified by (name, source), not by name alone: MIT-BIH has
    two HR indicators with different sources. The overall rule status still
    aggregates every matching window, while flagged_indicator_sources records
    exactly which sources produced a flag so the UI never propagates a flag
    from one HR source to another.
    """
    results = []
    for rule in rules:
        if data_quality_status != "valid":
            status = "not_evaluated"
            reason = f"record data quality is invalid, so this rule was not evaluated | {DISCLAIMER}"
            results.append(_rule_dict(rule, status, reason, 0, 0, [], []))
            continue

        matching = [ind for ind in indicators if ind["name"]
                    == rule["indicator"]]
        if not matching:
            status = "not_evaluated"
            reason = f"no {rule['indicator']} indicator in this record | {DISCLAIMER}"
            results.append(_rule_dict(rule, status, reason, 0, 0, [], []))
            continue

        n_evaluated = 0
        n_flagged = 0
        starts = []
        flagged_sources = []
        for indicator in matching:
            source = indicator["source"]
            source_flagged = False
            for window in indicator["windows"]:
                if window["status"] != "evaluated" or window["value"] is None:
                    continue
                outcome = evaluate(rule, window["value"], indicator["unit"])
                if outcome.status == "not_evaluated":
                    continue
                n_evaluated += 1
                if outcome.status == "flag":
                    n_flagged += 1
                    source_flagged = True
                    if len(starts) < 20:
                        starts.append(window["start_s"])
            if source_flagged and source not in flagged_sources:
                flagged_sources.append(source)

        if n_evaluated == 0:
            status = "not_evaluated"
            reason = f"no evaluated windows for {rule['indicator']} | {DISCLAIMER}"
        elif n_flagged:
            status = "flag"
            reason = (
                f"{rule.get('message', rule['id'])} | {n_flagged} of {n_evaluated} "
                f"evaluated windows flagged | {DISCLAIMER}"
            )
        else:
            status = "no_flag"
            reason = (
                f"{rule.get('message', rule['id'])} | none of {n_evaluated} "
                f"evaluated windows flagged | {DISCLAIMER}"
            )
        results.append(
            _rule_dict(
                rule, status, reason, n_evaluated, n_flagged, starts, flagged_sources
            )
        )
    return results


def _rule_dict(rule, status, reason, n_evaluated, n_flagged, starts, flagged_sources):
    return {
        "rule_id": rule["id"],
        "indicator": rule["indicator"],
        "rule_status": rule["status"],
        "source": rule["source"],
        "status": status,
        "reason": reason,
        "n_windows_evaluated": n_evaluated,
        "n_windows_flagged": n_flagged,
        "flagged_window_starts": starts,
        "flagged_indicator_sources": flagged_sources,
    }


# ----------------------------------------------------------------- public API


def analyze(path, *, data_type, source_label):
    """Analyse one record and return the analysis dict.

    `data_type` must be "actual_research_data" or "simulated"; synthetic data is
    never mixed into results quietly.
    """
    params = load_params()
    rules = load_rules()
    path = str(path)

    report = validate_record(path)
    info = _header_info(path)
    record_type = _record_type(path, info)

    limitations = list(LIMITATIONS)
    if record_type == "mitdb":
        limitations.append(
            "Detector-derived beats carry no rhythm labels, so RMSSD is never computed from them."
        )
    if record_type == "bidmc":
        limitations.append(
            "Monitor numerics and ECG-derived values come from different sources and are "
            "reported separately, never merged; rhythm labels are not available for BIDMC."
        )
    if record_type == "unknown":
        limitations.append(
            "Record type was not recognised, so no indicators were produced.")

    if report.status == "valid":
        indicators = _build_indicators(path, record_type, params)
        if record_type not in {"mitdb", "bidmc"}:
            limitations.append(
                "No indicators are produced for this record type by this prototype build."
            )
    else:
        indicators = _data_issue_indicators(record_type)

    return {
        "dataset": _dataset(path, info, record_type, data_type, source_label),
        "data_quality": {
            "status": report.status,
            "issues": [_issue_dict(issue) for issue in report.issues],
            "not_checked": list(report.not_checked),
        },
        "rhythm_context": _rhythm_context(path, record_type, info, params),
        "indicators": indicators,
        "rule_results": _rule_results(indicators, rules, report.status),
        "model_results": {"status": "not_run", "warnings": []},
        "limitations": limitations,
    }


def make_simulated_fixture():
    """A hand-built SIMULATED analysis dict showing every dashboard state."""
    rules = load_rules()

    hr_windows_fixture = [
        _window(0.0, 10.0, 80.0, "evaluated",
                "SIMULATED window, value invented"),
        _window(10.0, 20.0, 81.0, "evaluated",
                "SIMULATED window, value invented"),
    ]
    monitor_hr_windows = [
        _window(
            0.0,
            10.0,
            400.0,
            "evaluated",
            "SIMULATED monitor value above the engineered sensor range, to show a "
            "plausibility flag",
        )
    ]
    pulse_windows = [
        _window(0.0, 10.0, 72.0, "evaluated", "SIMULATED value, invented"),
    ]
    rmssd_windows_fixture = [
        _window(
            0.0,
            300.0,
            None,
            "not_evaluated",
            "SIMULATED: rhythm label '(AFIB' is not in the eligibility gate "
            "(configured condition, not a diagnosis)",
        )
    ]

    indicators = [
        _indicator(
            "HR",
            "bpm",
            "SIMULATED annotation file beats",
            "60 / mean RR of simulated beats inside each window",
            hr_windows_fixture,
        ),
        _indicator(
            "HR",
            "bpm",
            "SIMULATED monitor numerics",
            "mean of simulated monitor samples inside each window",
            monitor_hr_windows,
        ),
        _indicator(
            "RMSSD",
            "ms",
            "SIMULATED annotation file beats",
            "gated RMSSD over simulated rhythm labels",
            rmssd_windows_fixture,
        ),
        _indicator(
            "SpO2",
            "%",
            "SIMULATED monitor numerics",
            "mean of non-missing simulated samples inside each window",
            [],
            status="data_issue",
        ),
        _indicator(
            "PULSE",
            "bpm",
            "SIMULATED monitor numerics",
            "mean of simulated monitor samples inside each window",
            pulse_windows,
        ),
    ]

    return {
        "dataset": {
            "name": DATASET_NAMES["simulated"],
            "record": "sim-001",
            "version_label": "hand-built demonstration fixture, SIMULATED",
            "data_type": "simulated",
            "fs": 125.0,
            "duration_s": 60.0,
        },
        "data_quality": {
            "status": "valid",
            "issues": [
                {
                    "code": "SIMULATED_CHANNEL_FAILURE",
                    "severity": "warning",
                    "message": (
                        "SIMULATED: the SpO2 channel is marked data_issue to demonstrate "
                        "that state; no person was monitored"
                    ),
                    "field": "SpO2",
                }
            ],
            "not_checked": [],
        },
        "rhythm_context": {
            "source": "SIMULATED annotation file; labels invented for the fixture",
            "segments": [
                {"start_s": 0.0, "end_s": 300.0, "label": "(AFIB"},
            ],
        },
        "indicators": indicators,
        "rule_results": _rule_results(indicators, rules, "valid"),
        "model_results": {"status": "not_run", "warnings": []},
        "limitations": LIMITATIONS
        + [
            "SIMULATED data: every value in this fixture is invented to exercise the UI; "
            "no person was monitored.",
        ],
    }
