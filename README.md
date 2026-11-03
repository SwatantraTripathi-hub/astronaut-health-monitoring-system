# Astronaut Health Monitoring System (research prototype)

Team Syntax Debuggers · NASA Space Apps Challenge 2026

> **Not diagnostic software. Not NASA-endorsed. Not validated for clinical use
> or operational astronaut care.**

## What it does

A pipeline that validates health signals, computes indicators, and shows each
indicator's status as **Evaluated / Rule flag / Not evaluated / Data issue**,
with data source and limitations shown next to every result.

## Status

Work in progress (started 28 Oct 2026). Nothing below is claimed until it is built and tested.

## Setup

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Data

Raw datasets are not committed. See `data/README.md`.

## Run

    python -m streamlit run src/app.py

Setup first: activate the venv and `pip install -r requirements.txt`. The dashboard offers
the SIMULATED fixture, every record present in `data/raw/mitdb` and `data/raw/bidmc`, and
the small PhysioNet record catalog in `src/ingest/datasets.py` (downloaded on first
selection when it is not already local). Raw datasets are still never committed.

Checks used during development:

    python -m pytest -q
    python scripts/gate.py

## Tests

    pytest

## Limitations

- Research prototype: not a diagnosis, not a medical device, not endorsed by NASA, and not
  validated for clinical use or operational astronaut care.
- Data is terrestrial/hospital data (MIT-BIH, BIDMC), not astronaut data. MIT-BIH records
  201 and 202 are the same subject.
- A "no flag" result does not mean healthy; every rule is a configured condition.
- Window lengths, minimum counts and rhythm eligibility labels are design parameters with
  no clinical source (`configs/params.yaml`).
- Rhythm labels are dataset annotations, not output of this software; BIDMC has none.
- RMSSD is only computed inside rhythm labels this prototype is willing to use (currently
  only `(N`); on record 201's annotated `(AFIB` stretch it is not evaluated.
- The R-peak detector and the BIDMC cross-check are exploratory; see the tables below.
- No anomaly model runs, and no combined health score exists anywhere in the output.
- BIDMC numbers cover bidmc01 only; bidmc02-bidmc05 are not downloaded.

## Architecture (implemented)

    source -> ingest -> validate -> features -> (rules | optional ML) -> JSON output -> dashboard

- ingest: `src/ingest/wfdb_loader.py`, `src/ingest/rhythm.py` - implemented
- validate: `src/validate/validator.py` - implemented
- features: `src/features/rr.py`, `hrv.py`, `hr.py`, `rpeaks.py`, `crosscheck.py` - implemented
- config: `configs/params.yaml`, `src/config.py` - implemented
- rules: `configs/rules.yaml`, `src/rules/engine.py` - implemented (plausibility checks and
  illustrative demo rules only)
- pipeline: `src/pipeline.py` - implemented
- dashboard: `src/app.py` - implemented
- optional ML: not implemented; `model_results` is always
  `{"status": "not_run", "warnings": []}`

## Results (exploratory)

The numbers below are copied from `docs/rpeak_eval.md` and `docs/crosscheck.md`.
They are EXPLORATORY, on terrestrial data, and are not clinical results.

R-peak detector versus the MIT-BIH annotation file (channel MLII):

| record | fs (Hz) | duration (s) | beats annotated | peaks detected | TP | FP | FN | sensitivity | PPV |
|--------|---------|--------------|-----------------|----------------|----|----|----|-------------|-----|
| 100 | 360 | 1805.6 | 2273 | 2273 | 2273 | 0 | 0 | 1.0000 | 1.0000 |
| 201 | 360 | 1805.6 | 1963 | 1980 | 1939 | 41 | 24 | 0.9878 | 0.9793 |
| 202 | 360 | 1805.6 | 2136 | 2138 | 2130 | 8 | 6 | 0.9972 | 0.9963 |

ECG-derived HR versus the monitor's HR numerics on BIDMC bidmc01
(10 s windows fully inside both recordings):

| record | status | median absolute difference (bpm) | windows compared | windows skipped |
|--------|--------|----------------------------------|------------------|-----------------|
| bidmc01 | evaluated | 0.33 | 48 | 0 |

Caveats for both tables: exploratory; records 201 and 202 are the same subject; the
BIDMC check is a plausibility check on one ICU record, not an accuracy validation.

## Team

Syntax Debuggers
