# CONTEXT: Astronaut Health Monitoring System (Syntax Debuggers)

## What this is

Research prototype for NASA Space Apps 2026. NOT diagnostic, NOT NASA-endorsed.
Deadline: 7 Nov 2026. One active developer. Learning matters as much as shipping.

## One-sentence goal

Pipeline that validates health signals, computes indicators (HR, RMSSD),
and shows per-indicator status: Evaluated / Rule flag / Not evaluated / Data issue,
with provenance and limitations.

## Hard rules (never break)

1. Missing/unsupported data = "Not evaluated", never "normal".
2. Keep three outputs separate: data quality, rule flags, ML warnings. No fused score.
3. No invented thresholds, fields, or metrics. Demo rules labelled ILLUSTRATIVE.
4. Synthetic data always labelled SIMULATED.
5. No claim of diagnosis, accuracy %, or NASA approval.
6. Don't claim tests passed or files inspected unless actually run.

## Current decisions (update as they change)

- Datasets: MIT-BIH (primary), BIDMC (secondary, UNVERIFIED), MIMIC-IV dropped, OSD-343 timeboxed.
- Stack: Python core library + Streamlit dashboard. Rules in YAML.
- ML: optional Isolation Forest, one-day timebox.

## Status

(Done / In progress / Blocked, update daily)

## Architecture (proposed)

source -> ingest -> validate -> features -> [rules | optional ML] -> JSON output -> dashboard

## Folder map

src/ingest, src/validate, src/features, src/rules, src/ml, app/, tests/, data/, docs/

## How I work with AI

- I write function signatures and one test first, AI helps with bodies.
- State assumptions, make the smallest change, show how to run it.
- Explain concepts in plain English or simple Hinglish.

## Open questions

list things not yet verified
