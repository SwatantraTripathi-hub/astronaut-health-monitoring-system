#!/usr/bin/env python
"""Phase gate for this project. Standard library only.

Runs the checks defined in docs/FREEBUFF_PROMPT.md (Phase A, item A1).
Every failure is printed as `FAIL: <reason>`; the last line is exactly
`GATE: PASS` or `GATE: FAIL`.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = "pre-freebuff"

PROTECTED = [
    "scripts/download_data.py",
    "scripts/plot_ecg.py",
    "scripts/inspect_record.py",
    "scripts/show_rhythm_notes.py",
    "docs/inventory.md",
    "docs/day1_*.md",
    "docs/day2_*.md",
    "CONTEXT.md",
    "AGENTS.md",
    "LEARNING_LOG.md",
    "DECISIONS.md",
    "data/",
]

BANNED_PHRASES = [
    "diagnosis",
    "NASA-approved",
    "100% accurate",
    "guarantees safety",
    "real-time astronaut monitoring",
]

GOLDEN = [
    (
        "100",
        [
            "RMSSD (NN only): 19.91 ms from 8 differences",
            "RMSSD (all beats): 124.05 ms",
        ],
    ),
    ("201", ["RMSSD (NN only): 159.54 ms from 12 differences"]),
]

SCAN_DIRS = ["src", "app", "config"]
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".txt",
                 ".md", ".html", ".css", ".js", ".toml", ".cfg", ".ini"}

failures = []


def fail(reason):
    failures.append(reason)


def run(cmd, **kwargs):
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )


def git(*args):
    return run(["git", *args])


def one_line(text, limit=300):
    flat = " | ".join(line.strip()
                      for line in str(text).splitlines() if line.strip())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def tail(text, lines=6):
    parts = [line for line in str(text).splitlines() if line.strip()]
    return one_line("\n".join(parts[-lines:]))


def tag_ok():
    return git("rev-parse", "--verify", f"refs/tags/{TAG}").returncode == 0


def check_pytest():
    result = run([sys.executable, "-m", "pytest",
                 "-q", "-p", "no:cacheprovider"])
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"(\d+) passed", output)
    if result.returncode != 0:
        fail(f"pytest failed (exit {result.returncode}): {tail(output)}")
        return
    if not match:
        fail(
            f"pytest exited 0 but no 'N passed' line was found: {tail(output)}")
        return
    passed = int(match.group(1))
    min_file = ROOT / "docs" / "min_tests.txt"
    try:
        minimum = int(min_file.read_text(encoding="utf-8").strip())
    except Exception:
        fail("docs/min_tests.txt is missing or does not contain an integer")
        return
    if passed < minimum:
        fail(f"only {passed} tests passed, docs/min_tests.txt requires {minimum}")
    else:
        print(f"OK: pytest {passed} passed (minimum {minimum})")


def check_assert_counts():
    listing = git("ls-tree", "-r", "--name-only", TAG, "--", "tests/")
    if listing.returncode != 0:
        fail(f"cannot list tests/ at tag {TAG}")
        return
    files = [line.strip()
             for line in listing.stdout.splitlines() if line.strip()]
    if not files:
        fail(f"tag {TAG} has no test files to compare against")
        return
    for name in files:
        old = git("show", f"{TAG}:{name}")
        if old.returncode != 0:
            continue
        path = ROOT / name
        if not path.exists():
            fail(f"{name} existed at tag {TAG} but was deleted")
            continue
        old_count = sum(1 for line in old.stdout.splitlines()
                        if "assert" in line)
        new_count = sum(
            1
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "assert" in line
        )
        if new_count < old_count:
            fail(f"{name}: {new_count} assert lines now, {old_count} at tag {TAG}")
    print(f"OK: assert counts not reduced in {len(files)} tagged test files")


def check_protected():
    diff = git("diff", "--quiet", TAG, "--", *PROTECTED)
    if diff.returncode == 0:
        print("OK: protected files unchanged")
        return
    changed = git("diff", "--name-only", TAG, "--", *PROTECTED).stdout
    fail(f"protected files changed vs {TAG}: {one_line(changed)}")


def check_golden():
    if not (ROOT / "data" / "raw" / "mitdb" / "100.hea").exists():
        print("SKIP: data/raw absent")
        return
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    for record, needles in GOLDEN:
        result = run([sys.executable, "scripts/plot_ecg.py", record], env=env)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0:
            fail(
                f"plot_ecg.py {record} exited {result.returncode}: {tail(output)}")
            continue
        for needle in needles:
            if needle not in output:
                fail(f"golden line missing for record {record}: {needle!r}")
    print("OK: golden RMSSD regression checked (records 100 and 201)")


def check_banned_phrases():
    hits = 0
    for dirname in SCAN_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "not " in line or "NOT " in line:
                    continue
                lowered = line.lower()
                for phrase in BANNED_PHRASES:
                    if phrase.lower() in lowered:
                        rel = path.relative_to(ROOT).as_posix()
                        fail(f"banned phrase {phrase!r} in {rel}:{lineno}")
                        hits += 1
                        break
    if hits == 0:
        print("OK: no banned phrases in src/, app/, configs/")


def check_rules_status():
    path = ROOT / "configs" / "rules.yaml"
    if not path.exists():
        print("OK: configs/rules.yaml not present, nothing to check")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        fail("configs/rules.yaml cannot be read as UTF-8")
        return
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^\s*(?:-\s*)?status\s*:\s*(.+)$", line)
        if not match:
            continue
        value = match.group(1).split("#")[0].strip().strip("'\"").lower()
        if value == "clinical":
            fail(
                f"configs/rules.yaml:{lineno} uses forbidden status 'clinical'")


def check_requirements():
    path = ROOT / "requirements.txt"
    if not path.exists():
        fail("requirements.txt is missing")
        return
    raw = path.read_bytes()
    if b"\x00" in raw:
        fail("requirements.txt contains NUL bytes (probably UTF-16)")
        return
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"requirements.txt is not valid UTF-8: {exc}")
        return
    print("OK: requirements.txt is UTF-8 without NUL bytes")


def main():
    if not tag_ok():
        fail("run: git tag pre-freebuff")
    else:
        check_pytest()
        check_assert_counts()
        check_protected()
    check_golden()
    check_banned_phrases()
    check_rules_status()
    check_requirements()

    for reason in failures:
        print(f"FAIL: {reason}")
    print("GATE: FAIL" if failures else "GATE: PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
