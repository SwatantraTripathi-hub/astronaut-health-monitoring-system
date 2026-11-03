import os
import re

from dataclasses import dataclass
from typing import List

import numpy as np
import wfdb


@dataclass
class Issue:
    code: str
    severity: str  # "error","warning","info"
    message: str
    field: str | None = None


@dataclass
class ValidationReport:
    status: str  # "valid","invalid"
    issues: List[Issue]
    checks_run: List[str]
    not_checked: List[str]


def nan_fraction(array):
    arr = np.array(array)
    if arr.size == 0:
        return 0.0
    return float(np.isnan(arr).sum() / arr.size)


def _normalize_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = s.strip(",")
    return s


def _header_format_to_bytes_per_sample(fmt):
    try:
        f = str(fmt).strip()
    except Exception:
        return None
    if f == "16":
        return 2.0
    if f == "212":
        return 1.5
    return None


def re_has_question_sex(cstr):
    cl = str(cstr).lower()
    if "sex" in cl and "?" in cl:
        return True
    return False


# Only an explicit empty value counts as missing metadata: "<age>: nan" or "<sex>: ?".
# The bare word "nan" elsewhere in a comment is not a missing-metadata signal.
AGE_MISSING_RE = re.compile(r"<age>:\s*nan", re.IGNORECASE)
SEX_MISSING_RE = re.compile(r"<sex>:\s*\?", re.IGNORECASE)


def validate_record(path):
    issues = []
    checks_run = []
    not_checked = []
    base = str(path)
    hea = base + ".hea"
    dat = base + ".dat"
    try:
        checks_run.append("header_read")
        rec = wfdb.rdheader(base)
    except Exception:
        issues.append(Issue(code="HEADER_UNREADABLE", severity="error", message="Header could not be read"))
        if not os.path.exists(hea):
            issues.append(Issue(code="FILE_MISSING", severity="error", message=".hea missing", field="hea"))
        if not os.path.exists(dat):
            issues.append(Issue(code="FILE_MISSING", severity="error", message=".dat missing", field="dat"))
        return ValidationReport(status="invalid", issues=issues, checks_run=checks_run, not_checked=not_checked)

    if not os.path.exists(hea):
        issues.append(Issue(code="FILE_MISSING", severity="error", message=".hea missing", field="hea"))
    if not os.path.exists(dat):
        issues.append(Issue(code="FILE_MISSING", severity="error", message=".dat missing", field="dat"))
    if any(i.severity == "error" for i in issues):
        return ValidationReport(status="invalid", issues=issues, checks_run=checks_run, not_checked=not_checked)

    checks_run.append("fs")
    try:
        fs = rec.fs
    except Exception:
        fs = None
    if fs is None or fs <= 0:
        issues.append(Issue(code="BAD_FS", severity="error", message="Invalid fs", field="fs"))

    checks_run.append("sig_len")
    try:
        sig_len = rec.sig_len
    except Exception:
        sig_len = None
    if sig_len == 0:
        issues.append(Issue(code="EMPTY_SIGNAL", severity="error", message="sig_len is 0", field="sig_len"))

    checks_run.append("units")
    try:
        units = rec.units
        for i, u in enumerate(units):
            if u is None or str(u).strip() == "":
                issues.append(Issue(code="UNIT_MISSING", severity="warning", message="Unit missing", field=f"units[{i}]"))
                break
    except Exception:
        issues.append(Issue(code="UNIT_MISSING", severity="warning", message="Unit missing"))

    checks_run.append("names")
    try:
        names = []
        for n in rec.sig_name:
            names.append(_normalize_name(n))
        if len(names) != len(set(names)):
            issues.append(Issue(code="DUPLICATE_SIGNAL_NAME", severity="error", message="Duplicate signal names after normalization", field="sig_name"))
    except Exception:
        pass

    try:
        checks_run.append("size")
        n_sig = len(rec.sig_name) if hasattr(rec, "sig_name") else 0
        dat_size = os.path.getsize(dat)
        fmt_list = []
        try:
            fmt_list = rec.fmt
        except Exception:
            pass
        if fmt_list:
            fmt0 = fmt_list[0]
            bps = _header_format_to_bytes_per_sample(fmt0)
            if bps is None:
                issues.append(Issue(code="SIZE_CHECK_SKIPPED", severity="warning", message="Unknown format, cannot verify size"))
                not_checked.append("size")
            else:
                expected = n_sig * sig_len * bps if (n_sig is not None and sig_len is not None) else None
                if expected is not None:
                    if abs(float(expected) - float(dat_size)) > 1e-6:
                        if round(float(expected)) != int(dat_size):
                            issues.append(Issue(code="SIZE_MISMATCH", severity="error", message=f"Size mismatch: expected ~{expected} got {dat_size}", field="dat"))
        else:
            issues.append(Issue(code="SIZE_CHECK_SKIPPED", severity="warning", message="No format info"))
            not_checked.append("size")
    except Exception:
        issues.append(Issue(code="SIZE_CHECK_SKIPPED", severity="warning", message="Size check failed"))
        not_checked.append("size")

    try:
        checks_run.append("nan")
        try:
            rfull = wfdb.rdrecord(base)
            if hasattr(rfull, "p_signal") and rfull.p_signal is not None:
                sig_names = list(getattr(rfull, "sig_name", []) or [])
                for i in range(rfull.p_signal.shape[1]):
                    arr = rfull.p_signal[:, i]
                    n_nan = int(np.isnan(arr).sum())
                    if n_nan == 0:
                        continue
                    frac = nan_fraction(arr)
                    name = _normalize_name(sig_names[i]) if i < len(sig_names) else f"signal[{i}]"
                    issues.append(
                        Issue(
                            code="NAN_PRESENT",
                            severity="warning",
                            message=(
                                f"NaN present in {name}: {n_nan} of {arr.size} samples "
                                f"(fraction={frac:.6f}); missing values are not treated as normal"
                            ),
                            field=name,
                        )
                    )
        except Exception:
            pass
    except Exception:
        pass

    try:
        checks_run.append("metadata")
        comments = getattr(rec, "comments", []) or []
        age_reported = False
        sex_reported = False
        for c in comments:
            cstr = str(c)
            if not age_reported and AGE_MISSING_RE.search(cstr):
                issues.append(Issue(code="METADATA_MISSING", severity="info", message="Metadata missing: age NaN", field="comments"))
                age_reported = True
            if not sex_reported and SEX_MISSING_RE.search(cstr):
                issues.append(Issue(code="METADATA_MISSING", severity="info", message="Metadata missing: sex ?", field="comments"))
                sex_reported = True
    except Exception:
        pass

    if any(i.severity == "error" for i in issues):
        return ValidationReport(status="invalid", issues=issues, checks_run=checks_run, not_checked=not_checked)
    return ValidationReport(status="valid", issues=issues, checks_run=checks_run, not_checked=not_checked)
