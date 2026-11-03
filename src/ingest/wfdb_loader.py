from dataclasses import dataclass
import os
from typing import Dict, List

import numpy as np
import wfdb


class RecordLoadError(Exception):
    pass


class SignalNotFound(Exception):
    pass


# markers seen in records 100/201/202 and described by wfdb's ann_label_table as
# non-beat events; extend only after reading the MIT-BIH docs.
NON_BEAT_SYMBOLS = {"+", "~", "|", "x"}
MAX_SIGNALS = 128
MAX_SAMPLES = 50_000_000
MAX_ANNOTATIONS = 50_000_000


@dataclass
class LoadedRecord:
    record_name: str
    base_path: str
    fs: int | float
    sig_len: int
    duration_s: float
    signal_names: List[str]
    units: Dict[str, str]
    signals: Dict[str, np.ndarray]
    comments: List[str]
    data_type: str = "actual_research"
    source_label: str = ""


def normalize_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = s.strip(",")
    return s


def _record_path(path) -> str:
    """Normalize a WFDB base path and reject unsafe path values."""
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise RecordLoadError("record path must be path-like") from exc
    if not isinstance(value, str) or not value.strip():
        raise RecordLoadError("record path must be a non-empty string")
    if "\x00" in value:
        raise RecordLoadError("record path contains a NUL byte")
    if len(value) > 4096:
        raise RecordLoadError("record path is too long")
    return value


def _validated_signal(rec, index: int, sig_len: int) -> np.ndarray:
    """Extract one channel only after checking shape and bounded length."""
    for field in ("p_signal", "d_signal"):
        values = getattr(rec, field, None)
        if values is None:
            continue
        try:
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[0] != sig_len or index >= array.shape[1]:
                raise RecordLoadError(
                    f"WFDB {field} has an invalid signal shape")
            return np.asarray(array[:, index], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RecordLoadError(
                f"WFDB {field} contains non-numeric signal data") from exc
    raise RecordLoadError("WFDB record contains no readable signal data")


def load_record(path, *, data_type="actual_research", source_label=""):
    """Load one bounded, structurally valid WFDB record."""
    path = _record_path(path)
    try:
        rec = wfdb.rdrecord(path)
    except Exception as e:
        raise RecordLoadError(f"Failed to load record from {path}: {e}") from e

    try:
        fs = float(rec.fs)
        sig_len = int(rec.sig_len)
        raw_names = list(rec.sig_name)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RecordLoadError(
            "WFDB record has invalid required metadata") from exc
    if not np.isfinite(fs) or fs <= 0 or fs > 1_000_000:
        raise RecordLoadError(
            f"WFDB sampling frequency is out of bounds: {fs!r}")
    if sig_len <= 0 or sig_len > MAX_SAMPLES:
        raise RecordLoadError(
            f"WFDB sample count is out of bounds: {sig_len!r}")
    if not raw_names or len(raw_names) > MAX_SIGNALS:
        raise RecordLoadError("WFDB channel count is out of bounds")

    signal_names = []
    units = {}
    signals = {}

    for i, name in enumerate(raw_names):
        nname = normalize_name(name)
        if not nname or len(nname) > 128:
            raise RecordLoadError(f"WFDB channel name at index {i} is invalid")
        if nname in signals:
            raise RecordLoadError(
                f"WFDB record has duplicate channel name {nname!r}")
        signal_names.append(nname)
        units[nname] = rec.units[i] if i < len(rec.units) else ""
        signals[nname] = _validated_signal(rec, i, sig_len)

    duration_s = sig_len / fs
    return LoadedRecord(
        record_name=rec.record_name,
        base_path=str(path),
        fs=fs,
        sig_len=sig_len,
        duration_s=duration_s,
        signal_names=signal_names,
        units=units,
        signals=signals,
        comments=list(rec.comments) if hasattr(
            rec, "comments") and rec.comments is not None else [],
        data_type=data_type,
        source_label=source_label,
    )


def get_signal(rec: LoadedRecord, name: str):
    if rec is None:
        raise SignalNotFound("record is None")
    nname = normalize_name(name)
    if nname in rec.signals:
        return rec.signals[nname]
    lower_map = {k.lower(): v for k, v in rec.signals.items()}
    if nname.lower() in lower_map:
        return lower_map[nname.lower()]
    raise SignalNotFound(
        f"Signal '{name}' not found in record {rec.record_name}")


def load_beat_annotations(path, ext="atr", *, sig_len: int | None = None):
    path = _record_path(path)
    if not isinstance(ext, str) or not ext or len(ext) > 16 or not ext.isalnum():
        raise RecordLoadError("annotation extension is invalid")
    try:
        ann = wfdb.rdann(path, ext)
    except Exception as e:
        raise RecordLoadError(
            f"Failed to load annotations from {path}.{ext}: {e}") from e
    try:
        samples = np.asarray(ann.sample)
        if samples.ndim != 1 or samples.size > MAX_ANNOTATIONS:
            raise RecordLoadError("annotation sample array is out of bounds")
        if not np.issubdtype(samples.dtype, np.integer):
            if not np.all(np.isfinite(samples)) or not np.all(samples == np.floor(samples)):
                raise RecordLoadError(
                    "annotation samples must be finite integers")
        if np.any(samples < 0):
            raise RecordLoadError("annotation samples cannot be negative")
        if sig_len is not None and (sig_len <= 0 or np.any(samples >= sig_len)):
            raise RecordLoadError(
                "annotation sample is outside the record bounds")
        symbols = list(ann.symbol)
        if len(symbols) != samples.size:
            raise RecordLoadError(
                "annotation samples and symbols have different lengths")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RecordLoadError("annotation data is malformed") from exc
    aux_notes = list(getattr(ann, "aux_note", []))
    return samples, symbols, aux_notes


def beat_only(samples, symbols):
    out_samples = []
    out_symbols = []
    for s, sym in zip(samples, symbols):
        if sym in NON_BEAT_SYMBOLS:
            continue
        out_samples.append(s)
        out_symbols.append(sym)
    return np.array(out_samples), out_symbols
