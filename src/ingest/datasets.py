"""PhysioNet record catalog and on-demand download for the dashboard.

Raw datasets are intentionally not committed to this repository. This module
lists only the records this prototype was built and tested against, and fetches
the files for a record from PhysioNet the first time it is selected. Nothing
here invents records, telemetry, or annotations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import wfdb


class DatasetUnavailableError(RuntimeError):
    """Raised when a selected record cannot be made available locally."""


@dataclass(frozen=True)
class DatasetRecord:
    kind: str
    name: str
    files: tuple[str, ...]
    required: tuple[str, ...]

    @property
    def choice(self) -> str:
        return f"{self.kind}/{self.name}"


CATALOG = (
    DatasetRecord(
        "mitdb",
        "100",
        ("100.dat", "100.hea", "100.atr"),
        ("100.dat", "100.hea", "100.atr"),
    ),
    DatasetRecord(
        "mitdb",
        "201",
        ("201.dat", "201.hea", "201.atr"),
        ("201.dat", "201.hea", "201.atr"),
    ),
    DatasetRecord(
        "mitdb",
        "202",
        ("202.dat", "202.hea", "202.atr"),
        ("202.dat", "202.hea", "202.atr"),
    ),
    DatasetRecord(
        "bidmc",
        "bidmc01",
        (
            "bidmc01.dat",
            "bidmc01.hea",
            "bidmc01n.dat",
            "bidmc01n.hea",
            "bidmc01.breath",
            "LICENSE",
        ),
        ("bidmc01.dat", "bidmc01.hea", "bidmc01n.dat", "bidmc01n.hea"),
    ),
)

CATALOG_BY_CHOICE = {record.choice: record for record in CATALOG}


def catalog_choices() -> List[str]:
    """Return every record this prototype knows how to analyse."""
    return [record.choice for record in CATALOG]


def get_catalog_record(choice: str) -> DatasetRecord | None:
    return CATALOG_BY_CHOICE.get(choice)


def record_base(raw_dir, record: DatasetRecord) -> Path:
    return Path(raw_dir) / record.kind


def record_path(raw_dir, record: DatasetRecord) -> Path:
    return record_base(raw_dir, record) / record.name


def is_record_available(raw_dir, record: DatasetRecord) -> bool:
    base = record_base(raw_dir, record)
    return all((base / name).is_file() for name in record.required)


def discover_local_choices(raw_dir) -> List[str]:
    """List records already present in data/raw, including extra downloads."""
    choices = []
    for kind in ("mitdb", "bidmc"):
        base = Path(raw_dir) / kind
        if not base.exists():
            continue
        for header in sorted(base.glob("*.hea")):
            if kind == "bidmc" and header.stem.endswith("n"):
                continue
            choices.append(f"{kind}/{header.stem}")
    return choices


def record_choices(raw_dir) -> List[str]:
    """Catalog records first, then any additional local records."""
    choices = catalog_choices()
    for choice in discover_local_choices(raw_dir):
        if choice not in choices:
            choices.append(choice)
    return choices


def ensure_record(raw_dir, choice: str) -> Path:
    """Return a local record path, downloading missing catalog files if needed."""
    record = get_catalog_record(choice)
    if record is None:
        try:
            kind, name = choice.split("/", 1)
        except ValueError as exc:
            raise DatasetUnavailableError(
                f"Invalid record choice {choice!r}") from exc
        base = Path(raw_dir) / kind
        required = [f"{name}.hea", f"{name}.dat"]
        if kind == "mitdb":
            required.append(f"{name}.atr")
        elif kind == "bidmc":
            required.extend([f"{name}n.hea", f"{name}n.dat"])
        if not all((base / file_name).is_file() for file_name in required):
            raise DatasetUnavailableError(
                f"Record {choice!r} is not available in the local data directory")
        return base / name

    if is_record_available(raw_dir, record):
        return record_path(raw_dir, record)

    base = record_base(raw_dir, record)
    try:
        base.mkdir(parents=True, exist_ok=True)
        wfdb.dl_files(
            record.kind,
            str(base),
            list(record.files),
            keep_subdirs=False,
            overwrite=False,
        )
    except Exception as exc:
        raise DatasetUnavailableError(
            f"Could not download {choice} from PhysioNet: {exc}") from exc

    if not is_record_available(raw_dir, record):
        raise DatasetUnavailableError(
            f"Downloaded files for {choice} are incomplete")
    return record_path(raw_dir, record)
