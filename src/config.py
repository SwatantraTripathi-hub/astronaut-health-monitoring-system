"""Load and validate prototype parameters before the pipeline starts."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_PATH = Path(__file__).resolve().parent.parent / \
    "configs" / "params.yaml"

# Every key the code depends on. A missing key is a hard error: the prototype
# must never silently fall back to an invented value.
REQUIRED_KEYS = [
    "hrv.window_s",
    "hrv.min_differences",
    "hrv.min_coverage",
    "hr.window_s",
    "hr.step_s",
    "rpeaks.bandpass_low_hz",
    "rpeaks.bandpass_high_hz",
    "rpeaks.integration_ms",
    "rpeaks.refractory_ms",
    "rpeaks.match_tolerance_ms",
    "rpeaks.threshold_fraction",
    "rpeaks.threshold_window_s",
    "crosscheck.window_s",
    "numerics.min_valid_fraction",
]


class ConfigError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HrvParams(_StrictModel):
    window_s: float = Field(gt=0)
    min_differences: int = Field(ge=1)
    min_coverage: float = Field(ge=0, le=1)


class HrParams(_StrictModel):
    window_s: float = Field(gt=0)
    step_s: float = Field(gt=0)


class RpeakParams(_StrictModel):
    bandpass_low_hz: float = Field(gt=0)
    bandpass_high_hz: float = Field(gt=0)
    integration_ms: float = Field(gt=0)
    refractory_ms: float = Field(gt=0)
    match_tolerance_ms: float = Field(gt=0)
    threshold_fraction: float = Field(gt=0, le=1)
    threshold_window_s: float = Field(gt=0)


class CrosscheckParams(_StrictModel):
    window_s: float = Field(gt=0)


class NumericsParams(_StrictModel):
    min_valid_fraction: float = Field(ge=0, le=1)


class Params(_StrictModel):
    hrv: HrvParams
    hr: HrParams
    rpeaks: RpeakParams
    crosscheck: CrosscheckParams
    numerics: NumericsParams


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``params.yaml`` and return its validated mapping.

    The mapping return type preserves the existing pipeline API while Pydantic
    rejects unknown keys, missing sections, invalid ranges, and YAML scalars
    with the wrong type at startup.
    """
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read params file {target}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ConfigError(f"params file {target} must contain a mapping")
        return Params.model_validate(data).model_dump()
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"params file {target} is not valid YAML: {exc}") from exc
    except ValidationError as exc:
        raise ConfigError(f"invalid parameters in {target}: {exc}") from exc
