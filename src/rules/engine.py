"""Rule engine: configured conditions over single indicator values.

A rule is a condition someone configured. It is not a health judgement and a
"no_flag" result does not mean the person is healthy.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve(
).parent.parent.parent / "configs" / "rules.yaml"

REQUIRED_FIELDS = (
    "id",
    "indicator",
    "comparison",
    "value",
    "unit",
    "source",
    "status",
)

ALLOWED_STATUSES = {"plausibility_check", "illustrative_not_clinical"}
ALLOWED_COMPARISONS = {"lt", "gt", "lte", "gte", "outside_range"}

# Shown on every result, whatever the outcome.
DISCLAIMER = "configured condition, not a diagnosis; no flag does not mean healthy"


class RuleError(ValueError):
    pass


@dataclass
class RuleResult:
    rule_id: str
    indicator: str
    rule_status: str
    status: str  # "flag", "no_flag", "not_evaluated"
    reason: str
    source: str
    value: float | None
    unit: str


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_rule(rule):
    """Raise RuleError if the rule is malformed or uses a forbidden status."""
    if not isinstance(rule, dict):
        raise RuleError(f"rule must be a mapping, got {type(rule).__name__}")
    missing = [field for field in REQUIRED_FIELDS if field not in rule]
    if missing:
        raise RuleError(
            f"rule {rule.get('id', '<no id>')} is missing field(s): {', '.join(missing)}"
        )
    for field in ("id", "indicator", "unit", "source", "status"):
        if not isinstance(rule[field], str) or not str(rule[field]).strip():
            raise RuleError(
                f"rule {rule.get('id', '<no id>')} has an invalid {field}")

    status = str(rule["status"])
    if status not in ALLOWED_STATUSES:
        raise RuleError(
            f"rule '{rule['id']}' has status '{status}'. Allowed statuses are: "
            f"{', '.join(sorted(ALLOWED_STATUSES))}. A clinical rule is not supported."
        )

    comparison = str(rule["comparison"])
    if comparison not in ALLOWED_COMPARISONS:
        raise RuleError(
            f"rule '{rule['id']}' has comparison '{comparison}'. Allowed comparisons are: "
            f"{', '.join(sorted(ALLOWED_COMPARISONS))}."
        )

    value = rule["value"]
    if comparison == "outside_range":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise RuleError(
                f"rule '{rule['id']}' uses outside_range and needs a two-item [low, high] value"
            )
        if not all(_is_number(v) for v in value):
            raise RuleError(f"rule '{rule['id']}' has a non-numeric range")
        if value[0] >= value[1]:
            raise RuleError(f"rule '{rule['id']}' has an empty range {value}")
    elif not _is_number(value):
        raise RuleError(
            f"rule '{rule['id']}' has a non-numeric value {value!r}")

    if "message" in rule and not isinstance(rule["message"], str):
        raise RuleError(f"rule '{rule['id']}' has a non-text message")
    return rule


def load_rules(path=None):
    """Load and validate every rule in rules.yaml."""
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleError(f"cannot read rules file {target}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuleError(
            f"rules file {target} is not valid YAML: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("rules")
    if not isinstance(data, list) or not data:
        raise RuleError(
            f"rules file {target} must contain a non-empty 'rules' list")
    for rule in data:
        validate_rule(rule)
    return data


def _missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _flag(comparison, value, threshold):
    if comparison == "lt":
        return value < threshold
    if comparison == "gt":
        return value > threshold
    if comparison == "lte":
        return value <= threshold
    if comparison == "gte":
        return value >= threshold
    low, high = threshold
    return value < low or value > high


def _describe(comparison, threshold):
    if comparison == "outside_range":
        return f"outside {threshold[0]} to {threshold[1]}"
    words = {"lt": "below", "gt": "above",
             "lte": "at or below", "gte": "at or above"}
    return f"{words[comparison]} {threshold}"


def evaluate(rule, indicator_value, unit):
    """Evaluate one rule against one indicator value.

    Returns "flag", "no_flag" or "not_evaluated". A missing value, a value that
    is not a number, or a unit mismatch is always "not_evaluated": missing data
    is never treated as fine.
    """
    validate_rule(rule)
    message = str(rule.get("message", f"rule '{rule['id']}'")).strip()
    base = f"{message} | {DISCLAIMER}"

    if _missing(indicator_value):
        return RuleResult(
            rule_id=rule["id"],
            indicator=rule["indicator"],
            rule_status=rule["status"],
            status="not_evaluated",
            reason=f"rule '{rule['id']}': no value available for {rule['indicator']} | {DISCLAIMER}",
            source=rule["source"],
            value=None,
            unit=str(unit) if unit is not None else "",
        )

    try:
        numeric = float(indicator_value)
    except (TypeError, ValueError):
        return RuleResult(
            rule_id=rule["id"],
            indicator=rule["indicator"],
            rule_status=rule["status"],
            status="not_evaluated",
            reason=(
                f"rule '{rule['id']}': the value for {rule['indicator']} is not a number "
                f"({indicator_value!r}) | {DISCLAIMER}"
            ),
            source=rule["source"],
            value=None,
            unit=str(unit) if unit is not None else "",
        )
    if math.isnan(numeric):
        return RuleResult(
            rule_id=rule["id"],
            indicator=rule["indicator"],
            rule_status=rule["status"],
            status="not_evaluated",
            reason=f"rule '{rule['id']}': the value for {rule['indicator']} is NaN | {DISCLAIMER}",
            source=rule["source"],
            value=None,
            unit=str(unit) if unit is not None else "",
        )

    if str(unit) != str(rule["unit"]):
        return RuleResult(
            rule_id=rule["id"],
            indicator=rule["indicator"],
            rule_status=rule["status"],
            status="not_evaluated",
            reason=(
                f"rule '{rule['id']}': unit '{unit}' does not match the rule unit "
                f"'{rule['unit']}' | {DISCLAIMER}"
            ),
            source=rule["source"],
            value=numeric,
            unit=str(unit) if unit is not None else "",
        )

    hit = _flag(rule["comparison"], numeric, rule["value"])
    detail = (
        f"{rule['indicator']} {numeric:g} {rule['unit']} is "
        f"{_describe(rule['comparison'], rule['value'])}"
    )
    outcome = "flag" if hit else "no_flag"
    return RuleResult(
        rule_id=rule["id"],
        indicator=rule["indicator"],
        rule_status=rule["status"],
        status=outcome,
        reason=f"{base} ({detail})",
        source=rule["source"],
        value=numeric,
        unit=str(unit),
    )
