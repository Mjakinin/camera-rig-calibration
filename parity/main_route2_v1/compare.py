"""Ordered, duplicate-preserving record comparison for pre-solver evidence."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _canonical_record(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str)


def _duplicates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(_canonical_record(row) for row in rows)
    return [
        {"record": json.loads(record), "count": count}
        for record, count in sorted(counts.items())
        if count > 1
    ]


def _as_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _field_order(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*left.keys(), *right.keys())))


def compare_ordered_rows(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    float_fields: Iterable[str] = (),
    float_tolerance: float = 0.0,
    continue_after_mismatch: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare exact row order without collapsing duplicate records.

    Numeric tolerance is opt-in by field name so identifiers such as ``01``
    and ``1`` are never silently considered equal.
    """

    if float_tolerance < 0:
        raise ValueError("float_tolerance must be non-negative")
    numeric = set(float_fields)
    differences: list[dict[str, Any]] = []
    maximum = max(len(left_rows), len(right_rows))
    for index in range(maximum):
        if index >= len(left_rows) or index >= len(right_rows):
            differences.append(
                {
                    "row_index": index,
                    "field": "__row__",
                    "main_value": (
                        None if index >= len(left_rows) else dict(left_rows[index])
                    ),
                    "wizard_value": (
                        None if index >= len(right_rows) else dict(right_rows[index])
                    ),
                    "reason": "missing_row",
                }
            )
            if not continue_after_mismatch:
                break
            continue
        left = left_rows[index]
        right = right_rows[index]
        for field in _field_order(left, right):
            if field not in left or field not in right:
                differences.append(
                    {
                        "row_index": index,
                        "field": field,
                        "main_value": left.get(field),
                        "wizard_value": right.get(field),
                        "reason": "missing_field",
                    }
                )
            else:
                left_value = left[field]
                right_value = right[field]
                equal = left_value == right_value
                delta: float | None = None
                if not equal and field in numeric:
                    left_float = _as_finite_float(left_value)
                    right_float = _as_finite_float(right_value)
                    if left_float is not None and right_float is not None:
                        delta = abs(left_float - right_float)
                        equal = delta <= float_tolerance
                if not equal:
                    difference = {
                        "row_index": index,
                        "field": field,
                        "main_value": left_value,
                        "wizard_value": right_value,
                        "reason": "value_mismatch",
                    }
                    if delta is not None:
                        difference["absolute_delta"] = delta
                    differences.append(difference)
            if differences and not continue_after_mismatch:
                break
        if differences and not continue_after_mismatch:
            break

    report = {
        "schema_version": 1,
        "status": "equal" if not differences else "mismatch",
        "main_row_count": len(left_rows),
        "wizard_row_count": len(right_rows),
        "ordering_compared": True,
        "duplicates_preserved": True,
        "main_duplicate_records": _duplicates(left_rows),
        "wizard_duplicate_records": _duplicates(right_rows),
        "float_fields": sorted(numeric),
        "float_tolerance": float_tolerance,
        "continue_after_mismatch": continue_after_mismatch,
        "stopped_at_first_mismatch": bool(
            differences and not continue_after_mismatch
        ),
        "mismatch_count_reported": len(differences),
        "first_mismatch": differences[0] if differences else None,
    }
    return report, differences


def compare_ordered_values(
    left: Sequence[Any], right: Sequence[Any]
) -> dict[str, Any]:
    """Compare ordered parameter names or other scalar sequences."""

    maximum = max(len(left), len(right))
    first = next(
        (
            {
                "index": index,
                "main_value": left[index] if index < len(left) else None,
                "wizard_value": right[index] if index < len(right) else None,
            }
            for index in range(maximum)
            if index >= len(left)
            or index >= len(right)
            or left[index] != right[index]
        ),
        None,
    )
    return {
        "schema_version": 1,
        "status": "equal" if first is None else "mismatch",
        "main_count": len(left),
        "wizard_count": len(right),
        "ordering_compared": True,
        "first_mismatch": first,
    }

