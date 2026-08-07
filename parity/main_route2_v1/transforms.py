"""Rigid-transform comparison without a scientific-pipeline dependency."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _matrix(value: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = [[float(item) for item in row] for row in value]
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("transform must be a 4x4 matrix")
    return matrix


def compare_transforms(
    main: Sequence[Sequence[float]],
    wizard: Sequence[Sequence[float]],
    *,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict[str, float | str | int]:
    """Compare two transforms using translation norm and SO(3) angle."""

    left = _matrix(main)
    right = _matrix(wizard)
    translation_delta = math.sqrt(
        sum((left[index][3] - right[index][3]) ** 2 for index in range(3))
    )
    # trace(R_left^T R_right)
    trace = sum(
        sum(left[k][i] * right[k][i] for k in range(3))
        for i in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    rotation_delta = math.degrees(math.acos(cosine))
    return {
        "schema_version": 1,
        "status": (
            "equal"
            if translation_delta <= translation_tolerance_m
            and rotation_delta <= rotation_tolerance_deg
            else "mismatch"
        ),
        "translation_delta_m": translation_delta,
        "rotation_delta_deg": rotation_delta,
        "translation_tolerance_m": translation_tolerance_m,
        "rotation_tolerance_deg": rotation_tolerance_deg,
    }

