"""Focused scientific reporting responsibility."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..anchor_export import ensure_experiment_anchor_exports
from ..anchor_export.geometry import rotation_to_quaternion
from ..visualization.scene import ensure_visualization_artifacts
from .ap03_derived import ensure_ap03_derived_results
from .simulation_ground_truth import (
    ensure_simulation_ground_truth,
    resolve_simulation_ground_truth,
)

from ..methods.common.geometry import (
    R_to_rpy_deg,
    R_to_rvec,
    invT,
    make_T,
    rot_error_deg,
    rpy_to_R,
    rvec_to_R,
)

from .reporting_bindings import current_reporting_bindings

@dataclass(frozen=True)
class PoseRecord:
    entity_id: str
    transform: np.ndarray
    source: str
    reference_frame: str
    transform_convention: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    from io import StringIO

    buffer = StringIO(newline="")
    with buffer:
        writer = csv.DictWriter(
            buffer,
            fieldnames=fields or ["status"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        text = buffer.getvalue()
    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return default


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _mean(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.mean(clean)) if clean else None


def _median(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.median(clean)) if clean else None


def _maximum(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return max(clean) if clean else None


def _text_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(
            [
                len(str(header)),
                *(
                    len(row[index])
                    for row in rendered
                    if index < len(row)
                ),
            ]
        )
        for index, header in enumerate(headers)
    ]
    output = [
        " | ".join(
            str(header).ljust(widths[index])
            for index, header in enumerate(headers)
        ).rstrip(),
        "-+-".join("-" * width for width in widths),
    ]
    output.extend(
        " | ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        ).rstrip()
        for row in rendered
    )
    return "\n".join(output)


def _pose_from_row(row: dict[str, Any]) -> np.ndarray:
    if all(str(row.get(key, "")).strip() for key in ("rvec_x", "rvec_y", "rvec_z")):
        rotation = rvec_to_R(
            np.asarray(
                [
                    _float(row, "rvec_x"),
                    _float(row, "rvec_y"),
                    _float(row, "rvec_z"),
                ],
                dtype=np.float64,
            )
        )
    else:
        rotation = rpy_to_R(
            math.radians(_float(row, "roll_deg")),
            math.radians(_float(row, "pitch_deg")),
            math.radians(_float(row, "yaw_deg")),
        )
    return make_T(
        rotation,
        [_float(row, "x_m"), _float(row, "y_m"), _float(row, "z_m")],
    )


def load_pose_records(path: Path) -> dict[str, PoseRecord]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records: dict[str, PoseRecord] = {}
    for row in rows:
        entity_id = str(
            row.get("entity_id")
            or row.get("static_camera")
            or row.get("camera")
            or ""
        ).strip()
        if not entity_id:
            continue
        records[entity_id] = PoseRecord(
            entity_id=entity_id,
            transform=_pose_from_row(row),
            source=str(row.get("source", "")),
            reference_frame=str(row.get("reference_frame", "")),
            transform_convention=str(row.get("transform_convention", "")),
        )
    return records


def _pose_columns(prefix: str, transform: np.ndarray) -> dict[str, float]:
    rpy = R_to_rpy_deg(transform[:3, :3])
    rvec = R_to_rvec(transform[:3, :3])
    return {
        f"{prefix}x_m": float(transform[0, 3]),
        f"{prefix}y_m": float(transform[1, 3]),
        f"{prefix}z_m": float(transform[2, 3]),
        f"{prefix}roll_deg": float(rpy[0]),
        f"{prefix}pitch_deg": float(rpy[1]),
        f"{prefix}yaw_deg": float(rpy[2]),
        f"{prefix}rvec_x": float(rvec[0]),
        f"{prefix}rvec_y": float(rvec[1]),
        f"{prefix}rvec_z": float(rvec[2]),
    }


def _direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.full(3, np.nan)


def _angle_between(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-12 or second_norm < 1e-12:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def pairwise_rows(
    records: dict[str, PoseRecord],
    *,
    method: str,
    label: str,
) -> list[dict[str, Any]]:
    """Return gauge-invariant T_from_camera_to_camera relations."""
    rows: list[dict[str, Any]] = []
    for first, second in combinations(sorted(records), 2):
        transform = invT(records[first].transform) @ records[second].transform
        translation = transform[:3, 3]
        direction = _direction(translation)
        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "from_camera": first,
            "to_camera": second,
            "pair": f"{first}-{second}",
            "transform_convention": (
                "T_from_camera_to_camera = "
                "inv(T_reference_from_camera) @ T_reference_to_camera"
            ),
            "baseline_m": float(np.linalg.norm(translation)),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "direction_z": float(direction[2]),
        }
        row.update(_pose_columns("", transform))
        rows.append(row)
    return rows



__all__ = [
    'PoseRecord',
    '_now',
    '_read_json',
    '_write_json',
    '_write_csv',
    '_write_text',
    '_sha256',
    '_float',
    '_finite',
    '_fmt',
    '_mean',
    '_median',
    '_maximum',
    '_text_table',
    '_pose_from_row',
    'load_pose_records',
    '_pose_columns',
    '_direction',
    '_angle_between',
    'pairwise_rows',
]
