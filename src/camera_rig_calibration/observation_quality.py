from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config.models import MarkerSettings, ObservationQualitySettings


FILTER_VERSION = "observation_quality_v1"
REQUIRED_COLUMNS = {
    "observer_type",
    "marker_id",
    "marker_length_m",
    "fx",
    "fy",
    "cx",
    "cy",
    "distortion_model",
    "pnp_success",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
    "area_px2",
    *(f"d{index}" for index in range(8)),
    *(f"corner{index}_{axis}" for index in range(4) for axis in ("u", "v")),
}
DECISION_COLUMNS = [
    "job_id",
    "decision",
    "reason",
    "threshold",
    "measured_value",
]


class ObservationQualityError(RuntimeError):
    """Raised when raw observations cannot be filtered reproducibly."""


@dataclass(frozen=True)
class ObservationFilterResult:
    job_id: str
    output_directory: Path
    accepted_path: Path
    rejected_path: Path
    filtered_observations_root: Path
    accepted_count: int
    rejected_count: int
    summary: dict[str, Any]


def _truthy(value: object, *, missing_default: bool = False) -> bool:
    if value in {None, ""}:
        return missing_default
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_values(row: dict[str, str], names: Iterable[str]) -> list[float] | None:
    values: list[float] = []
    for name in names:
        try:
            value = float(row.get(name, ""))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def _object_points(marker_length_m: float) -> np.ndarray:
    half = marker_length_m / 2.0
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _reprojection_rmse(row: dict[str, str]) -> float:
    intrinsic_values = _finite_values(row, ("fx", "fy", "cx", "cy"))
    pose_values = _finite_values(
        row,
        (
            "rvec_x",
            "rvec_y",
            "rvec_z",
            "tvec_x_m",
            "tvec_y_m",
            "tvec_z_m",
        ),
    )
    corner_values = _finite_values(
        row,
        (
            "corner0_u",
            "corner0_v",
            "corner1_u",
            "corner1_v",
            "corner2_u",
            "corner2_v",
            "corner3_u",
            "corner3_v",
        ),
    )
    marker_values = _finite_values(row, ("marker_length_m",))
    if (
        intrinsic_values is None
        or pose_values is None
        or corner_values is None
        or marker_values is None
        or marker_values[0] <= 0
    ):
        raise ObservationQualityError(
            "PnP reprojection RMSE cannot be reconstructed because an "
            "intrinsic, pose, corner, or marker-length value is missing/non-finite"
        )

    fx, fy, cx, cy = intrinsic_values
    camera = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [
            float(row.get(f"d{index}") or 0.0)
            for index in range(8)
        ],
        dtype=np.float64,
    )
    rvec = np.asarray(pose_values[:3], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose_values[3:], dtype=np.float64).reshape(3, 1)
    corners = np.asarray(corner_values, dtype=np.float64).reshape(4, 2)
    objects = _object_points(marker_values[0])
    model = str(row.get("distortion_model", "plumb_bob")).strip().lower()
    if model in {"equidistant", "fisheye"}:
        projected, _ = cv2.fisheye.projectPoints(
            objects.reshape(-1, 1, 3),
            rvec,
            tvec,
            camera,
            distortion[:4].reshape(4, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            objects, rvec, tvec, camera, distortion, None
        )
    residuals = projected.reshape(4, 2) - corners
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def observation_succeeded(row: dict[str, str]) -> bool:
    """Public fixed-validity predicate shared by method initializers."""
    return _truthy(row.get("pnp_success"))


def pnp_reprojection_rmse(row: dict[str, str]) -> float:
    """Return the same four-corner PnP RMSE used by the quality filter."""
    try:
        return _reprojection_rmse(row)
    except (ObservationQualityError, TypeError, ValueError, cv2.error):
        return float("inf")


def _identity(row: dict[str, str]) -> dict[str, Any]:
    return {
        "observer_type": str(row.get("observer_type", "")),
        "observer_id": str(
            row.get("observer_id") or row.get("camera_name") or ""
        ),
        "frame_id": str(row.get("frame_id", "")),
        "marker_id": int(float(row["marker_id"])),
    }


def _decision(
    row: dict[str, str],
    *,
    marker_settings: MarkerSettings,
    quality: ObservationQualitySettings,
) -> tuple[bool, str, object, object]:
    if not _truthy(row.get("detection_success"), missing_default=True):
        return False, "aruco_detection_failed", True, row.get("detection_success")

    corners = _finite_values(
        row,
        (
            "corner0_u",
            "corner0_v",
            "corner1_u",
            "corner1_v",
            "corner2_u",
            "corner2_v",
            "corner3_u",
            "corner3_v",
        ),
    )
    if corners is None:
        return False, "corners_missing_or_non_finite", "four finite corners", ""
    if not _truthy(row.get("pnp_success")):
        return False, "pnp_failed", True, row.get("pnp_success")

    pose = _finite_values(
        row,
        (
            "rvec_x",
            "rvec_y",
            "rvec_z",
            "tvec_x_m",
            "tvec_y_m",
            "tvec_z_m",
        ),
    )
    if pose is None:
        return False, "pnp_pose_non_finite", "finite rotation/translation", ""
    tvec = np.asarray(pose[3:], dtype=np.float64)
    if tvec[2] <= 0:
        return False, "marker_depth_not_positive", "> 0 m", float(tvec[2])
    distance = float(np.linalg.norm(tvec))
    if not math.isfinite(distance) or distance <= 0:
        return False, "marker_distance_not_positive_finite", "> 0 m", distance

    marker_id = int(float(row["marker_id"]))
    if (
        marker_settings.accepted_ids != "all_detected"
        and marker_id not in marker_settings.accepted_ids
    ):
        return (
            False,
            "marker_id_not_accepted",
            ",".join(str(value) for value in marker_settings.accepted_ids),
            marker_id,
        )

    area = float(row["area_px2"])
    if not math.isfinite(area) or area < quality.minimum_marker_area_px2:
        return (
            False,
            "marker_area_below_minimum",
            quality.minimum_marker_area_px2,
            area,
        )

    reprojection = float(row["pnp_reprojection_rmse_px"])
    reprojection_limit = quality.maximum_pnp_reprojection_error_px
    if (
        reprojection_limit != "disabled"
        and reprojection > float(reprojection_limit)
    ):
        return (
            False,
            "pnp_reprojection_error_above_maximum",
            reprojection_limit,
            reprojection,
        )

    distance_limit = quality.maximum_marker_distance_m
    if distance_limit != "disabled" and distance > float(distance_limit):
        return (
            False,
            "marker_distance_above_maximum",
            distance_limit,
            distance,
        )
    return True, "accepted", "", ""


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def filter_observations(
    raw_observations_csv: Path,
    output_directory: Path,
    *,
    job_id: str,
    marker_settings: MarkerSettings,
    quality: ObservationQualitySettings,
) -> ObservationFilterResult:
    """Apply the immutable checks and job-specific v1 quality thresholds."""
    source = raw_observations_csv.resolve()
    if not source.is_file():
        raise ObservationQualityError(f"Raw observation table is missing: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = sorted(REQUIRED_COLUMNS - set(source_fields))
    if missing:
        raise ObservationQualityError(
            "Raw observation table cannot satisfy observation_quality_v1; "
            f"missing columns: {', '.join(missing)}"
        )
    if not rows:
        raise ObservationQualityError(f"Raw observation table is empty: {source}")

    fields = list(source_fields)
    if "detection_success" not in fields:
        fields.append("detection_success")
    if "pnp_reprojection_rmse_px" not in fields:
        fields.append("pnp_reprojection_rmse_px")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for index, row in enumerate(rows, 2):
        row.setdefault("detection_success", "True")
        if _truthy(row.get("pnp_success")):
            try:
                row["pnp_reprojection_rmse_px"] = _reprojection_rmse(row)
            except ObservationQualityError as exc:
                raise ObservationQualityError(
                    f"{source}:{index}: {exc}"
                ) from exc
        else:
            row["pnp_reprojection_rmse_px"] = ""
        try:
            is_accepted, reason, threshold, measured = _decision(
                row, marker_settings=marker_settings, quality=quality
            )
            identity = _identity(row)
        except (TypeError, ValueError, KeyError) as exc:
            raise ObservationQualityError(
                f"{source}:{index}: invalid observation value: {exc}"
            ) from exc
        decision = {
            **identity,
            "job_id": job_id,
            "decision": "accepted" if is_accepted else "rejected",
            "reason": reason,
            "threshold": threshold,
            "measured_value": measured,
        }
        decisions.append(decision)
        reasons[reason] += 1
        enriched = {**row, **decision}
        (accepted if is_accepted else rejected).append(enriched)

    destination = output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    decision_fields = list(
        dict.fromkeys(fields + list(_identity(rows[0])) + DECISION_COLUMNS)
    )
    accepted_path = destination / "accepted_observations.csv"
    rejected_path = destination / "rejected_observations.csv"
    _write_csv(accepted_path, accepted, decision_fields)
    _write_csv(rejected_path, rejected, decision_fields)

    filtered_root = destination / "observations"
    _write_csv(
        filtered_root / "shared_all_aruco_observations.csv",
        accepted,
        decision_fields,
    )
    _write_csv(
        filtered_root / "shared_static_aruco_observations.csv",
        [row for row in accepted if row["observer_type"] == "static"],
        decision_fields,
    )
    _write_csv(
        filtered_root / "shared_moving_aruco_observations.csv",
        [row for row in accepted if row["observer_type"] == "moving"],
        decision_fields,
    )

    accepted_markers = sorted(
        {int(float(row["marker_id"])) for row in accepted}
    )
    summary = {
        "schema_version": 5,
        "filter": FILTER_VERSION,
        "job_id": job_id,
        "source": str(source),
        "status": "READY" if accepted else "FAILED_PREFLIGHT",
        "settings": quality.model_dump(mode="json"),
        "marker_input": marker_settings.model_dump(mode="json"),
        "total_observations": len(rows),
        "accepted_observations": len(accepted),
        "rejected_observations": len(rejected),
        "accepted_marker_ids": accepted_markers,
        "accepted_static_observations": sum(
            row["observer_type"] == "static" for row in accepted
        ),
        "accepted_moving_observations": sum(
            row["observer_type"] == "moving" for row in accepted
        ),
        "decision_counts": dict(sorted(reasons.items())),
        "decisions": decisions,
    }
    summary_path = destination / "observation_filter_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return ObservationFilterResult(
        job_id=job_id,
        output_directory=destination,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        filtered_observations_root=filtered_root,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        summary=summary,
    )
