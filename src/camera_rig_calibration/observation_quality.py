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


FILTER_VERSION = "observation_quality_v2"
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
    "selection_score",
    "score_area",
    "score_reprojection",
    "score_border",
    "score_distance",
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


def _resolve_dimensions(
    row: dict[str, Any],
    *,
    source: Path,
    line_number: int,
) -> tuple[int, int]:
    try:
        width = int(float(row.get("image_width_px", "")))
        height = int(float(row.get("image_height_px", "")))
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height

    image_value = str(row.get("image_path", "")).strip()
    candidates: list[Path] = []
    if image_value:
        image_path = Path(image_value)
        candidates.append(image_path)
        if not image_path.is_absolute():
            candidates.append(source.parent / image_path)
    for candidate in candidates:
        image = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            height, width = image.shape[:2]
            return int(width), int(height)
    raise ObservationQualityError(
        f"{source}:{line_number}: exact image dimensions are missing and "
        f"the referenced image cannot be read: {image_value or '<missing>'}"
    )


def observation_selection_score(row: dict[str, Any]) -> dict[str, float]:
    """Return the deterministic, auditable ranking used by every selector."""

    width = float(row["image_width_px"])
    height = float(row["image_height_px"])
    ratio = max(0.0, float(row["marker_area_ratio"]))
    reprojection = max(0.0, float(row["pnp_reprojection_rmse_px"]))
    distance = max(0.0, float(row["distance_m"]))
    center_u = float(row["center_u"])
    center_v = float(row["center_v"])
    edge_distance = min(
        center_u,
        center_v,
        width - center_u,
        height - center_v,
    )
    border = max(0.0, min(1.0, edge_distance / (0.5 * min(width, height))))
    area = min(1.0, math.sqrt(ratio / 0.01))
    reprojection_score = 1.0 / (1.0 + reprojection)
    distance_score = 1.0 / (1.0 + distance)
    total = (
        0.40 * area
        + 0.30 * reprojection_score
        + 0.20 * border
        + 0.10 * distance_score
    )
    return {
        "selection_score": total,
        "score_area": area,
        "score_reprojection": reprojection_score,
        "score_border": border,
        "score_distance": distance_score,
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
    if quality.require_positive_depth and tvec[2] <= 0:
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

    area_ratio = float(row["marker_area_ratio"])
    if (
        not math.isfinite(area_ratio)
        or area_ratio < quality.minimum_marker_area_ratio
    ):
        return (
            False,
            "marker_area_ratio_below_minimum",
            quality.minimum_marker_area_ratio,
            area_ratio,
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
    """Apply immutable checks and the job-specific v2 quality thresholds."""
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
            "Raw observation table cannot satisfy observation_quality_v2; "
            f"missing columns: {', '.join(missing)}"
        )
    if not rows:
        raise ObservationQualityError(f"Raw observation table is empty: {source}")

    fields = list(source_fields)
    if "detection_success" not in fields:
        fields.append("detection_success")
    if "pnp_reprojection_rmse_px" not in fields:
        fields.append("pnp_reprojection_rmse_px")
    for field_name in (
        "image_width_px",
        "image_height_px",
        "marker_area_ratio",
        "selection_score",
        "score_area",
        "score_reprojection",
        "score_border",
        "score_distance",
    ):
        if field_name not in fields:
            fields.append(field_name)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for index, row in enumerate(rows, 2):
        width, height = _resolve_dimensions(
            row, source=source, line_number=index
        )
        row["image_width_px"] = width
        row["image_height_px"] = height
        try:
            area_px2 = float(row["area_px2"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ObservationQualityError(
                f"{source}:{index}: invalid marker area: {exc}"
            ) from exc
        if not math.isfinite(area_px2) or area_px2 < 0:
            raise ObservationQualityError(
                f"{source}:{index}: marker area must be finite and non-negative"
            )
        row["marker_area_ratio"] = area_px2 / float(width * height)
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
        if is_accepted:
            score = observation_selection_score(row)
            row.update(score)
            decision.update(score)
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
    inventory: list[dict[str, Any]] = []
    for marker_id in sorted({int(float(row["marker_id"])) for row in rows}):
        raw_marker = [
            row for row in rows if int(float(row["marker_id"])) == marker_id
        ]
        accepted_marker = [
            row
            for row in accepted
            if int(float(row["marker_id"])) == marker_id
        ]
        rejected_marker = [
            row
            for row in rejected
            if int(float(row["marker_id"])) == marker_id
        ]
        observer_ids = sorted(
            {
                str(row.get("observer_id") or row.get("camera_name") or "")
                for row in accepted_marker
            }
        )
        frames = sorted(
            {
                str(row.get("frame_id", ""))
                for row in accepted_marker
                if str(row.get("frame_id", ""))
            }
        )
        ratios = sorted(float(row["marker_area_ratio"]) for row in accepted_marker)
        reprojections = sorted(
            float(row["pnp_reprojection_rmse_px"])
            for row in accepted_marker
        )
        whitelisted = (
            marker_settings.accepted_ids == "all_detected"
            or marker_id in marker_settings.accepted_ids
        )
        suspicious_reasons: list[str] = []
        if len(accepted_marker) < 2:
            suspicious_reasons.append("fewer_than_two_accepted_observations")
        if len(observer_ids) < 2 and len(frames) < 2:
            suspicious_reasons.append("no_repeated_observer_or_frame_support")
        inventory.append(
            {
                "marker_id": marker_id,
                "whitelisted": whitelisted,
                "raw_observations": len(raw_marker),
                "accepted_observations": len(accepted_marker),
                "rejected_observations": len(rejected_marker),
                "observer_ids": observer_ids,
                "frame_ids": frames,
                "minimum_area_ratio": ratios[0] if ratios else None,
                "median_area_ratio": (
                    float(np.median(ratios)) if ratios else None
                ),
                "maximum_area_ratio": ratios[-1] if ratios else None,
                "minimum_pnp_rmse_px": (
                    reprojections[0] if reprojections else None
                ),
                "median_pnp_rmse_px": (
                    float(np.median(reprojections))
                    if reprojections
                    else None
                ),
                "maximum_pnp_rmse_px": (
                    reprojections[-1] if reprojections else None
                ),
                "reject_reasons": dict(
                    sorted(
                        Counter(
                            str(row.get("reason", "unknown"))
                            for row in rejected_marker
                        ).items()
                    )
                ),
                "suspicious": bool(suspicious_reasons),
                "suspicious_reasons": suspicious_reasons,
                "automatic_candidate": (
                    whitelisted
                    and len(accepted_marker) >= 2
                    and (len(observer_ids) >= 2 or len(frames) >= 2)
                ),
            }
        )
    inventory_json = destination / "marker_inventory.json"
    inventory_json.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory_csv_rows = [
        {
            **item,
            "observer_ids": ",".join(item["observer_ids"]),
            "frame_ids": ",".join(item["frame_ids"]),
            "reject_reasons": json.dumps(
                item["reject_reasons"], sort_keys=True
            ),
            "suspicious_reasons": ",".join(item["suspicious_reasons"]),
        }
        for item in inventory
    ]
    _write_csv(
        destination / "marker_inventory.csv",
        inventory_csv_rows,
        list(inventory_csv_rows[0]) if inventory_csv_rows else [
            "marker_id",
            "whitelisted",
            "raw_observations",
            "accepted_observations",
        ],
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
        "marker_inventory": inventory,
        "marker_inventory_json": str(inventory_json),
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
