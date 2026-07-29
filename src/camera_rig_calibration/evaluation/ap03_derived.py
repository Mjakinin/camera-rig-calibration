from __future__ import annotations

import csv
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ..anchor_export.geometry import (
    invert_transform,
    pose_payload,
    rvec_to_rotation,
)
from ..methods.common.geometry import R_to_rpy_deg, R_to_rvec


DERIVED_CONTRACT = "rigcal_ap03_scale_variants_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_text_if_changed(path: Path, text: str) -> bool:
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    temporary.replace(path)
    return True


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    return _write_text_if_changed(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _write_csv_if_changed(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> bool:
    fieldnames = fields or list(
        dict.fromkeys(key for row in rows for key in row)
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames or ["status"])
    writer.writeheader()
    writer.writerows(rows)
    return _write_text_if_changed(path, stream.getvalue())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(paths: list[Path], extra: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(extra, sort_keys=True, separators=(",", ":")).encode()
    )
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(_sha256(path).encode() if path.is_file() else b"missing")
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _pose(row: dict[str, str]) -> np.ndarray:
    rotation = rvec_to_rotation(
        (
            float(row["rvec_x"]),
            float(row["rvec_y"]),
            float(row["rvec_z"]),
        )
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [
        float(row["x_m"]),
        float(row["y_m"]),
        float(row["z_m"]),
    ]
    return transform


def _canonical_camera_rows(
    source: Path,
    *,
    mode: str,
    best_model: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    poses: dict[str, np.ndarray] = {}
    for source_row in _rows(source):
        camera = str(source_row.get("entity_id", "")).strip()
        if not camera:
            continue
        transform = _pose(source_row)
        poses[camera] = transform
        rpy = R_to_rpy_deg(transform[:3, :3])
        rvec = R_to_rvec(transform[:3, :3])
        rows.append(
            {
                "entity_type": "static_camera",
                "entity_id": camera,
                "source": f"ap03_{mode}_scale_from_shared_colmap",
                "reference_frame": f"COLMAP best_model {best_model}",
                "transform_convention": (
                    "T_reference_camera (camera pose expressed in reference frame)"
                ),
                "x_m": float(transform[0, 3]),
                "y_m": float(transform[1, 3]),
                "z_m": float(transform[2, 3]),
                "roll_deg": float(rpy[0]),
                "pitch_deg": float(rpy[1]),
                "yaw_deg": float(rpy[2]),
                "rvec_x": float(rvec[0]),
                "rvec_y": float(rvec[1]),
                "rvec_z": float(rvec[2]),
            }
        )
    return rows, poses


def _pairwise_rows(
    poses: dict[str, np.ndarray],
    *,
    method: str,
    label: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for first, second in combinations(sorted(poses), 2):
        transform = invert_transform(poses[first]) @ poses[second]
        translation = transform[:3, 3]
        norm = float(np.linalg.norm(translation))
        details = pose_payload(transform)
        output.append(
            {
                "method": method,
                "label": label,
                "from_camera": first,
                "to_camera": second,
                "pair": f"{first}-{second}",
                "transform_convention": (
                    "T_from_camera_to_camera = "
                    "inv(T_reference_from_camera) @ T_reference_to_camera"
                ),
                "baseline_m": norm,
                "direction_x": (
                    float(translation[0] / norm) if norm > 1e-12 else None
                ),
                "direction_y": (
                    float(translation[1] / norm) if norm > 1e-12 else None
                ),
                "direction_z": (
                    float(translation[2] / norm) if norm > 1e-12 else None
                ),
                **details,
            }
        )
    return output


def _selection_anchor(experiment_root: Path) -> int | None:
    selection = _read_json(
        experiment_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    value = selection.get("evaluation_anchor", {}).get("selected")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _shared_anchor_geometry(
    container: Path,
    *,
    anchor_marker_id: int,
) -> tuple[dict[str, Any], Path] | tuple[None, Path]:
    source = (
        container
        / "diagnostics"
        / "method"
        / "scale_multi"
        / "AP03_MARKER_SIZE_SCALE_ONLY_TRIANGULATED_CORNERS.csv"
    )
    selected = []
    for row in _rows(source):
        try:
            marker = int(float(row["marker_id"]))
            corner = int(float(row["corner_idx"]))
            point = [
                float(row["x_colmap"]),
                float(row["y_colmap"]),
                float(row["z_colmap"]),
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            marker == anchor_marker_id
            and corner in {0, 1, 2, 3}
            and row.get("status", "OK") == "OK"
            and np.all(np.isfinite(point))
        ):
            selected.append(
                {
                    "marker_id": marker,
                    "corner_idx": corner,
                    "x_colmap": point[0],
                    "y_colmap": point[1],
                    "z_colmap": point[2],
                    "observation_count": row.get("obs_count"),
                    "inlier_count": row.get("inlier_count"),
                    "median_reprojection_px": row.get("median_reproj_px"),
                }
            )
    selected.sort(key=lambda item: item["corner_idx"])
    output = (
        container
        / "diagnostics"
        / "derived"
        / "shared_anchor_geometry"
        / f"marker_{anchor_marker_id}_corners_colmap.json"
    )
    if [item["corner_idx"] for item in selected] != [0, 1, 2, 3]:
        return None, output
    payload = {
        "schema_version": 1,
        "contract": DERIVED_CONTRACT,
        "anchor_marker_id": anchor_marker_id,
        "coordinate_frame": "shared COLMAP best-model units; no metric scale",
        "source": source.relative_to(container).as_posix(),
        "source_sha256": _sha256(source),
        "corners": selected,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    _write_json_if_changed(output, payload)
    _write_csv_if_changed(
        output.with_suffix(".csv"),
        selected,
    )
    return payload, output


def _copy_if_changed(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    _write_text_if_changed(
        destination, source.read_text(encoding="utf-8")
    )


def ensure_ap03_derived_results(
    experiment_root: Path,
) -> dict[str, dict[str, Any]]:
    """Publish scale-specific AP03 results from one completed COLMAP container."""
    experiment_root = experiment_root.resolve()
    anchor = _selection_anchor(experiment_root)
    outcomes: dict[str, dict[str, Any]] = {}
    if anchor is None:
        return outcomes
    for result_path in sorted(
        (experiment_root / "methods" / "ap03").glob("*/RESULT.json")
    ):
        container = result_path.parent
        base_result = _read_json(result_path)
        label = str(base_result.get("label") or container.name)
        shared_geometry, shared_path = _shared_anchor_geometry(
            container, anchor_marker_id=anchor
        )
        if shared_geometry is None:
            outcomes[label] = {
                "status": "unavailable",
                "reason": "shared evaluation-anchor corners are unavailable",
            }
            continue
        mode_details: dict[str, tuple[Path, Path]] = {}
        for mode in ("single", "multi"):
            scale_root = (
                container / "diagnostics" / "method" / f"scale_{mode}"
            )
            mode_details[mode] = (
                scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json",
                scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv",
            )
        metadata_by_mode = {
            mode: _read_json(paths[0])
            for mode, paths in mode_details.items()
        }
        best_models = {
            str(value.get("best_model"))
            for value in metadata_by_mode.values()
            if value.get("best_model") is not None
        }
        if len(best_models) != 1:
            outcomes[label] = {
                "status": "unavailable",
                "reason": "AP03 Single and Multi do not identify one shared best_model",
            }
            continue
        best_model = next(iter(best_models))
        resolved_config = container / "provenance" / "resolved_config.yaml"
        for mode in ("single", "multi"):
            method = f"ap03_{mode}"
            metadata_path, camera_path = mode_details[mode]
            metadata = metadata_by_mode[mode]
            try:
                scale = float(metadata["scale_m_per_colmap_unit"])
            except (KeyError, TypeError, ValueError):
                outcomes[f"{method}/{label}"] = {
                    "status": "unavailable",
                    "reason": f"AP03 {mode} scale is unavailable",
                }
                continue
            camera_rows, poses = _canonical_camera_rows(
                camera_path, mode=mode, best_model=best_model
            )
            if not poses:
                outcomes[f"{method}/{label}"] = {
                    "status": "unavailable",
                    "reason": f"AP03 {mode} camera poses are unavailable",
                }
                continue
            derived_root = experiment_root / "methods" / method / label
            fingerprint = _fingerprint(
                [
                    result_path,
                    metadata_path,
                    camera_path,
                    shared_path,
                    resolved_config,
                ],
                {
                    "contract": DERIVED_CONTRACT,
                    "method": method,
                    "label": label,
                    "anchor_marker_id": anchor,
                    "best_model": best_model,
                    "scale_m_per_colmap_unit": scale,
                },
            )
            _write_csv_if_changed(
                derived_root / "camera_extrinsics.csv", camera_rows
            )
            _write_csv_if_changed(
                derived_root / "pairwise_camera_extrinsics.csv",
                _pairwise_rows(poses, method=method, label=label),
            )
            _copy_if_changed(
                resolved_config,
                derived_root / "provenance" / "resolved_config.yaml",
            )
            provenance = {
                "schema_version": 5,
                "layout_version": 2,
                "contract": DERIVED_CONTRACT,
                "fingerprint": fingerprint,
                "scientific_role": f"ap03_{mode}_scale_result",
                "shared_colmap_container": str(
                    container.relative_to(experiment_root).as_posix()
                ),
                "shared_colmap_best_model": best_model,
                "shared_anchor_geometry": str(
                    shared_path.relative_to(experiment_root).as_posix()
                ),
                "scale_metadata": str(
                    metadata_path.relative_to(experiment_root).as_posix()
                ),
                "camera_pose_source": str(
                    camera_path.relative_to(experiment_root).as_posix()
                ),
                "scale_m_per_colmap_unit": scale,
                "method_rerun": False,
                "colmap_rerun": False,
            }
            _write_json_if_changed(
                derived_root / "provenance" / "derived_result.json",
                provenance,
            )
            result = {
                **{
                    key: value
                    for key, value in base_result.items()
                    if key
                    not in {
                        "method",
                        "primary_result",
                        "metrics",
                        "detail_artifacts",
                        "camera_extrinsics_anchor",
                        "camera_extrinsics_anchor_json",
                        "camera_extrinsics_anchor_yaml",
                        "anchor_alignment",
                    }
                },
                "schema_version": 5,
                "layout_version": 2,
                "method": method,
                "label": label,
                "scientific_role": f"ap03_{mode}_scale_result",
                "shared_colmap_container": str(
                    container.relative_to(experiment_root).as_posix()
                ),
                "derived_fingerprint": fingerprint,
                "artifact_status": "available",
                "execution_status": "completed_shared_colmap",
                "solver_status": "not_applicable",
                "calibration_status": "available",
                "evaluation_status": "not_run",
                "anchor_export_status": "PENDING",
                "visualization_status": "not_generated",
                "primary_result": mode,
                "camera_extrinsics": "camera_extrinsics.csv",
                "pairwise_camera_extrinsics": "pairwise_camera_extrinsics.csv",
                "static_camera_count": len(poses),
                "available_static_cameras": sorted(poses),
                "config_summary": {
                    **dict(base_result.get("config_summary", {})),
                    "ap03_scale_mode": mode,
                    "scale_m_per_colmap_unit": scale,
                    "best_model": best_model,
                },
                "metrics": {
                    "ap03_scale": metadata,
                    "shared_colmap": {
                        "container": str(
                            container.relative_to(experiment_root)
                        ),
                        "best_model": best_model,
                    },
                },
                "detail_artifacts": [
                    metadata_path.relative_to(experiment_root).as_posix(),
                    shared_path.relative_to(experiment_root).as_posix(),
                ],
            }
            _write_json_if_changed(derived_root / "RESULT.json", result)
            outcomes[f"{method}/{label}"] = {
                "status": "available",
                "fingerprint": fingerprint,
                "scale_m_per_colmap_unit": scale,
                "best_model": best_model,
                "camera_count": len(poses),
            }
        if all(
            outcomes.get(f"ap03_{mode}/{label}", {}).get("status")
            == "available"
            for mode in ("single", "multi")
        ):
            base_result["scientific_role"] = (
                "shared_colmap_reconstruction_container"
            )
            base_result["comparison_visibility"] = (
                "hidden_when_scale_variants_available"
            )
            base_result["derived_results"] = [
                f"../../ap03_single/{label}",
                f"../../ap03_multi/{label}",
            ]
            _write_json_if_changed(result_path, base_result)
    return outcomes
