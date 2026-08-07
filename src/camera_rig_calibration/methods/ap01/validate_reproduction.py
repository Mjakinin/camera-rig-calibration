"""Fail closed unless a locked AP01 historical reproduction is equivalent."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, colmap_images, parser, read_json
from .contracts import AP01MethodContract, resolve_ap01_method_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_rows(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = core.rpy_deg_to_R(
                float(row["roll_deg"]),
                float(row["pitch_deg"]),
                float(row["yaw_deg"]),
            )
            transform[:3, 3] = [
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"]),
            ]
            result[str(row["entity_id"])] = transform
    return result


def _reference_poses(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = read_json(path)
    if payload.get("ground_truth_used") is not False:
        raise RuntimeError("Locked AP01 pose reference is not declared no-GT")
    records = payload.get("camera_records")
    if not isinstance(records, list):
        raise RuntimeError("Locked AP01 pose reference has no camera records")
    poses = {
        str(record["camera_id"]): np.asarray(
            record["homogeneous_transform_4x4"], dtype=np.float64
        )
        for record in records
    }
    return poses, payload


def validate_reproduction_outputs(
    *,
    repository_root: Path,
    dataset: Path,
    output_root: Path,
    root_camera: str,
    contract: AP01MethodContract,
) -> dict[str, Any]:
    """Compare production outputs directly to contract-locked invariants."""

    if contract.reproduction_validation_policy != "locked_main_route2_v1":
        raise ValueError(
            "Unknown AP01 reproduction-validation policy: "
            f"{contract.reproduction_validation_policy}"
        )

    registered = len(core.parse_colmap_poses(colmap_images(output_root)))
    moving_files = [
        path
        for path in (dataset / "raw_images" / "moving").iterdir()
        if path.is_file()
    ]
    scale_payload = read_json(
        output_root / "02_metric_scale" / "SCALE_DIAGNOSTICS.json"
    )
    scale = float(scale_payload["scale_m_per_colmap_unit"])

    candidates = read_json(
        output_root / "03_candidates" / "transform_candidates.json"
    )
    candidate_counts: Counter[tuple[str, str]] = Counter(
        (str(row["target_camera"]), str(row["mode"]))
        for row in candidates
    )
    candidate_counts[(root_camera, "root")] += 1
    actual_candidate_counts = tuple(
        sorted(
            (camera, mode, count)
            for (camera, mode), count in candidate_counts.items()
        )
    )
    expected_candidate_counts = tuple(
        sorted(contract.reproduction_expected_candidate_counts)
    )

    solution = read_json(
        output_root / "03_static_extrinsics" / "solution_summary.json"
    )
    selections = {root_camera: "root"}
    selections.update(
        {
            str(camera): str(details.get("selected_candidate_type"))
            for camera, details in solution["per_target_diagnostics"].items()
        }
    )
    actual_selections = tuple(sorted(selections.items()))
    expected_selections = tuple(
        sorted(contract.reproduction_expected_selections)
    )

    actual_poses = _pose_rows(
        output_root
        / "03_static_extrinsics"
        / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
    )
    locked_relative = contract.reproduction_locked_final_pose_path
    locked_hash = contract.reproduction_locked_final_pose_sha256
    if locked_relative is None or locked_hash is None:
        raise RuntimeError("AP01 reproduction contract has no locked pose evidence")
    locked_path = repository_root / locked_relative
    actual_locked_hash = _sha256(locked_path)
    if actual_locked_hash != locked_hash:
        raise RuntimeError(
            "Locked AP01 pose evidence hash mismatch: "
            f"expected {locked_hash}, got {actual_locked_hash}"
        )
    reference_poses, reference_payload = _reference_poses(locked_path)
    expected_inventory = tuple(
        contract.reproduction_expected_camera_inventory
    )
    actual_inventory = tuple(
        camera for camera in expected_inventory if camera in actual_poses
    )
    inventory_exact = (
        set(actual_poses) == set(expected_inventory) == set(reference_poses)
        and actual_inventory == expected_inventory
    )

    translation_tolerance = float(
        contract.reproduction_translation_tolerance_m or 0.0
    )
    rotation_tolerance = float(
        contract.reproduction_rotation_matrix_tolerance or 0.0
    )
    per_camera: dict[str, dict[str, Any]] = {}
    for camera in sorted(set(actual_poses) | set(reference_poses)):
        actual = actual_poses.get(camera)
        reference = reference_poses.get(camera)
        if actual is None or reference is None:
            per_camera[camera] = {
                "status": "missing",
                "translation_norm_delta_m": None,
                "rotation_matrix_max_abs_delta": None,
            }
            continue
        translation_delta = float(
            np.linalg.norm(actual[:3, 3] - reference[:3, 3])
        )
        rotation_delta = float(
            np.max(np.abs(actual[:3, :3] - reference[:3, :3]))
        )
        per_camera[camera] = {
            "status": (
                "equivalent"
                if translation_delta <= translation_tolerance
                and rotation_delta <= rotation_tolerance
                else "different"
            ),
            "translation_norm_delta_m": translation_delta,
            "rotation_matrix_max_abs_delta": rotation_delta,
        }

    expected_scale = float(
        contract.reproduction_expected_scale_m_per_colmap_unit or 0.0
    )
    scale_tolerance = float(
        contract.reproduction_scale_absolute_tolerance or 0.0
    )
    checks = {
        "registered_images": (
            registered == contract.reproduction_expected_registered_images
        ),
        "total_moving_images": (
            len(moving_files)
            == contract.reproduction_expected_total_moving_images
        ),
        "metric_scale": abs(scale - expected_scale) <= scale_tolerance,
        "candidate_counts": actual_candidate_counts
        == expected_candidate_counts,
        "selections": actual_selections == expected_selections,
        "camera_inventory": inventory_exact,
        "final_poses": bool(per_camera)
        and all(row["status"] == "equivalent" for row in per_camera.values()),
        "locked_reference_no_ground_truth": (
            reference_payload.get("ground_truth_used") is False
        ),
    }
    success = all(checks.values())
    return {
        "schema_version": 1,
        "status": "END_TO_END_EQUIVALENT" if success else "DIFFERENT",
        "success": success,
        "ground_truth_used": False,
        "comparison_method": "direct_transform_comparison_without_alignment",
        "method_contract": contract.fingerprint_payload(),
        "method_contract_sha256": contract.scientific_fingerprint(),
        "checks": checks,
        "registered_images": {
            "actual": registered,
            "expected": contract.reproduction_expected_registered_images,
        },
        "total_moving_images": {
            "actual": len(moving_files),
            "expected": contract.reproduction_expected_total_moving_images,
        },
        "metric_scale": {
            "actual": scale,
            "expected": expected_scale,
            "absolute_delta": abs(scale - expected_scale),
            "absolute_tolerance": scale_tolerance,
        },
        "candidate_counts": {
            "actual": actual_candidate_counts,
            "expected": expected_candidate_counts,
            "total_actual": sum(count for _, _, count in actual_candidate_counts),
            "total_expected": sum(
                count for _, _, count in expected_candidate_counts
            ),
        },
        "selections": {
            "actual": actual_selections,
            "expected": expected_selections,
        },
        "camera_inventory": {
            "actual": list(actual_poses),
            "expected": list(expected_inventory),
        },
        "final_pose_comparison": {
            "locked_reference": locked_relative,
            "locked_reference_sha256": actual_locked_hash,
            "translation_tolerance_m": translation_tolerance,
            "rotation_matrix_tolerance": rotation_tolerance,
            "per_camera": per_camera,
        },
    }


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
    method_contract: str = "baseline_v1",
    direct_target_camera: str = "cam_edge_1",
    top_moving_per_marker: int | None = 8,
    scale_top_per_marker: int | None = 30,
) -> StageResult:
    del observations_root, camera_ids, moving_camera_id
    stage_root = output_root / "06_reproduction_validation"
    contract = resolve_ap01_method_contract(
        method_contract,
        direct_target_camera=direct_target_camera,
        top_moving_per_marker=top_moving_per_marker,
        scale_top_per_marker=scale_top_per_marker,
    )

    def action() -> dict[str, Path | str | bool]:
        report = validate_reproduction_outputs(
            repository_root=Path(__file__).resolve().parents[4],
            dataset=dataset,
            output_root=output_root,
            root_camera=root_camera,
            contract=contract,
        )
        report_path = stage_root / "AP01_REPRODUCTION_VALIDATION.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if not report["success"]:
            failed = [name for name, passed in report["checks"].items() if not passed]
            raise RuntimeError(
                "AP01 historical reproduction validation failed: "
                + ", ".join(failed)
            )
        return {
            "report": report_path,
            "status": str(report["status"]),
            "success": True,
        }

    return run_stage(
        "ap01.validate_reproduction",
        stage_root,
        action,
        inputs={
            "colmap_images": colmap_images(output_root),
            "scale": output_root / "02_metric_scale" / "SCALE_DIAGNOSTICS.json",
            "candidates": (
                output_root / "03_candidates" / "transform_candidates.json"
            ),
            "solution": (
                output_root / "03_static_extrinsics" / "solution_summary.json"
            ),
            "poses": (
                output_root
                / "03_static_extrinsics"
                / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
            ),
        },
        parameters={
            "method_contract": contract.fingerprint_payload(),
            "method_contract_sha256": contract.scientific_fingerprint(),
        },
    )


def main() -> None:
    args = parser(__doc__ or "Validate AP01 historical reproduction").parse_args()
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=cameras(args),
        root_camera=args.root_camera,
        moving_camera_id=args.moving_camera_id,
        method_contract=args.method_contract,
        direct_target_camera=args.direct_target_camera,
        top_moving_per_marker=args.top_moving_per_marker,
        scale_top_per_marker=args.scale_top_per_marker,
    )


if __name__ == "__main__":
    main()
