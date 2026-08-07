from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, decode_candidate, parser
from .contracts import AP01MethodContract, resolve_ap01_method_contract


def evaluate_path_gate(
    candidates: list[dict],
    statistics: dict | None,
    *,
    minimum_inlier_ratio: float,
    maximum_translation_dispersion_m: float,
    maximum_rotation_dispersion_deg: float,
    minimum_independent_markers: int | None = None,
) -> dict:
    """Evaluate a GT-free AP01 consensus without changing its pose estimate."""

    stats = dict(statistics or {})
    inliers = [item for item in candidates if bool(item.get("inlier"))]
    marker_ids = sorted(
        {
            int(item.get("root_marker", item.get("target_marker")))
            for item in inliers
            if item.get("root_marker", item.get("target_marker"))
            is not None
        }
    )
    inlier_ratio = (
        len(inliers) / len(candidates) if candidates else 0.0
    )
    translation_dispersion = stats.get(
        "maximum_inlier_translation_dispersion_m"
    )
    rotation_dispersion = stats.get(
        "maximum_inlier_rotation_dispersion_deg"
    )
    checks = {
        "minimum_inlier_ratio": (
            inlier_ratio >= minimum_inlier_ratio
        ),
        "maximum_translation_dispersion_m": (
            translation_dispersion is not None
            and float(translation_dispersion)
            <= maximum_translation_dispersion_m
        ),
        "maximum_rotation_dispersion_deg": (
            rotation_dispersion is not None
            and float(rotation_dispersion)
            <= maximum_rotation_dispersion_deg
        ),
    }
    if minimum_independent_markers is not None:
        checks["minimum_independent_markers"] = (
            len(marker_ids) >= minimum_independent_markers
        )
    return {
        **stats,
        "inlier_marker_ids": marker_ids,
        "independent_inlier_marker_count": len(marker_ids),
        "inlier_ratio": inlier_ratio,
        "gate": {
            "minimum_inlier_ratio": minimum_inlier_ratio,
            "maximum_translation_dispersion_m": (
                maximum_translation_dispersion_m
            ),
            "maximum_rotation_dispersion_deg": (
                maximum_rotation_dispersion_deg
            ),
            "minimum_independent_markers": (
                minimum_independent_markers
            ),
        },
        "gate_checks": checks,
        "stable": bool(checks) and all(checks.values()),
    }


def compare_paths(
    direct_pose: np.ndarray | None,
    relay_pose: np.ndarray | None,
    *,
    maximum_translation_disagreement_m: float,
    maximum_rotation_disagreement_deg: float,
) -> dict:
    if direct_pose is None or relay_pose is None:
        return {
            "available": False,
            "consistent": None,
            "translation_disagreement_m": None,
            "rotation_disagreement_deg": None,
        }
    translation = float(
        np.linalg.norm(direct_pose[:3, 3] - relay_pose[:3, 3])
    )
    rotation = float(
        core.rotation_difference_deg(
            direct_pose[:3, :3], relay_pose[:3, :3]
        )
    )
    return {
        "available": True,
        "consistent": (
            translation <= maximum_translation_disagreement_m
            and rotation <= maximum_rotation_disagreement_deg
        ),
        "translation_disagreement_m": translation,
        "rotation_disagreement_deg": rotation,
        "maximum_translation_disagreement_m": (
            maximum_translation_disagreement_m
        ),
        "maximum_rotation_disagreement_deg": (
            maximum_rotation_disagreement_deg
        ),
    }


def select_candidate_aggregates(
    records: list[dict],
    *,
    camera_ids: tuple[str, ...],
    root_camera: str,
    contract: AP01MethodContract,
    direct_minimum_independent_markers: int = 3,
    direct_minimum_inlier_ratio: float = 0.70,
    direct_maximum_translation_dispersion_m: float = 0.12,
    direct_maximum_rotation_dispersion_deg: float = 4.0,
    relay_minimum_inlier_ratio: float = 0.70,
    relay_maximum_translation_dispersion_m: float = 0.30,
    relay_maximum_rotation_dispersion_deg: float = 7.0,
    maximum_path_translation_disagreement_m: float = 0.12,
    maximum_path_rotation_disagreement_deg: float = 4.0,
    include_flattened: bool = True,
) -> dict:
    """Pure AP01 aggregate/eligibility selection; writes and solvers are absent."""

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in records:
        grouped[str(item["target_camera"])][str(item["mode"])].append(item)

    poses = {root_camera: np.eye(4, dtype=np.float64)}
    accepted_poses = {root_camera: np.eye(4, dtype=np.float64)}
    methods = {root_camera: "gauge_identity"}
    camera_statuses: dict[str, dict] = {
        root_camera: {
            "estimate_status": "available",
            "quality_status": "gauge_identity",
            "deployment_eligible": True,
            "evaluation_status": "available",
            "selected_method": "gauge_identity",
        }
    }
    diagnostics: dict[str, dict] = {}
    quality_warnings: list[str] = []
    relay_chain_reports: dict[str, list[dict]] = {}

    direct_targets = set(contract.direct_targets(camera_ids, root_camera))
    for target in camera_ids:
        if target == root_camera:
            continue
        direct = grouped[target]["direct"]
        relay = grouped[target]["relay"]
        if contract.eligibility_policy == "available_aggregate_is_eligible":
            direct_pose = direct_stats = None
            relay_pose = relay_stats = None
            if direct:
                direct_pose, direct_stats = core.aggregate_legacy_direct_candidates(
                    direct, contract
                )
            if relay:
                relay_pose, relay_stats = core.aggregate_legacy_relay_candidates(
                    relay, contract
                )
            fixed_type = "direct" if target in direct_targets else "relay"
            selected = direct_pose if fixed_type == "direct" else relay_pose
            selected_stats = direct_stats if fixed_type == "direct" else relay_stats
            if selected is None:
                source_name = "unavailable"
                quality_status = (
                    "unavailable_missing_direct_aggregate"
                    if fixed_type == "direct"
                    else "unavailable_missing_relay_aggregate"
                )
                deployment_eligible = False
                quality_warnings.append(f"{target}:{quality_status}")
            else:
                source_name = (
                    "direct_static_aruco_multimarker"
                    if fixed_type == "direct"
                    else "moving_relay_multichain_colmap_motion_aruco_metric_scale"
                )
                quality_status = "accepted_legacy_available_aggregate"
                deployment_eligible = bool(np.all(np.isfinite(selected)))
                if not deployment_eligible:
                    selected = None
                    source_name = "unavailable"
                    quality_status = "unavailable_non_finite_estimate"
                    quality_warnings.append(f"{target}:{quality_status}")
            if selected is not None:
                poses[target] = selected
                methods[target] = source_name
                if deployment_eligible:
                    accepted_poses[target] = selected
            camera_statuses[target] = {
                "estimate_status": "available" if selected is not None else "unavailable",
                "quality_status": quality_status,
                "deployment_eligible": deployment_eligible,
                "evaluation_status": "available" if selected is not None else "unavailable",
                "selected_method": source_name,
            }
            diagnostics[target] = {
                "selected_method": source_name,
                "selected_candidate_type": fixed_type if selected is not None else None,
                "quality_warning": None if deployment_eligible else quality_status,
                **camera_statuses[target],
                "selected_aggregate": selected_stats,
                "direct": direct_stats,
                "relay": relay_stats,
                "relay_raw_candidate_count": len(relay),
                "relay_independent_chain_count": None,
                "direct_relay_consistency": {
                    "available": direct_pose is not None and relay_pose is not None,
                    "consistent": None,
                    "gate_applied": False,
                },
            }
            relay_chain_reports[target] = []
            continue

        if contract.eligibility_policy != "configured_consensus_gates":
            raise ValueError(
                f"Unknown AP01 eligibility policy: {contract.eligibility_policy}"
            )
        direct_pose = direct_stats = None
        relay_pose = relay_stats = None
        relay_chains: list[dict] = []
        if direct:
            direct_pose, direct_stats = core.aggregate_direct_marker_estimates(direct)
        if relay:
            relay_pose, relay_stats, relay_chains = (
                core.aggregate_relay_marker_chains(relay)
            )
        direct_stats = evaluate_path_gate(
            direct,
            direct_stats,
            minimum_inlier_ratio=direct_minimum_inlier_ratio,
            maximum_translation_dispersion_m=direct_maximum_translation_dispersion_m,
            maximum_rotation_dispersion_deg=direct_maximum_rotation_dispersion_deg,
            minimum_independent_markers=direct_minimum_independent_markers,
        )
        relay_stats = evaluate_path_gate(
            relay_chains,
            relay_stats,
            minimum_inlier_ratio=relay_minimum_inlier_ratio,
            maximum_translation_dispersion_m=relay_maximum_translation_dispersion_m,
            maximum_rotation_dispersion_deg=relay_maximum_rotation_dispersion_deg,
        )
        path_comparison = compare_paths(
            direct_pose,
            relay_pose,
            maximum_translation_disagreement_m=maximum_path_translation_disagreement_m,
            maximum_rotation_disagreement_deg=maximum_path_rotation_disagreement_deg,
        )
        direct_stable = bool(direct_stats["stable"])
        relay_stable = bool(relay_stats["stable"])
        warning = None
        deployment_eligible = False
        if direct_stable and relay_stable and path_comparison["consistent"] is False:
            selected, source_name = direct_pose, "direct_multimarker"
            quality_status = "rejected_direct_relay_disagreement"
            warning = quality_status
        elif direct_stable:
            selected, source_name = direct_pose, "direct_multimarker"
            deployment_eligible = True
            quality_status = "accepted"
        elif relay_stable:
            selected, source_name = relay_pose, "moving_colmap_relay"
            deployment_eligible = True
            quality_status = "accepted"
        else:
            selected = direct_pose if direct_pose is not None else relay_pose
            source_name = (
                "direct_multimarker_diagnostic"
                if direct_pose is not None
                else "moving_colmap_relay_diagnostic"
                if relay_pose is not None
                else "unavailable"
            )
            quality_status = (
                "rejected_unstable_consensus"
                if selected is not None
                else "unavailable_no_finite_estimate"
            )
            warning = quality_status
        if warning:
            quality_warnings.append(f"{target}:{warning}")
        if selected is not None:
            if not np.all(np.isfinite(selected)):
                selected = None
                source_name = "unavailable"
                quality_status = "unavailable_non_finite_estimate"
                deployment_eligible = False
            else:
                poses[target] = selected
                methods[target] = source_name
                if deployment_eligible:
                    accepted_poses[target] = selected
        camera_statuses[target] = {
            "estimate_status": "available" if selected is not None else "unavailable",
            "quality_status": quality_status,
            "deployment_eligible": deployment_eligible,
            "evaluation_status": "available" if selected is not None else "unavailable",
            "selected_method": source_name,
        }
        diagnostics[target] = {
            "selected_method": source_name,
            "selected_candidate_type": (
                "direct" if selected is direct_pose and selected is not None
                else "relay" if selected is relay_pose and selected is not None
                else None
            ),
            "quality_warning": warning,
            **camera_statuses[target],
            "direct": direct_stats,
            "relay": relay_stats,
            "relay_raw_candidate_count": len(relay),
            "relay_independent_chain_count": len(relay_chains),
            "direct_relay_consistency": path_comparison,
        }
        relay_chain_reports[target] = list(relay_stats.get("chain_reports", []))

    return {
        "poses": poses,
        "accepted_poses": accepted_poses,
        "methods": methods,
        "camera_statuses": camera_statuses,
        "diagnostics": diagnostics,
        "quality_warnings": quality_warnings,
        "flattened": (
            [core.serializable_candidate(item) for item in records]
            if include_flattened
            else []
        ),
        "relay_chain_reports": relay_chain_reports,
    }


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
    direct_minimum_independent_markers: int = 3,
    direct_minimum_inlier_ratio: float = 0.70,
    direct_maximum_translation_dispersion_m: float = 0.12,
    direct_maximum_rotation_dispersion_deg: float = 4.0,
    relay_minimum_inlier_ratio: float = 0.70,
    relay_maximum_translation_dispersion_m: float = 0.30,
    relay_maximum_rotation_dispersion_deg: float = 7.0,
    maximum_path_translation_disagreement_m: float = 0.12,
    maximum_path_rotation_disagreement_deg: float = 4.0,
    method_contract: str = "baseline_v1",
    direct_target_camera: str = "cam_edge_1",
    top_moving_per_marker: int | None = 8,
) -> StageResult:
    stage_root = output_root / "03_static_extrinsics"
    contract = resolve_ap01_method_contract(
        method_contract,
        direct_target_camera=direct_target_camera,
        top_moving_per_marker=top_moving_per_marker,
    )

    def action() -> dict[str, Path | int]:
        source = output_root / "03_candidates/transform_candidates.json"
        records = json.loads(source.read_text(encoding="utf-8"))
        selected_result = select_candidate_aggregates(
            [decode_candidate(encoded) for encoded in records],
            camera_ids=camera_ids,
            root_camera=root_camera,
            contract=contract,
            direct_minimum_independent_markers=direct_minimum_independent_markers,
            direct_minimum_inlier_ratio=direct_minimum_inlier_ratio,
            direct_maximum_translation_dispersion_m=direct_maximum_translation_dispersion_m,
            direct_maximum_rotation_dispersion_deg=direct_maximum_rotation_dispersion_deg,
            relay_minimum_inlier_ratio=relay_minimum_inlier_ratio,
            relay_maximum_translation_dispersion_m=relay_maximum_translation_dispersion_m,
            relay_maximum_rotation_dispersion_deg=relay_maximum_rotation_dispersion_deg,
            maximum_path_translation_disagreement_m=maximum_path_translation_disagreement_m,
            maximum_path_rotation_disagreement_deg=maximum_path_rotation_disagreement_deg,
        )
        poses = selected_result["poses"]
        accepted_poses = selected_result["accepted_poses"]
        methods = selected_result["methods"]
        camera_statuses = selected_result["camera_statuses"]
        diagnostics = selected_result["diagnostics"]
        quality_warnings = selected_result["quality_warnings"]
        flattened = selected_result["flattened"]
        relay_chain_reports = selected_result["relay_chain_reports"]

        poses = {
            camera: (
                transform
                if camera == root_camera
                else core.serialize_final_pose(transform, contract)
            )
            for camera, transform in poses.items()
        }
        accepted_poses = {
            camera: poses[camera] for camera in accepted_poses
        }

        stage_root.mkdir(parents=True, exist_ok=True)
        pose_fields = [
            "entity_type",
            "entity_id",
            "source",
            "x_m",
            "y_m",
            "z_m",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "rvec_x",
            "rvec_y",
            "rvec_z",
            "estimate_status",
            "quality_status",
            "deployment_eligible",
            "evaluation_status",
        ]
        pose_rows = []
        for camera, pose in sorted(poses.items()):
            row = core.pose_row(camera, pose, methods[camera])
            row.update(camera_statuses[camera])
            row.pop("selected_method", None)
            pose_rows.append(row)
        pose_file = (
            stage_root / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
        )
        core.write_csv(pose_file, pose_rows, pose_fields)
        accepted_pose_file = (
            stage_root / "AP01_STATIC_CAMERA_POSES_ACCEPTED.csv"
        )
        core.write_csv(
            accepted_pose_file,
            [
                row
                for row in pose_rows
                if bool(camera_statuses[row["entity_id"]][
                    "deployment_eligible"
                ])
            ],
            pose_fields,
        )
        if root_camera == "cam_edge_3":
            core.write_csv(
                stage_root / "AP01_STATIC_CAMERA_POSES_CAM3_REFERENCE.csv",
                pose_rows,
                pose_fields,
            )
        pairwise = stage_root / "AP01_PAIRWISE_DISTANCES.csv"
        core.CAMERAS = list(camera_ids)
        core.write_csv(
            pairwise,
            core.pairwise_rows(poses),
            ["camera_a", "camera_b", "distance_m"],
        )
        core.write_csv(
            stage_root / "AP01_TRANSFORM_CANDIDATES.csv", flattened
        )
        (
            stage_root / "AP01_RELAY_CHAIN_DIAGNOSTICS.json"
        ).write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "algorithm": contract.relay_aggregation_policy,
                    "ground_truth_used": False,
                    "targets": relay_chain_reports,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        solution = stage_root / "solution_summary.json"
        solution.write_text(
            json.dumps(
                {
                    "root_camera": root_camera,
                    "method_contract": contract.fingerprint_payload(),
                    "method_contract_sha256": contract.scientific_fingerprint(),
                    "camera_methods": methods,
                    "camera_statuses": camera_statuses,
                    "per_target_diagnostics": diagnostics,
                    "available_static_cameras": sorted(poses),
                    "deployment_eligible_cameras": sorted(accepted_poses),
                    "missing_static_cameras": sorted(
                        set(camera_ids) - set(poses)
                    ),
                    "quality_warnings": quality_warnings,
                    "quality_status": (
                        "rejected_unstable_consensus"
                        if any(
                            "rejected_unstable_consensus" in item
                            for item in quality_warnings
                        )
                        else "rejected_direct_relay_disagreement"
                        if quality_warnings
                        else "good"
                    ),
                    "quality_gates": {
                        "direct": {
                            "minimum_independent_markers": (
                                direct_minimum_independent_markers
                            ),
                            "minimum_inlier_ratio": (
                                direct_minimum_inlier_ratio
                            ),
                            "maximum_translation_dispersion_m": (
                                direct_maximum_translation_dispersion_m
                            ),
                            "maximum_rotation_dispersion_deg": (
                                direct_maximum_rotation_dispersion_deg
                            ),
                        },
                        "relay": {
                            "minimum_inlier_ratio": (
                                relay_minimum_inlier_ratio
                            ),
                            "maximum_translation_dispersion_m": (
                                relay_maximum_translation_dispersion_m
                            ),
                            "maximum_rotation_dispersion_deg": (
                                relay_maximum_rotation_dispersion_deg
                            ),
                        },
                        "direct_relay_consistency": {
                            "maximum_translation_disagreement_m": (
                                maximum_path_translation_disagreement_m
                            ),
                            "maximum_rotation_disagreement_deg": (
                                maximum_path_rotation_disagreement_deg
                            ),
                        },
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "poses": pose_file,
            "accepted_poses": accepted_pose_file,
            "pairwise": pairwise,
            "solution_summary": solution,
            "solved_cameras": len(poses),
        }

    return run_stage(
        "ap01.solve_extrinsics",
        stage_root,
        action,
        inputs={
            "candidates": (
                output_root / "03_candidates/transform_candidates.json"
            )
        },
        parameters={
            "root_camera": root_camera,
            "method_contract": contract.fingerprint_payload(),
            "method_contract_sha256": contract.scientific_fingerprint(),
            "direct_quality_gate": {
                "minimum_independent_markers": direct_minimum_independent_markers,
                "minimum_inlier_ratio": direct_minimum_inlier_ratio,
                "maximum_translation_dispersion_m": direct_maximum_translation_dispersion_m,
                "maximum_rotation_dispersion_deg": direct_maximum_rotation_dispersion_deg,
            },
            "relay_quality_gate": {
                "minimum_inlier_ratio": relay_minimum_inlier_ratio,
                "maximum_translation_dispersion_m": relay_maximum_translation_dispersion_m,
                "maximum_rotation_dispersion_deg": relay_maximum_rotation_dispersion_deg,
            },
            "direct_relay_consistency": {
                "maximum_translation_disagreement_m": maximum_path_translation_disagreement_m,
                "maximum_rotation_disagreement_deg": maximum_path_rotation_disagreement_deg,
            },
        },
    )


def main() -> None:
    args = parser(__doc__ or "Solve AP01 static extrinsics").parse_args()
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=cameras(args),
        root_camera=args.root_camera,
        moving_camera_id=args.moving_camera_id,
        direct_minimum_independent_markers=(
            args.direct_minimum_independent_markers
        ),
        direct_minimum_inlier_ratio=args.direct_minimum_inlier_ratio,
        direct_maximum_translation_dispersion_m=(
            args.direct_maximum_translation_dispersion_m
        ),
        direct_maximum_rotation_dispersion_deg=(
            args.direct_maximum_rotation_dispersion_deg
        ),
        relay_minimum_inlier_ratio=args.relay_minimum_inlier_ratio,
        relay_maximum_translation_dispersion_m=(
            args.relay_maximum_translation_dispersion_m
        ),
        relay_maximum_rotation_dispersion_deg=(
            args.relay_maximum_rotation_dispersion_deg
        ),
        maximum_path_translation_disagreement_m=(
            args.maximum_path_translation_disagreement_m
        ),
        maximum_path_rotation_disagreement_deg=(
            args.maximum_path_rotation_disagreement_deg
        ),
        method_contract=args.method_contract,
        direct_target_camera=args.direct_target_camera,
        top_moving_per_marker=args.top_moving_per_marker,
    )


if __name__ == "__main__":
    main()
