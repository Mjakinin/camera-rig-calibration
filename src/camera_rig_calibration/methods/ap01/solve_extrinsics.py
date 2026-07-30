from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, decode_candidate, parser


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
) -> StageResult:
    stage_root = output_root / "03_static_extrinsics"

    def action() -> dict[str, Path | int]:
        source = output_root / "03_candidates/transform_candidates.json"
        records = json.loads(source.read_text(encoding="utf-8"))
        grouped: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for encoded in records:
            item = decode_candidate(encoded)
            grouped[str(item["target_camera"])][str(item["mode"])].append(
                item
            )

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
        flattened: list[dict] = []
        relay_chain_reports: dict[str, list[dict]] = {}
        for target in camera_ids:
            if target == root_camera:
                continue
            direct = grouped[target]["direct"]
            relay = grouped[target]["relay"]
            direct_pose = direct_stats = None
            relay_pose = relay_stats = None
            relay_chains: list[dict] = []
            if direct:
                direct_pose, direct_stats = (
                    core.aggregate_direct_marker_estimates(direct)
                )
            if relay:
                relay_pose, relay_stats, relay_chains = (
                    core.aggregate_relay_marker_chains(relay)
                )
            direct_stats = evaluate_path_gate(
                direct,
                direct_stats,
                minimum_inlier_ratio=direct_minimum_inlier_ratio,
                maximum_translation_dispersion_m=(
                    direct_maximum_translation_dispersion_m
                ),
                maximum_rotation_dispersion_deg=(
                    direct_maximum_rotation_dispersion_deg
                ),
                minimum_independent_markers=(
                    direct_minimum_independent_markers
                ),
            )
            relay_stats = evaluate_path_gate(
                relay_chains,
                relay_stats,
                minimum_inlier_ratio=relay_minimum_inlier_ratio,
                maximum_translation_dispersion_m=(
                    relay_maximum_translation_dispersion_m
                ),
                maximum_rotation_dispersion_deg=(
                    relay_maximum_rotation_dispersion_deg
                ),
            )
            path_comparison = compare_paths(
                direct_pose,
                relay_pose,
                maximum_translation_disagreement_m=(
                    maximum_path_translation_disagreement_m
                ),
                maximum_rotation_disagreement_deg=(
                    maximum_path_rotation_disagreement_deg
                ),
            )
            direct_stable = bool(direct_stats["stable"])
            relay_stable = bool(relay_stats["stable"])
            warning = None
            deployment_eligible = False
            quality_status = "unavailable"
            if (
                direct_stable
                and relay_stable
                and path_comparison["consistent"] is False
            ):
                selected, source_name = direct_pose, "direct_multimarker"
                quality_status = "rejected_direct_relay_disagreement"
                warning = quality_status
                quality_warnings.append(f"{target}:{warning}")
            elif direct_stable:
                selected, source_name = direct_pose, "direct_multimarker"
                deployment_eligible = True
                quality_status = "accepted"
            elif relay_stable:
                selected, source_name = relay_pose, "moving_colmap_relay"
                deployment_eligible = True
                quality_status = "accepted"
            else:
                # Preserve the strongest finite GT-free estimate for scientific
                # evaluation while explicitly blocking deployment.
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
                "estimate_status": (
                    "available" if selected is not None else "unavailable"
                ),
                "quality_status": quality_status,
                "deployment_eligible": deployment_eligible,
                "evaluation_status": (
                    "available" if selected is not None else "unavailable"
                ),
                "selected_method": source_name,
            }
            diagnostics[target] = {
                "selected_method": source_name,
                "quality_warning": warning,
                **camera_statuses[target],
                "direct": direct_stats,
                "relay": relay_stats,
                "relay_raw_candidate_count": len(relay),
                "relay_independent_chain_count": len(relay_chains),
                "direct_relay_consistency": path_comparison,
            }
            relay_chain_reports[target] = list(
                relay_stats.get("chain_reports", [])
            )
            flattened.extend(core.serializable_candidate(item) for item in direct)
            flattened.extend(core.serializable_candidate(item) for item in relay)

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
                    "algorithm": (
                        "hierarchical_weighted_mean_of_mad_inliers_"
                        "no_gt_selection"
                    ),
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
        parameters={"root_camera": root_camera},
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
    )


if __name__ == "__main__":
    main()
