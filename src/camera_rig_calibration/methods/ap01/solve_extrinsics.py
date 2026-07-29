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
        methods = {root_camera: "gauge_identity"}
        diagnostics: dict[str, dict] = {}
        quality_warnings: list[str] = []
        flattened: list[dict] = []
        for target in camera_ids:
            if target == root_camera:
                continue
            direct = grouped[target]["direct"]
            relay = grouped[target]["relay"]
            direct_pose = direct_stats = None
            relay_pose = relay_stats = None
            if direct:
                direct_pose, direct_stats = core.aggregate_candidates(
                    direct, translation_floor=0.12, rotation_floor=4.0
                )
            if relay:
                relay_pose, relay_stats = core.aggregate_candidates(
                    relay, translation_floor=0.30, rotation_floor=7.0
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
                relay,
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
            if direct_stable:
                selected, source_name = direct_pose, "direct_multimarker"
                if (
                    relay_stable
                    and path_comparison["consistent"] is False
                ):
                    warning = "warning_direct_relay_disagreement"
                    quality_warnings.append(f"{target}:{warning}")
            elif relay_stable:
                selected, source_name = relay_pose, "moving_colmap_relay"
            else:
                selected, source_name = None, "quality_rejected"
                warning = "warning_unstable_consensus"
                quality_warnings.append(f"{target}:{warning}")
            if selected is not None:
                poses[target] = selected
                methods[target] = source_name
            diagnostics[target] = {
                "selected_method": source_name,
                "quality_warning": warning,
                "direct": direct_stats,
                "relay": relay_stats,
                "relay_candidates": len(relay),
                "direct_relay_consistency": path_comparison,
            }
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
        ]
        pose_rows = [
            core.pose_row(camera, pose, methods[camera])
            for camera, pose in sorted(poses.items())
        ]
        pose_file = (
            stage_root / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
        )
        core.write_csv(pose_file, pose_rows, pose_fields)
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
        solution = stage_root / "solution_summary.json"
        solution.write_text(
            json.dumps(
                {
                    "root_camera": root_camera,
                    "camera_methods": methods,
                    "per_target_diagnostics": diagnostics,
                    "available_static_cameras": sorted(poses),
                    "missing_static_cameras": sorted(
                        set(camera_ids) - set(poses)
                    ),
                    "quality_warnings": quality_warnings,
                    "quality_status": (
                        "warning_unstable_consensus"
                        if any(
                            "warning_unstable_consensus" in item
                            for item in quality_warnings
                        )
                        else "warning_direct_relay_disagreement"
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
