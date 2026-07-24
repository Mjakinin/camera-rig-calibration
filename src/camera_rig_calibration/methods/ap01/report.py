from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from ._shared import cameras, parser, read_json


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
) -> StageResult:
    stage_root = output_root / "05_report"

    def action() -> dict[str, Path | int | str]:
        solution = read_json(
            output_root / "03_static_extrinsics/solution_summary.json"
        )
        scale = read_json(
            output_root / "02_metric_scale/SCALE_DIAGNOSTICS.json"
        )
        available = list(solution["available_static_cameras"])
        expected = len(camera_ids)
        status = (
            "OK_FULL"
            if len(available) == expected
            else f"PARTIAL_{len(available)}_OF_{expected}"
        )
        diagnostics = {
            "schema_version": 5,
            "approach": "AP01_marker_direct_and_moving_colmap_relay",
            "root_camera": root_camera,
            "metric_scale": scale,
            "static_camera_methods": solution["camera_methods"],
            "per_target_diagnostics": solution[
                "per_target_diagnostics"
            ],
            "available_static_cameras": available,
            "missing_static_cameras": solution[
                "missing_static_cameras"
            ],
            "ground_truth_used": False,
        }
        diagnostics_path = (
            output_root / "03_static_extrinsics/AP01_DIAGNOSTICS.json"
        )
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2) + "\n",
            encoding="utf-8",
        )
        status_path = output_root / "METHOD_STATUS.json"
        status_path.write_text(
            json.dumps(
                {
                    "method": "AP01",
                    "status": status,
                    "success": len(available) == expected,
                    "available_static_cameras": available,
                    "pose_file": str(
                        output_root
                        / "03_static_extrinsics"
                        / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
                    ),
                    "diagnostics_file": str(diagnostics_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if len(available) != expected:
            raise RuntimeError(
                f"AP01 produced only {len(available)}/{expected} "
                "static camera poses"
            )
        return {
            "method_status": status_path,
            "diagnostics": diagnostics_path,
            "status": status,
        }

    return run_stage(
        "ap01.report",
        stage_root,
        action,
        inputs={
            "solution": (
                output_root / "03_static_extrinsics/solution_summary.json"
            ),
            "scale": output_root / "02_metric_scale/SCALE_DIAGNOSTICS.json",
        },
    )


def main() -> None:
    args = parser(__doc__ or "Write AP01 report").parse_args()
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=cameras(args),
        root_camera=args.root_camera,
        moving_camera_id=args.moving_camera_id,
    )


if __name__ == "__main__":
    main()
