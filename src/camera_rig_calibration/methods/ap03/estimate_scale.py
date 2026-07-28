from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage


def run(
    *,
    repository_root: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    mode: str,
    marker_ids: tuple[int, ...],
    marker_length_m: float,
    reprojection_threshold_px: float,
    ransac_iterations: int,
    minimum_inliers: int,
    dictionary: str,
    detection_mode: str = "baseline",
) -> StageResult:
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unsupported AP03 scale mode: {mode}")
    stage_root = output_root / f"scale_{mode}"
    def action() -> dict[str, Path | str]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "camera_rig_calibration.methods.ap03.scale_core",
                "--out-dir",
                str(stage_root),
                "--marker-ids",
                ",".join(str(value) for value in marker_ids),
                "--marker-length-m",
                str(marker_length_m),
                "--reproj-thresh-px",
                str(reprojection_threshold_px),
                "--ransac-iters",
                str(ransac_iterations),
                "--min-inliers",
                str(minimum_inliers),
                "--dictionary",
                dictionary,
                "--detection-mode",
                detection_mode,
                "--txt-root",
                str(output_root / "colmap/reconstruction/sparse_txt"),
                "--image-dir",
                str(output_root / "colmap/dataset/images"),
                "--inspect-summary",
                str(output_root / "colmap/inspection/colmap_model_summary.csv"),
                "--accepted-observations",
                str(observations_root / "shared_all_aruco_observations.csv"),
                "--static-cameras",
                ",".join(camera_ids),
            ],
            cwd=repository_root,
            check=True,
        )
        metadata = (
            stage_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
        )
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if payload.get("scale_m_per_colmap_unit") is None:
            raise RuntimeError(
                f"AP03 {mode} scale is unavailable: "
                f"{payload.get('failure_reason') or payload.get('status')}"
            )
        return {
            "metadata": metadata,
            "static_poses": stage_root
            / "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv",
            "status": str(payload.get("status", "UNKNOWN")),
        }

    return run_stage(
        f"ap03.estimate_scale.{mode}",
        stage_root,
        action,
        inputs={
            "reconstruction": output_root
            / "colmap/reconstruction/sparse_txt",
            "accepted_observations": observations_root,
        },
        parameters={
            "mode": mode,
            "marker_ids": list(marker_ids),
            "marker_length_m": marker_length_m,
            "reprojection_threshold_px": reprojection_threshold_px,
            "ransac_iterations": ransac_iterations,
            "minimum_inliers": minimum_inliers,
            "uses_all_quality_accepted_observations": True,
            "detection_mode": detection_mode,
        },
        failure_is_diagnostic=mode == "single",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--observations-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--marker-ids", required=True)
    parser.add_argument("--marker-length-m", type=float, required=True)
    parser.add_argument("--reprojection-threshold-px", type=float, required=True)
    parser.add_argument("--ransac-iterations", type=int, required=True)
    parser.add_argument("--minimum-inliers", type=int, required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument(
        "--detection-mode",
        choices=("baseline", "subpixel_refined", "high_sensitivity"),
        default="baseline",
    )
    args = parser.parse_args()
    run(
        repository_root=args.repository_root.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=tuple(
            item.strip() for item in args.cameras.split(",") if item.strip()
        ),
        mode=args.mode,
        marker_ids=tuple(
            int(item.strip())
            for item in args.marker_ids.split(",")
            if item.strip()
        ),
        marker_length_m=args.marker_length_m,
        reprojection_threshold_px=args.reprojection_threshold_px,
        ransac_iterations=args.ransac_iterations,
        minimum_inliers=args.minimum_inliers,
        dictionary=args.dictionary,
        detection_mode=args.detection_mode,
    )


if __name__ == "__main__":
    main()
