from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage


def run(
    *,
    output_root: Path,
    camera_ids: tuple[str, ...],
) -> StageResult:
    stage_root = output_root / "colmap" / "inspection"
    dataset = output_root / "colmap" / "dataset"
    reconstruction = output_root / "colmap" / "reconstruction"

    def action() -> dict[str, Path]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "camera_rig_calibration.methods.ap03.inspect",
                "--dataset-root",
                str(dataset),
                "--txt-root",
                str(reconstruction / "sparse_txt"),
                "--out",
                str(stage_root),
                "--cameras",
                ",".join(camera_ids),
            ],
            check=True,
        )
        return {
            "model_summary": stage_root / "colmap_model_summary.csv",
            "registered_images": stage_root
            / "registered_images_by_model.csv",
        }

    return run_stage(
        "ap03.inspect",
        stage_root,
        action,
        inputs={"reconstruction": reconstruction},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    args = parser.parse_args()
    run(
        output_root=args.out.resolve(),
        camera_ids=tuple(
            item.strip() for item in args.cameras.split(",") if item.strip()
        ),
    )


if __name__ == "__main__":
    main()
