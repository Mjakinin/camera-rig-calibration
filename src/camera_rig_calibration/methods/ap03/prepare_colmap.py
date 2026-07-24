from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def _first_image(directory: Path, prefix: str) -> Path:
    matches = sorted(
        path
        for path in directory.glob(f"{prefix}.*")
        if path.is_file()
    )
    if not matches:
        raise RuntimeError(f"Missing canonical image for {prefix}")
    return matches[0]


def run(
    *,
    dataset: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    moving_camera_id: str,
) -> StageResult:
    stage_root = output_root / "colmap" / "dataset"
    image_root = stage_root / "images"

    def action() -> dict[str, Path | int]:
        if stage_root.exists():
            shutil.rmtree(stage_root)
        image_root.mkdir(parents=True)
        rows: list[dict[str, str]] = []
        static_root = dataset / "raw_images" / "static"
        for camera in camera_ids:
            source = _first_image(static_root, camera)
            name = f"static_{camera}.png"
            _link_or_copy(source, image_root / name)
            rows.append(
                {
                    "image_name": name,
                    "source_type": "static",
                    "source_id": camera,
                    "source_path": str(source),
                }
            )
        moving = sorted(
            path
            for path in (dataset / "raw_images" / "moving").glob(
                "frame_*.*"
            )
            if path.is_file()
        )
        if not moving:
            raise RuntimeError("AP03 requires moving-camera frames")
        for source in moving:
            name = f"moving_{source.name}"
            _link_or_copy(source, image_root / name)
            rows.append(
                {
                    "image_name": name,
                    "source_type": "moving",
                    "source_id": moving_camera_id,
                    "source_path": str(source),
                }
            )
        manifest = stage_root / "image_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_name",
                    "source_type",
                    "source_id",
                    "source_path",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return {
            "images": image_root,
            "manifest": manifest,
            "static_images": len(camera_ids),
            "moving_images": len(moving),
        }

    return run_stage(
        "ap03.prepare_colmap",
        stage_root,
        action,
        inputs={"canonical_dataset": dataset},
        parameters={
            "camera_ids": list(camera_ids),
            "moving_camera_id": moving_camera_id,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--moving-camera-id", required=True)
    args = parser.parse_args()
    run(
        dataset=args.dataset.resolve(),
        output_root=args.out.resolve(),
        camera_ids=tuple(
            item.strip() for item in args.cameras.split(",") if item.strip()
        ),
        moving_camera_id=args.moving_camera_id,
    )


if __name__ == "__main__":
    main()
