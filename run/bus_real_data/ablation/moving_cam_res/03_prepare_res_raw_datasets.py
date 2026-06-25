#!/usr/bin/env python3

import json
import shutil
from pathlib import Path


BASELINE_RAW = Path("results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images")
WORLD_VARIANTS = Path("results/bus_real_data/ablation/moving_cam_res/00_world_variants")
CAPTURES = Path("results/bus_real_data/ablation/moving_cam_res/00_captures")
OUT_ROOT = Path("results/bus_real_data/ablation/moving_cam_res/00_prepared_datasets")

PROFILES = [
    "res_640x360",
    "res_960x540",
    "res_1280x720_baseline",
    "res_1920x1080",
]


def clean(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    if not BASELINE_RAW.exists():
        raise RuntimeError(f"Missing baseline raw_images: {BASELINE_RAW}")

    for variant in PROFILES:
        capture_images = CAPTURES / variant / "images"
        if not capture_images.exists():
            print(f"[SKIP] {variant}: missing capture images: {capture_images}")
            continue

        n_frames = len(list(capture_images.glob("frame_*.png")))
        if n_frames == 0:
            print(f"[SKIP] {variant}: no frame_*.png in {capture_images}")
            continue

        out_variant = OUT_ROOT / variant
        raw_out = out_variant / "raw_images"
        clean(out_variant)
        raw_out.mkdir(parents=True, exist_ok=True)

        copytree(BASELINE_RAW / "static", raw_out / "static")
        copytree(capture_images, raw_out / "moving")
        copytree(BASELINE_RAW / "camera_info", raw_out / "camera_info")

        moving_info_src = WORLD_VARIANTS / variant / "moving_calib_camera.json"
        moving_info_dst = raw_out / "camera_info" / "moving_calib_camera.json"

        if not moving_info_src.exists():
            raise RuntimeError(f"Missing generated moving camera_info: {moving_info_src}")

        shutil.copy2(moving_info_src, moving_info_dst)

        base_meta = json.loads((WORLD_VARIANTS / variant / "metadata.json").read_text())
        meta = {
            **base_meta,
            "prepared_dataset": str(out_variant),
            "prepared_raw_images": str(raw_out),
            "baseline_static_images_source": str(BASELINE_RAW / "static"),
            "baseline_static_camera_info_source": str(BASELINE_RAW / "camera_info"),
            "moving_capture_source": str(capture_images),
            "moving_frames": n_frames,
            "static_images_copied_from_baseline": True,
            "moving_images_rendered_in_gazebo": True,
            "ready_for_shared_observations": True,
        }

        (out_variant / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

        (out_variant / "README.txt").write_text(
            f"""Moving Camera Resolution Ablation Dataset
=========================================

Variant: {variant}

Static camera images and static camera_info are copied from the baseline.
Moving camera images are rendered from the Gazebo resolution variant.
Moving camera_info is generated for this resolution variant.

Moving frames: {n_frames}

raw_images/static/
raw_images/moving/
raw_images/camera_info/
"""
        )

        print(f"[OK] {variant}")
        print(f"     raw_images: {raw_out}")
        print(f"     frames:     {n_frames}")

    print()
    print("[OK] prepared datasets written to:", OUT_ROOT)


if __name__ == "__main__":
    main()
