#!/usr/bin/env python3
from pathlib import Path
import shutil
import json

BASE_RAW = Path("results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images")
CAPTURE_ROOT = Path("results/bus_real_data/ablation/moving_cam/fov/00_captures")
WORLD_VARIANTS = Path("results/bus_real_data/ablation/moving_cam/fov/00_world_variants")
OUT_ROOT = Path("results/bus_real_data/ablation/moving_cam/fov/00_prepared_datasets")

VARIANTS = [
    "fov_40deg",
    "fov_50deg",
    "fov_60deg",
    "fov_69deg_baseline",
    "fov_80deg",
    "fov_90deg",
    "fov_100deg",
    "fov_110deg",
    "fov_120deg",
    "fov_140deg_extreme",
]

def copytree_clean(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)

def main():
    if not BASE_RAW.exists():
        raise FileNotFoundError(BASE_RAW)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for v in VARIANTS:
        capture_images = CAPTURE_ROOT / v / "images"
        cam_info = WORLD_VARIANTS / v / "moving_calib_camera.json"
        out_raw = OUT_ROOT / v / "raw_images"

        if not capture_images.exists():
            print(f"[SKIP] {v}: missing capture images: {capture_images}")
            continue
        if not cam_info.exists():
            print(f"[SKIP] {v}: missing camera info: {cam_info}")
            continue

        if out_raw.exists():
            shutil.rmtree(out_raw)
        out_raw.mkdir(parents=True, exist_ok=True)

        # Copy baseline raw structure first, then replace moving camera.
        for item in BASE_RAW.iterdir():
            dst = out_raw / item.name
            if item.name == "moving":
                continue
            if item.is_dir():
                shutil.copytree(item, dst, symlinks=False)
            else:
                shutil.copy2(item, dst)

        moving_out = out_raw / "moving"
        shutil.copytree(capture_images, moving_out, symlinks=False)

        cam_info_out_dir = out_raw / "camera_info"
        cam_info_out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cam_info, cam_info_out_dir / "moving_calib_camera.json")

        meta = {
            "ablation_scope": "moving_cam",
            "ablation_study": "fov",
            "variant": v,
            "static_images": "copied from shared baseline",
            "moving_images": str(capture_images),
            "camera_info": str(cam_info),
            "resolution": "1280x720 constant",
        }
        (out_raw / "ablation_metadata.json").write_text(json.dumps(meta, indent=2))

        frame_count = len(list(moving_out.glob("frame_*.png")))
        print(f"[OK] {v}")
        print(f"     raw_images: {out_raw}")
        print(f"     frames:     {frame_count}")

    print(f"\n[OK] prepared FOV datasets written to: {OUT_ROOT}")

if __name__ == "__main__":
    main()
