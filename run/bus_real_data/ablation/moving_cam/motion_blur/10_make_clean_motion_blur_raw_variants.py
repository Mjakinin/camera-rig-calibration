#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path("results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1")
OUT_ROOT = Path("results/bus_real_data/ablation/moving_cam/motion_blur")

VARIANTS = {
    "moving_blur_k00_baseline": 0,
    "moving_blur_k09_mild": 9,
    "moving_blur_k21_strong": 21,
    "moving_blur_k41_extreme": 41,
}

def ensure_odd(k: int) -> int:
    if k <= 1:
        return 1
    return k if k % 2 == 1 else k + 1

def motion_blur_kernel(length: int, angle_deg: float = 0.0):
    length = ensure_odd(length)
    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0

    center = (length / 2 - 0.5, length / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    s = kernel.sum()
    if s <= 0:
        kernel[length // 2, :] = 1.0
        s = kernel.sum()
    kernel /= s
    return kernel

def blur_image(img, ksize: int, angle_deg: float = 0.0):
    if ksize <= 1:
        return img
    kernel = motion_blur_kernel(ksize, angle_deg)
    return cv2.filter2D(img, -1, kernel)

def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def main():
    if not SRC.exists():
        raise SystemExit(f"[ERROR] source not found: {SRC}")

    for variant, ksize in VARIANTS.items():
        out_dir = OUT_ROOT / variant
        raw_dst = out_dir / "raw_images"

        print(f"[INFO] building {variant} (kernel={ksize})")
        copy_tree(SRC / "raw_images", raw_dst)

        moving_dir = raw_dst / "moving"
        if not moving_dir.exists():
            raise SystemExit(f"[ERROR] missing moving dir: {moving_dir}")

        for img_path in sorted(moving_dir.glob("frame_*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"[ERROR] failed reading: {img_path}")

            # Main ablation decision:
            # use one fixed linear motion-blur model across all variants.
            # Angle fixed at 0 deg to isolate blur magnitude only.
            out = blur_image(img, ksize, angle_deg=0.0)
            cv2.imwrite(str(img_path), out)

        meta = {
            "variant": variant,
            "effect": "linear_motion_blur",
            "kernel_size": int(ksize),
            "angle_deg": 0.0,
            "applied_to": "moving_images_only",
            "static_images_changed": False,
            "camera_info_changed": False,
            "source_dataset": str(SRC),
        }
        (out_dir / "VARIANT_METADATA.json").write_text(json.dumps(meta, indent=2))
        print(f"[OK] {variant}")

if __name__ == "__main__":
    main()
