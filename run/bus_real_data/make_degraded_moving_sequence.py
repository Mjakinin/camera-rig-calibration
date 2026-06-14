#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def ensure_odd(k: int) -> int:
    if k <= 1:
        return 1
    return k if k % 2 == 1 else k + 1


def apply_gaussian_blur(img, ksize: int):
    ksize = ensure_odd(ksize)
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_motion_blur(img, length: int):
    length = max(1, int(length))
    if length <= 1:
        return img

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    kernel /= float(length)

    return cv2.filter2D(img, -1, kernel)


def apply_brightness(img, factor: float):
    out = img.astype(np.float32) * float(factor)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_contrast(img, factor: float):
    out = (img.astype(np.float32) - 127.5) * float(factor) + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(img, gamma: float):
    gamma = max(float(gamma), 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255.0
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(img, table)


def copy_metadata(src_seq: Path, dst_seq: Path):
    for name in [
        "route_commanded.csv",
        "README.txt",
        "capture_report.txt",
    ]:
        src = src_seq / name
        if src.exists():
            shutil.copy2(src, dst_seq / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="results/bus_real_data/03_moving_camera_sequence",
        help="Input moving sequence folder containing images/",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output sequence folder to create",
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=[
            "gaussian_blur",
            "motion_blur",
            "brightness",
            "contrast",
            "gamma",
        ],
    )
    ap.add_argument(
        "--value",
        required=True,
        type=float,
        help="Blur kernel/length or brightness/contrast/gamma factor",
    )
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    src_seq = Path(args.input)
    dst_seq = Path(args.output)

    src_img_dir = src_seq / "images"
    dst_img_dir = dst_seq / "images"

    if not src_img_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {src_img_dir}")

    if args.clean and dst_seq.exists():
        shutil.rmtree(dst_seq)

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata(src_seq, dst_seq)

    images = sorted(src_img_dir.glob("frame_*.png"))
    if not images:
        raise RuntimeError(f"No frame_*.png images found in {src_img_dir}")

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] could not read {img_path}")
            continue

        if args.mode == "gaussian_blur":
            out = apply_gaussian_blur(img, int(args.value))
        elif args.mode == "motion_blur":
            out = apply_motion_blur(img, int(args.value))
        elif args.mode == "brightness":
            out = apply_brightness(img, args.value)
        elif args.mode == "contrast":
            out = apply_contrast(img, args.value)
        elif args.mode == "gamma":
            out = apply_gamma(img, args.value)
        else:
            raise ValueError(args.mode)

        cv2.imwrite(str(dst_img_dir / img_path.name), out)

    metadata = {
        "input_sequence": str(src_seq),
        "output_sequence": str(dst_seq),
        "mode": args.mode,
        "value": args.value,
        "num_images": len(images),
    }

    (dst_seq / "degradation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("[OK] wrote degraded sequence:", dst_seq)
    print("[OK] images:", dst_img_dir)
    print("[OK] metadata:", dst_seq / "degradation_metadata.json")


if __name__ == "__main__":
    main()
