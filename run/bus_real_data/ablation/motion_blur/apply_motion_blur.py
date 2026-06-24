#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import numpy as np

def motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    if size <= 1:
        return np.eye(1, dtype=np.float32)

    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0

    center = (size / 2 - 0.5, size / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    kernel /= max(kernel.sum(), 1e-8)
    return kernel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", required=True, help="Output image folder")
    parser.add_argument("--kernel", type=int, default=15, help="Blur kernel size")
    parser.add_argument("--angle", type=float, default=0.0, help="Blur angle in degrees")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel = motion_blur_kernel(args.kernel, args.angle)

    image_paths = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )

    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable image: {path}")
            continue

        if args.kernel > 1:
            img = cv2.filter2D(img, -1, kernel)

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), img)

    print(f"Processed {len(image_paths)} images")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    main()
