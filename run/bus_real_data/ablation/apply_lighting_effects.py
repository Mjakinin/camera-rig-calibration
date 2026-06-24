#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import numpy as np


def apply_low_light(img, alpha=0.45, beta=-20):
    # dunkler + etwas weniger Kontrast
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_side_light(img, strength=0.75, direction="left"):
    # links/rechts gerichteter Beleuchtungsgradient
    h, w = img.shape[:2]

    x = np.linspace(0, 1, w, dtype=np.float32)
    if direction == "right":
        x = x[::-1]

    gradient = 1.0 + strength * (1.0 - x)
    gradient = gradient.reshape(1, w, 1)

    out = img.astype(np.float32) * gradient
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_glare(img, strength=1.2, center_x=0.35, center_y=0.35, radius=0.35):
    # heller Spot / Überbelichtung
    h, w = img.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]
    cx = center_x * w
    cy = center_y * h
    r = radius * min(w, h)

    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = np.exp(-dist2 / (2 * (r ** 2))).astype(np.float32)

    glare = np.zeros_like(img, dtype=np.float32)
    glare[:, :, 0] = 200
    glare[:, :, 1] = 220
    glare[:, :, 2] = 255

    out = img.astype(np.float32) + strength * mask[:, :, None] * glare
    return np.clip(out, 0, 255).astype(np.uint8)


def process_folder(input_dir, output_dir, mode):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        if mode == "low_light":
            out = apply_low_light(img)
        elif mode == "side_light":
            out = apply_side_light(img)
        elif mode == "glare":
            out = apply_glare(img)
        else:
            raise SystemExit(f"Unknown mode: {mode}")

        cv2.imwrite(str(output_dir / path.name), out)

    print(f"Processed {len(image_paths)} images")
    print(f"Mode: {mode}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=["low_light", "side_light", "glare"])
    args = parser.parse_args()

    process_folder(args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
