#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import numpy as np


def apply_ev(img, ev):
    # EV steps: +1 doubles brightness, -1 halves brightness
    factor = 2.0 ** ev
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    # gamma < 1 brightens shadows, gamma > 1 darkens midtones
    x = img.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def apply_flicker(img, frame_idx, amplitude_ev=0.7, period=30):
    ev = amplitude_ev * np.sin(2.0 * np.pi * frame_idx / period)
    return apply_ev(img, ev)


def process_folder(input_dir, output_dir, mode, ev=0.0, gamma=1.0):
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

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable image: {path}")
            continue

        if mode == "ev":
            out = apply_ev(img, ev)
        elif mode == "gamma":
            out = apply_gamma(img, gamma)
        elif mode == "flicker":
            out = apply_flicker(img, i)
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
    parser.add_argument("--mode", required=True, choices=["ev", "gamma", "flicker"])
    parser.add_argument("--ev", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    process_folder(args.input, args.output, args.mode, args.ev, args.gamma)


if __name__ == "__main__":
    main()
