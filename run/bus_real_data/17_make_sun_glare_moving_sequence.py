#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_pair(text):
    a, b = text.split(",")
    return float(a), float(b)


def frame_number(path):
    return int(path.stem.split("_")[1])


def copy_metadata(input_dir, output_dir):
    for p in input_dir.iterdir():
        if p.name == "images":
            continue
        dst = output_dir / p.name
        if p.is_file():
            shutil.copy2(p, dst)
        elif p.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(p, dst)


def apply_sun_glare(
    img,
    center_xy,
    radius_rel,
    intensity,
    haze,
    bloom_strength,
    hard_clip,
):
    h, w = img.shape[:2]

    cx = center_xy[0] * w
    cy = center_xy[1] * h
    radius = radius_rel * min(w, h)

    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2

    # Smooth circular sunlight mask.
    mask = np.exp(-d2 / (2.0 * radius * radius)).astype(np.float32)
    mask3 = mask[..., None]

    img_f = img.astype(np.float32)

    # Local exposure increase: pixels near the sun spot become brighter.
    img_f = img_f * (1.0 + intensity * mask3)

    # White haze / washed-out glare component.
    img_f = img_f * (1.0 - haze * mask3) + 255.0 * (haze * mask3)

    if hard_clip:
        img_f = np.clip(img_f, 0, 255)

    # Bloom: saturated bright regions bleed into nearby pixels.
    gray = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    bright = (gray > 220).astype(np.float32)

    bloom = cv2.GaussianBlur(
        bright,
        ksize=(0, 0),
        sigmaX=max(8.0, radius * 0.20),
        sigmaY=max(8.0, radius * 0.20),
    )

    bloom3 = bloom[..., None]
    img_f = img_f + bloom_strength * 255.0 * bloom3

    img_f = np.clip(img_f, 0, 255).astype(np.uint8)

    return img_f, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="results/bus_real_data/03_moving_camera_sequence",
        help="Input moving sequence folder",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output sun-glare degraded sequence folder",
    )
    ap.add_argument(
        "--start",
        default="0.15,0.25",
        help="Sun spot start center as normalized x,y",
    )
    ap.add_argument(
        "--end",
        default="0.85,0.65",
        help="Sun spot end center as normalized x,y",
    )
    ap.add_argument(
        "--radius",
        type=float,
        default=0.22,
        help="Sun spot radius relative to min(image width, height)",
    )
    ap.add_argument(
        "--intensity",
        type=float,
        default=2.8,
        help="Local exposure gain strength",
    )
    ap.add_argument(
        "--haze",
        type=float,
        default=0.50,
        help="White haze strength near sun spot",
    )
    ap.add_argument(
        "--bloom",
        type=float,
        default=0.35,
        help="Bloom strength around saturated regions",
    )
    ap.add_argument(
        "--no-hard-clip",
        action="store_true",
        help="Disable hard clipping before bloom",
    )
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input)
    input_images = input_dir / "images"

    output_dir = Path(args.output)
    output_images = output_dir / "images"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_images.mkdir(parents=True, exist_ok=True)

    copy_metadata(input_dir, output_dir)

    image_paths = sorted(input_images.glob("frame_*.png"), key=frame_number)

    if not image_paths:
        raise RuntimeError(f"No frame_*.png images found in {input_images}")

    start = parse_pair(args.start)
    end = parse_pair(args.end)

    metadata_rows = []

    for i, src in enumerate(image_paths):
        t = i / max(1, len(image_paths) - 1)

        cx = (1.0 - t) * start[0] + t * end[0]
        cy = (1.0 - t) * start[1] + t * end[1]

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {src}")
            continue

        degraded, mask = apply_sun_glare(
            img,
            center_xy=(cx, cy),
            radius_rel=args.radius,
            intensity=args.intensity,
            haze=args.haze,
            bloom_strength=args.bloom,
            hard_clip=not args.no_hard_clip,
        )

        dst = output_images / src.name
        cv2.imwrite(str(dst), degraded)

        metadata_rows.append({
            "frame": frame_number(src),
            "source": str(src),
            "output": str(dst),
            "sun_center_x_norm": cx,
            "sun_center_y_norm": cy,
            "radius": args.radius,
            "intensity": args.intensity,
            "haze": args.haze,
            "bloom": args.bloom,
        })

        if i % 25 == 0:
            print(f"[INFO] wrote {dst}")

    metadata = {
        "type": "sun_glare_moving_sequence",
        "input": str(input_dir),
        "output": str(output_dir),
        "start": args.start,
        "end": args.end,
        "radius": args.radius,
        "intensity": args.intensity,
        "haze": args.haze,
        "bloom": args.bloom,
        "description": (
            "Localized moving sunlight glare with overexposure, haze, clipping, "
            "and bloom. This is intended to be more realistic than global "
            "brightness scaling."
        ),
        "frames": metadata_rows,
    }

    (output_dir / "sun_glare_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print()
    print("[OK] Sun-glare moving sequence written:")
    print(output_dir)
    print("[OK] Images:")
    print(output_images)
    print("[OK] Metadata:")
    print(output_dir / "sun_glare_metadata.json")


if __name__ == "__main__":
    main()
