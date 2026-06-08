#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_txt",
        default="results/beintelli_bus_model/colmap/moving_route_poc/sparse_txt/images.txt",
    )
    args = parser.parse_args()

    path = Path(args.images_txt)
    if not path.exists():
        raise FileNotFoundError(path)

    registered = []

    with path.open() as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()

        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]

            registered.append({
                "image_id": image_id,
                "image_name": name,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "camera_id": camera_id,
            })

            # Next line contains 2D keypoints; skip it.
            i += 2
        else:
            i += 1

    print(f"registered_images: {len(registered)}")

    print("")
    print("registered image names:")
    for row in sorted(registered, key=lambda r: r["image_name"]):
        print(row["image_name"])

    out_csv = path.parent / "registered_images_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "image_name",
                "qw",
                "qx",
                "qy",
                "qz",
                "tx",
                "ty",
                "tz",
                "camera_id",
            ],
        )
        writer.writeheader()
        writer.writerows(registered)

    print("")
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
