#!/usr/bin/env python3
import argparse
from pathlib import Path

def read_registered_image_names(images_txt: Path):
    names = []
    with images_txt.open() as f:
        lines = list(f)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            names.append(parts[9])
            i += 2
        else:
            i += 1

    return names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    images_dir = dataset / "images"
    images_txt = dataset / "sparse_txt" / "images.txt"
    points_txt = dataset / "sparse_txt" / "points3D.txt"

    image_files = sorted([p.name for p in images_dir.glob("*.jpg")])
    if not image_files:
        image_files = sorted([p.name for p in images_dir.glob("*.png")])

    registered = read_registered_image_names(images_txt)

    n_total = len(image_files)
    n_reg = len(registered)
    ratio = 100.0 * n_reg / n_total if n_total else 0.0

    reg_set = set(registered)
    missing = [name for name in image_files if name not in reg_set]

    n_points = 0
    if points_txt.exists():
        with points_txt.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    n_points += 1

    print("")
    print("NO-GT COLMAP ROUTE COVERAGE")
    print("===========================")
    print(f"dataset_dir:        {dataset}")
    print(f"total images:       {n_total}")
    print(f"registered images:  {n_reg}")
    print(f"registration ratio: {ratio:.2f}%")
    print(f"sparse points:      {n_points}")

    if missing:
        print("")
        print("missing/unregistered images:")
        for name in missing[:50]:
            print("  ", name)
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    else:
        print("")
        print("[OK] all captured images are registered")

if __name__ == "__main__":
    main()
