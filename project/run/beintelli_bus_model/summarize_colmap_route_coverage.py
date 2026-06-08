#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from collections import Counter


def read_registered(images_txt: Path):
    names = []
    with images_txt.open() as f:
        lines = [line.strip() for line in f]

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            names.append(parts[9])
            i += 2
        else:
            i += 1

    return set(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    images_txt = dataset / "sparse_txt" / "images.txt"
    route_gt = dataset / "route_gt.csv"

    if not images_txt.exists():
        raise FileNotFoundError(images_txt)
    if not route_gt.exists():
        raise FileNotFoundError(route_gt)

    registered = read_registered(images_txt)

    rows = []
    with route_gt.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["registered"] = row["image_name"] in registered
            rows.append(row)

    total = len(rows)
    reg = sum(1 for r in rows if r["registered"])

    print(f"dataset: {dataset}")
    print(f"total route frames: {total}")
    print(f"registered frames:  {reg}")
    print(f"registration rate:  {100.0 * reg / total:.1f}%")

    reg_rows = [r for r in rows if r["registered"]]

    if reg_rows:
        first = reg_rows[0]
        last = reg_rows[-1]

        print("")
        print(
            "first registered:",
            first["image_name"],
            "tag=", first["tag"],
            "pose=",
            first["x"],
            first["y"],
            first["z"],
            "yaw=",
            first["yaw"],
        )
        print(
            "last registered: ",
            last["image_name"],
            "tag=", last["tag"],
            "pose=",
            last["x"],
            last["y"],
            last["z"],
            "yaw=",
            last["yaw"],
        )

    print("")
    print("registered by tag:")
    counter = Counter(r["tag"] for r in reg_rows)
    for tag, count in sorted(counter.items()):
        print(f"  {tag}: {count}")

    print("")
    print("missing ranges:")
    missing = [r["image_name"] for r in rows if not r["registered"]]
    if not missing:
        print("  none")
    else:
        ranges = []
        start = prev = None

        for name in missing:
            idx = int(name.replace("moving_", "").replace(".jpg", ""))
            if start is None:
                start = prev = idx
            elif idx == prev + 1:
                prev = idx
            else:
                ranges.append((start, prev))
                start = prev = idx

        ranges.append((start, prev))

        for a, b in ranges:
            if a == b:
                print(f"  moving_{a:04d}.jpg")
            else:
                print(f"  moving_{a:04d}.jpg ... moving_{b:04d}.jpg")


if __name__ == "__main__":
    main()
