#!/usr/bin/env python3
"""Attach complete camera models to shared ArUco observation CSV files.

Older observation files only stored fx, fy, cx and cy. This post-processing
step adds image dimensions, ROS distortion_model and up to eight distortion
coefficients. It is deterministic and safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


CSV_NAMES = [
    "shared_static_aruco_observations.csv",
    "shared_moving_aruco_observations.csv",
    "shared_all_aruco_observations.csv",
]
EXTRA_FIELDS = [
    "distortion_model",
    "image_width",
    "image_height",
    *[f"d{i}" for i in range(8)],
]


def load_camera_info(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing camera_info JSON: {path}")
    data = json.loads(path.read_text())
    distortion = data.get("D", data.get("d"))
    if distortion is None and "distortion_coefficients" in data:
        value = data["distortion_coefficients"]
        distortion = value.get("data") if isinstance(value, dict) else value
    if distortion is None:
        distortion = data.get("distortion", [])
    values = [float(value) for value in distortion]
    values = (values + [0.0] * 8)[:8]
    return {
        "distortion_model": str(data.get("distortion_model", "plumb_bob")),
        "image_width": int(data.get("width", data.get("image_width", 0)) or 0),
        "image_height": int(data.get("height", data.get("image_height", 0)) or 0),
        **{f"d{i}": values[i] for i in range(8)},
    }


def write_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    os.replace(temp_name, path)


def process_csv(path: Path, camera_info_root: Path, cache: dict[str, dict]) -> int:
    if not path.is_file():
        raise RuntimeError(f"Missing observation CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])

    for row in rows:
        camera_name = str(row.get("camera_name", "")).strip()
        if not camera_name:
            raise RuntimeError(f"Observation without camera_name in {path}")
        if camera_name not in cache:
            cache[camera_name] = load_camera_info(
                camera_info_root / f"{camera_name}.json"
            )
        row.update(cache[camera_name])

    fields = original_fields + [field for field in EXTRA_FIELDS if field not in original_fields]
    write_atomic(path, rows, fields)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "results/bus_real_data/00_shared_baseline/"
            "bus_real_data_ref_marker_v1/raw_images"
        ),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(
            "results/bus_real_data/00_shared_baseline/"
            "bus_real_data_ref_marker_v1/aruco_observations"
        ),
    )
    args = parser.parse_args()

    camera_info_root = args.dataset / "camera_info"
    if not camera_info_root.is_dir():
        raise RuntimeError(f"Missing camera_info directory: {camera_info_root}")

    cache: dict[str, dict] = {}
    total = 0
    for name in CSV_NAMES:
        path = args.observations / name
        count = process_csv(path, camera_info_root, cache)
        total += count
        print(f"[OK] camera model attached: {path} ({count} rows)")

    print(f"[OK] attached camera models to {total} observation rows")
    print(f"[OK] physical cameras: {sorted(cache)}")


if __name__ == "__main__":
    main()
