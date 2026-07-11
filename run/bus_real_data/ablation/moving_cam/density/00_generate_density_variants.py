#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

VARIANTS = {
    "density_stride_1_100pct": 1,
    "density_stride_2_50pct": 2,
    "density_stride_4_25pct": 4,
    "density_stride_8_12p5pct": 8,
}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def copy_tree_files(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def filtered_route(source: Path, destination: Path, keep: set[int], raw: Path) -> None:
    if not source.is_file():
        return
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        try:
            number = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if number not in keep:
            continue
        updated = dict(row)
        if "image" in updated:
            updated["image"] = str(raw / "moving" / f"frame_{number:04d}.png")
        selected.append(updated)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not selected:
        raise RuntimeError(f"No route rows retained from {source}")
    fields = list(selected[0])
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create controlled moving-camera frame-density variants from the "
            "canonical Route-2 shared input. Only temporal frame density changes."
        )
    )
    parser.add_argument(
        "--source",
        default=(
            "results/bus_real_data/00_shared_baseline/"
            "bus_real_data_ref_marker_v1"
        ),
    )
    parser.add_argument(
        "--out",
        default="results/bus_real_data/ablation/moving_cam/density",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    source_raw = source / "raw_images"
    moving = sorted((source_raw / "moving").glob("frame_*.png"))
    if not moving:
        raise RuntimeError(f"No source moving images in {source_raw / 'moving'}")

    all_numbers = [frame_number(path) for path in moving]
    if all_numbers != list(range(all_numbers[0], all_numbers[0] + len(all_numbers))):
        raise RuntimeError("Canonical source moving frames are not contiguous")
    if all_numbers[0] != 0:
        raise RuntimeError("Canonical source must start at frame 0")

    route_sources = [
        source / "metadata/route_commanded.csv",
        source_raw / "ap1_metadata/route_commanded.csv",
    ]

    for variant, stride in VARIANTS.items():
        root = out_root / variant
        shutil.rmtree(root, ignore_errors=True)
        raw = root / "raw_images"
        metadata = root / "metadata"
        (raw / "moving").mkdir(parents=True)
        metadata.mkdir(parents=True)

        copy_tree_files(source_raw / "static", raw / "static")
        copy_tree_files(source_raw / "camera_info", raw / "camera_info")
        copy_tree_files(source_raw / "static_multi", raw / "static_multi")

        selected = moving[::stride]
        if selected[-1] != moving[-1]:
            selected.append(moving[-1])
        keep = {frame_number(path) for path in selected}

        for path in selected:
            link_or_copy(path, raw / "moving" / path.name)

        filtered_route(
            route_sources[0],
            metadata / "route_commanded.csv",
            keep,
            raw,
        )
        filtered_route(
            route_sources[1],
            raw / "ap1_metadata/route_commanded.csv",
            keep,
            raw,
        )

        fraction = len(selected) / len(moving)
        payload = {
            "group": "moving_cam/density",
            "variant": variant,
            "parameter": "moving camera temporal frame density",
            "source_dataset": str(source),
            "source_frame_count": len(moving),
            "selected_frame_count": len(selected),
            "stride": stride,
            "retained_fraction": fraction,
            "first_frame": frame_number(selected[0]),
            "last_frame": frame_number(selected[-1]),
            "endpoint_forced": selected[-1] == moving[-1],
            "static_images_unchanged": True,
            "camera_info_unchanged": True,
            "route_geometry_unchanged": True,
            "moving_images_resized": False,
            "selection_rule": (
                f"retain every {stride}th source frame and retain the final "
                "endpoint frame when necessary"
            ),
            "changed_factor_only": "temporal sampling density",
        }
        (root / "VARIANT_METADATA.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        (metadata / "density_variant.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[OK] {variant}: {len(selected)}/{len(moving)} frames "
            f"({100.0 * fraction:.2f}%), stride={stride}"
        )


if __name__ == "__main__":
    main()
