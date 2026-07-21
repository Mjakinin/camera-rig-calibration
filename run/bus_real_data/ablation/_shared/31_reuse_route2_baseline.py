#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path("results/bus_real_data/ablation/world/route/route2")
REQUIRED_DATASET_PATHS = (
    "raw_images",
    "aruco_observations",
    "metadata",
)


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def replace_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"Missing source directory: {source}")
    shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=link_or_copy)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def patch_run_status(path: Path, variant: str, source: Path) -> None:
    values: dict[str, str] = {}
    order: list[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in order:
                order.append(key)
            values[key] = value.strip()

    values["variant"] = variant
    values["BASELINE_REUSED"] = "1"
    values["BASELINE_SOURCE_VARIANT"] = "route2"
    values["BASELINE_SOURCE_FINAL_RESULTS"] = str(source / "FINAL_RESULTS")
    for key in (
        "variant",
        "BASELINE_REUSED",
        "BASELINE_SOURCE_VARIANT",
        "BASELINE_SOURCE_FINAL_RESULTS",
    ):
        if key not in order:
            order.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key}={values[key]}" for key in order if key in values) + "\n",
        encoding="utf-8",
    )


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Install the freshly computed nominal Route-2 dataset, and optionally its "
            "AP01/AP02/AP03 FINAL_RESULTS, as an exactly equivalent ablation baseline."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()

    for relative in REQUIRED_DATASET_PATHS:
        if not (source / relative).is_dir():
            raise RuntimeError(f"Route-2 baseline source is incomplete: {source / relative}")
    if not (source / "aruco_observations/shared_all_aruco_observations.csv").is_file():
        raise RuntimeError("Route-2 ArUco observations are missing")
    if not args.dataset_only and not (source / "FINAL_RESULTS/RUN_STATUS.txt").is_file():
        raise RuntimeError("Route-2 FINAL_RESULTS are missing")

    previous_metadata = load_json(target / "VARIANT_METADATA.json")

    for relative in REQUIRED_DATASET_PATHS:
        replace_tree(source / relative, target / relative)

    if not args.dataset_only:
        replace_tree(source / "FINAL_RESULTS", target / "FINAL_RESULTS")

    route_frame_count = len(list((source / "raw_images/moving").glob("frame_*.png")))
    payload = {
        **previous_metadata,
        "group": args.group,
        "variant": args.variant,
        "baseline_reused": True,
        "baseline_source_variant": "route2",
        "baseline_source": str(source),
        "route_frame_count": route_frame_count,
        "equivalence_contract": (
            "Exact nominal Route-2 raw images, CameraInfo, ArUco observations and "
            "method outputs are reused; AP01/AP02/AP03 are not rerun for this "
            "mathematically identical baseline variant."
        ),
        "methods_reused_without_rerun": ["AP01", "AP02", "AP03"],
    }

    metadata_text = json.dumps(payload, indent=2) + "\n"
    (target / "VARIANT_METADATA.json").write_text(metadata_text, encoding="utf-8")
    (target / "metadata").mkdir(parents=True, exist_ok=True)
    (target / "metadata/BASELINE_REUSE.json").write_text(
        metadata_text,
        encoding="utf-8",
    )

    if not args.dataset_only:
        (target / "FINAL_RESULTS/VARIANT_METADATA.json").write_text(
            metadata_text,
            encoding="utf-8",
        )
        patch_run_status(
            target / "FINAL_RESULTS/RUN_STATUS.txt",
            args.variant,
            source,
        )

    print(f"[OK] Route-2 baseline installed: {target}")
    print(f"     variant: {args.variant}")
    print(f"     group: {args.group}")
    print(f"     moving frames: {route_frame_count}")
    print(f"     raw files: {count_files(target / 'raw_images')}")
    print(f"     observation files: {count_files(target / 'aruco_observations')}")
    print(f"     method outputs reused: {not args.dataset_only}")


if __name__ == "__main__":
    main()
