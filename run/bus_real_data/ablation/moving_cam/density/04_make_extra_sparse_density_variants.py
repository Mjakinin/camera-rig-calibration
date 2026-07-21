#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path


SOURCE = Path(
    "results/bus_real_data/ablation/world/route/route2"
)

OUT_ROOT = Path(
    "results/bus_real_data/ablation/moving_cam/density"
)

VARIANTS = {
    "density_stride_8_offset4": {
        "stride": 8,
        "offset": 4,
    },
    "density_stride_16_6p25pct": {
        "stride": 16,
        "offset": 0,
    },
}


def frame_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def observation_frame(row: dict[str, str]) -> int | None:
    try:
        return int(float(row.get("frame_id", "")))
    except (TypeError, ValueError):
        pass

    observer_id = str(row.get("observer_id", ""))

    try:
        return int(observer_id.rsplit("_", 1)[-1])
    except ValueError:
        return None


def copy_optional_tree(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
        )


def main() -> None:
    source_raw = SOURCE / "raw_images"
    source_obs = SOURCE / "aruco_observations"

    moving = sorted(
        (source_raw / "moving").glob("frame_*.png"),
        key=frame_number,
    )

    if len(moving) != 189:
        raise RuntimeError(
            f"Expected 189 Route-2 images, found {len(moving)}"
        )

    for name, config in VARIANTS.items():
        stride = int(config["stride"])
        offset = int(config["offset"])

        root = OUT_ROOT / name

        if root.exists():
            shutil.rmtree(root)

        raw = root / "raw_images"
        observations = root / "aruco_observations"
        metadata = root / "metadata"

        copy_optional_tree(
            source_raw / "static",
            raw / "static",
        )
        copy_optional_tree(
            source_raw / "camera_info",
            raw / "camera_info",
        )
        copy_optional_tree(
            source_raw / "static_multi",
            raw / "static_multi",
        )

        selected = list(moving[offset::stride])

        # Keep identical Route-2 endpoints for fair route coverage.
        selected.extend([moving[0], moving[-1]])
        selected = sorted(set(selected), key=frame_number)

        keep = {
            frame_number(path)
            for path in selected
        }

        for source_image in selected:
            link_or_copy(
                source_image,
                raw / "moving" / source_image.name,
            )

        source_route = (
            SOURCE / "metadata" / "route_commanded.csv"
        )

        fields, route_rows = read_csv(source_route)
        filtered_route = []

        for row in route_rows:
            frame = int(row["frame"])

            if frame not in keep:
                continue

            updated = dict(row)
            updated["image"] = str(
                root
                / "raw_images"
                / "moving"
                / f"frame_{frame:04d}.png"
            )
            filtered_route.append(updated)

        write_csv(
            metadata / "route_commanded.csv",
            fields,
            filtered_route,
        )
        write_csv(
            raw / "ap1_metadata" / "route_commanded.csv",
            fields,
            filtered_route,
        )

        static_name = "shared_static_aruco_observations.csv"
        moving_name = "shared_moving_aruco_observations.csv"
        all_name = "shared_all_aruco_observations.csv"

        static_fields, static_rows = read_csv(
            source_obs / static_name
        )
        moving_fields, moving_rows = read_csv(
            source_obs / moving_name
        )
        all_fields, all_rows = read_csv(
            source_obs / all_name
        )

        filtered_moving = [
            row
            for row in moving_rows
            if observation_frame(row) in keep
        ]

        filtered_all = [
            row
            for row in all_rows
            if (
                row.get("observer_type") != "moving"
                or observation_frame(row) in keep
            )
        ]

        write_csv(
            observations / static_name,
            static_fields,
            static_rows,
        )
        write_csv(
            observations / moving_name,
            moving_fields,
            filtered_moving,
        )
        write_csv(
            observations / all_name,
            all_fields,
            filtered_all,
        )

        reference_file = source_obs / "REFERENCE_MARKER_ID.txt"
        if reference_file.is_file():
            shutil.copy2(
                reference_file,
                observations / reference_file.name,
            )

        payload = {
            "group": "moving_cam/density",
            "variant": name,
            "parameter": "moving camera temporal frame density",
            "source_dataset": str(SOURCE),
            "source_frame_count": len(moving),
            "selected_frame_count": len(selected),
            "retained_fraction": len(selected) / len(moving),
            "stride": stride,
            "offset": offset,
            "first_frame": frame_number(selected[0]),
            "last_frame": frame_number(selected[-1]),
            "route_endpoints_forced": True,
            "moving_observation_count": len(filtered_moving),
            "static_images_unchanged": True,
            "camera_info_unchanged": True,
            "route_geometry_unchanged": True,
            "images_newly_rendered": False,
            "image_interpolation_used": False,
            "changed_factor_only": (
                "temporal frame subset and sampling phase"
            ),
        }

        (root / "VARIANT_METADATA.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / "density_variant.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = [
            "Extended Route-2 frame-density variant",
            "=======================================",
            "",
            f"Variant: {name}",
            f"Stride: {stride}",
            f"Offset: {offset}",
            f"Frames: {len(selected)} / {len(moving)}",
            f"Moving ArUco observations: {len(filtered_moving)}",
        ]

        (
            observations
            / "SHARED_ARUCO_DETECTION_SUMMARY.txt"
        ).write_text(
            "\n".join(summary) + "\n",
            encoding="utf-8",
        )

        print(
            f"[OK] {name}: "
            f"{len(selected)}/{len(moving)} frames, "
            f"moving observations={len(filtered_moving)}"
        )


if __name__ == "__main__":
    main()
