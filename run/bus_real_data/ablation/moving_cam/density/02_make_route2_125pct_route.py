#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


SOURCE = Path(
    "src/calib_lab/bus_real_data/config/"
    "moving_camera_route2_interpolated_final.json"
)

OUTPUT = Path(
    "src/calib_lab/bus_real_data/config/"
    "moving_camera_route2_density_125pct.json"
)

TARGET_FACTOR = 1.25


def interpolate_angle(first: float, second: float, alpha: float) -> float:
    """Shortest-path interpolation for angles in radians."""
    difference = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    return first + alpha * difference


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_frames = data["frames"]

    if len(source_frames) != 189:
        raise RuntimeError(
            f"Expected 189 Route-2 frames, found {len(source_frames)}"
        )

    target_count = round(
        (len(source_frames) - 1) * TARGET_FACTOR
    ) + 1

    if target_count != 236:
        raise RuntimeError(
            f"Expected target count 236, computed {target_count}"
        )

    output_frames = []

    for target_index in range(target_count):
        source_position = (
            target_index
            * (len(source_frames) - 1)
            / (target_count - 1)
        )

        left_index = int(math.floor(source_position))
        right_index = min(left_index + 1, len(source_frames) - 1)
        alpha = source_position - left_index

        left = source_frames[left_index]
        right = source_frames[right_index]

        row = dict(left)
        row["frame"] = target_index

        for key in ("x", "y", "z"):
            row[key] = (
                (1.0 - alpha) * float(left[key])
                + alpha * float(right[key])
            )

        for key in ("roll", "pitch", "yaw"):
            row[key] = interpolate_angle(
                float(left[key]),
                float(right[key]),
                alpha,
            )

        row["segment"] = (
            right.get("segment", left.get("segment", ""))
            if alpha >= 0.999999
            else left.get("segment", "")
        )

        output_frames.append(row)

    output = dict(data)
    output["name"] = "moving_camera_route2_density_125pct"
    output["description"] = (
        "Route 2 with 125 percent temporal-spatial pose sampling. "
        "Images must be newly rendered in Gazebo."
    )
    output["source_route"] = str(SOURCE)
    output["density_factor"] = TARGET_FACTOR
    output["num_frames"] = len(output_frames)
    output["frames"] = output_frames

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    positions = np.asarray([
        [row["x"], row["y"], row["z"]]
        for row in output_frames
    ], dtype=np.float64)

    steps = np.linalg.norm(
        positions[1:] - positions[:-1],
        axis=1,
    )

    print("[OK] wrote:", OUTPUT)
    print("[OK] source frames:", len(source_frames))
    print("[OK] target frames:", len(output_frames))
    print("[OK] mean translation step [m]:", float(np.mean(steps)))
    print("[OK] maximum translation step [m]:", float(np.max(steps)))


if __name__ == "__main__":
    main()
