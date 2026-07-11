#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Repository root not found")


def main() -> None:
    root = repository_root()
    shared = (
        root
        / "results/real_vehicle_data/real_05x_4k_3hz_v1/"
        "00_shared_input"
    )
    raw = shared / "raw_images"
    observations = shared / "aruco_observations"

    required = [
        raw / "moving",
        raw / "static",
        raw / "camera_info",
        observations / "shared_all_aruco_observations.csv",
        observations / "shared_moving_aruco_observations.csv",
        observations / "shared_static_aruco_observations.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "[ERROR] incomplete canonical real shared input:\n- "
            + "\n- ".join(missing)
        )

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_count = sum(
        1
        for path in raw.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if image_count <= 0:
        raise SystemExit("[ERROR] canonical real shared input has no images")

    print("[OK] canonical real shared input")
    print("     root:", shared)
    print("     raw images:", raw)
    print("     ArUco observations:", observations)
    print("     image count:", image_count)


if __name__ == "__main__":
    main()
