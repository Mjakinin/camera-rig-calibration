#!/usr/bin/env python3
"""Compatibility entry point for safe real static-image extraction.

The current rosbag has edge_0 and edge_5 device assignments swapped. This
entry point therefore delegates to the canonical image-only extractor, which
contains an explicit physical-camera/topic map. Static intrinsics are never
taken from this rosbag; install the trusted camera_info.zip intrinsics instead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate-times", type=int, default=15)
    parser.add_argument("--save-all-candidates", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).with_name("03_extract_static_images_only_from_mcap.py")
    command = [
        sys.executable,
        str(script),
        "--mcap",
        args.mcap,
        "--dataset",
        args.dataset,
    ]
    if args.save_all_candidates:
        command.append("--save-all-candidates")

    print("[INFO] safe canonical MCAP extraction")
    print("[INFO] /edge_0/ -> cam_edge_5 (back_right)")
    print("[INFO] /edge_5/ -> cam_edge_0 (center_left)")
    print("[INFO] static intrinsics are not read from the rosbag")
    subprocess.run(command, check=True)

    info_root = Path(args.dataset).resolve() / "raw_images/camera_info"
    missing = [
        str(info_root / f"{camera}.json")
        for camera in ("cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5")
        if not (info_root / f"{camera}.json").is_file()
    ]
    if missing:
        print()
        print("[NEXT] Install trusted static intrinsics from camera_info.zip:")
        for path in missing:
            print("  missing:", path)


if __name__ == "__main__":
    main()
