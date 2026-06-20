#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
from pathlib import Path


def quat_from_rpy(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def update_placements_json(path, name, pose_string):
    path = Path(path)
    data = json.loads(path.read_text())

    found = False
    for item in data:
        if item["name"] == name:
            item["pose"] = pose_string
            found = True
            break

    if not found:
        raise RuntimeError(f"Marker name not found in {path}: {name}")

    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[OK] saved pose to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Gazebo entity name, e.g. marker_000_door_area")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("z", type=float)
    ap.add_argument("roll", type=float)
    ap.add_argument("pitch", type=float)
    ap.add_argument("yaw", type=float)
    ap.add_argument("--world", default="bus_real_data_camera_layout")
    ap.add_argument("--save", action="store_true", help="Also save pose into a4_marker_placements.json")
    ap.add_argument(
        "--placements",
        default="src/calib_lab/bus_real_data/config/a4_marker_placements.json",
    )
    args = ap.parse_args()

    qx, qy, qz, qw = quat_from_rpy(args.roll, args.pitch, args.yaw)

    req = (
        f'name: "{args.name}" '
        f'position {{x: {args.x} y: {args.y} z: {args.z}}} '
        f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}'
    )

    cmd = [
        "ign", "service",
        "-s", f"/world/{args.world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", req,
    ]

    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    pose_string = f"{args.x:.6f} {args.y:.6f} {args.z:.6f} {args.roll:.8f} {args.pitch:.8f} {args.yaw:.8f}"

    if args.save:
        update_placements_json(args.placements, args.name, pose_string)
        print("[NEXT] regenerate world:")
        print("python3 src/calib_lab/bus_real_data/scripts/03_build_world_with_a4_markers.py")

    print("[OK] live pose set:")
    print(args.name, pose_string)


if __name__ == "__main__":
    main()
