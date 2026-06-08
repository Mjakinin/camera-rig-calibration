#!/usr/bin/env python3
# AUTO_IMPORT_COMMON_START
from pathlib import Path as _CalibLabPath
import sys as _CalibLabSys
for _p in _CalibLabPath(__file__).resolve().parents:
    if _p.name == "calib_lab":
        if str(_p) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_p))
        _common_scripts = _p / "common" / "scripts"
        if _common_scripts.exists() and str(_common_scripts) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_common_scripts))
        break
# AUTO_IMPORT_COMMON_END

import argparse
import math
import subprocess
import sys


def quat_from_euler(roll, pitch, yaw):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="bus_static_camera_test")
    parser.add_argument("--model", required=True, choices=["front_static_camera", "rear_static_camera"])
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--degrees", action="store_true", help="Interpret roll/pitch/yaw as degrees.")
    parser.add_argument("--timeout", type=int, default=2000)
    args = parser.parse_args()

    roll = args.roll
    pitch = args.pitch
    yaw = args.yaw

    if args.degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    qx, qy, qz, qw = quat_from_euler(roll, pitch, yaw)

    service = f"/world/{args.world}/set_pose"

    req = (
        f'name: "{args.model}" '
        f'position {{ x: {args.x} y: {args.y} z: {args.z} }} '
        f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}'
    )

    cmd = [
        "ign", "service",
        "-s", service,
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(args.timeout),
        "--req", req,
    ]

    print("[INFO] Setting pose live:")
    print(f"  world: {args.world}")
    print(f"  model: {args.model}")
    print(f"  xyz:   {args.x:.3f}, {args.y:.3f}, {args.z:.3f}")
    print(f"  rpy:   {args.roll:.3f}, {args.pitch:.3f}, {args.yaw:.3f}" + (" deg" if args.degrees else " rad"))
    print(f"  service: {service}")

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
