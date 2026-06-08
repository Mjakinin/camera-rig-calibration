#!/usr/bin/env python3

import argparse
import math
import subprocess


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="bus_aruco_board_visibility_test")
    parser.add_argument("--name", required=True)
    parser.add_argument("--pose", nargs=6, type=float, required=True, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    parser.add_argument("--timeout", type=int, default=1000)
    args = parser.parse_args()

    x, y, z, roll, pitch, yaw = args.pose
    qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)

    req = (
        f'name: "{args.name}" '
        f'position {{ x: {x} y: {y} z: {z} }} '
        f'orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}'
    )

    service = f"/world/{args.world}/set_pose"

    cmd = [
        "ign", "service",
        "-s", service,
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(args.timeout),
        "--req", req,
    ]

    print("[INFO] Calling:", service)
    print("[INFO] Entity:", args.name)
    print("[INFO] Pose:", x, y, z, roll, pitch, yaw)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
