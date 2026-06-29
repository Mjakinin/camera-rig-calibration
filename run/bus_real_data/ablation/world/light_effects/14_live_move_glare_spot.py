#!/usr/bin/env python3
import math
import subprocess
import sys

WORLD = "bus_real_data_camera_layout"
NAME = "temp_window_glare_spot"

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
    if len(sys.argv) != 7:
        print("usage: python3 14_live_move_glare_spot.py x y z roll pitch yaw")
        print("example: python3 14_live_move_glare_spot.py 5.8 1.4 2.20 0 0 3.14")
        sys.exit(1)

    x, y, z, roll, pitch, yaw = map(float, sys.argv[1:])
    qx, qy, qz, qw = quat_from_euler(roll, pitch, yaw)

    req = (
        f'name: "{NAME}", '
        f'position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}'
    )

    cmd = [
        "ign", "service",
        "-s", f"/world/{WORLD}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", req,
    ]

    print("[INFO]", " ".join(cmd))
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
