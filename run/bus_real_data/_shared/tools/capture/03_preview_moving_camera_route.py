#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
import time
from pathlib import Path


def rpy_to_quat(roll, pitch, yaw):
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


def set_pose(world, name, pose):
    x, y, z, roll, pitch, yaw = pose
    qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)

    req = (
        f'name: "{name}" '
        f'position {{x: {x} y: {y} z: {z}}} '
        f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}'
    )

    cmd = [
        "ign", "service",
        "-s", f"/world/{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", req,
    ]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=3)
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and "data: true" in out
    if not ok:
        raise RuntimeError(
            f"set_pose failed for world={world!r}, name={name!r}, pose={pose}\n"
            f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json")
    ap.add_argument("--world", default="bus_real_data_camera_layout")
    ap.add_argument("--name", default="moving_calib_camera")
    ap.add_argument("--sleep", type=float, default=0.08)
    args = ap.parse_args()

    data = json.loads(Path(args.route).read_text())
    frames = data["frames"]

    print(f"[INFO] replaying {len(frames)} frames")
    print("[INFO] watch /bus_real_data/moving_calib_camera/image in rqt_image_view")

    for r in frames:
        pose = [r["x"], r["y"], r["z"], r["roll"], r["pitch"], r["yaw"]]
        set_pose(args.world, args.name, pose)
        print(
            f"frame {r['frame']:04d}: "
            f"x={r['x']:.2f} y={r['y']:.2f} z={r['z']:.2f} "
            f"pitch={r['pitch']:.2f} yaw={r['yaw']:.2f}"
        )
        time.sleep(args.sleep)

    print("[OK] preview finished")


if __name__ == "__main__":
    main()
