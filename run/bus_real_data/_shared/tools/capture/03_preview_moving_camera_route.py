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


def set_pose(world, name, pose, retries=8):
    x, y, z, roll, pitch, yaw = pose
    qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)

    req = (
        f'name: "{name}" '
        f'position {{x: {x} y: {y} z: {z}}} '
        f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}'
    )

    cmd = [
        "ign",
        "service",
        "-s",
        f"/world/{world}/set_pose",
        "--reqtype",
        "ignition.msgs.Pose",
        "--reptype",
        "ignition.msgs.Boolean",
        "--timeout",
        "5000",
        "--req",
        req,
    ]

    last_output = ""
    last_returncode = None

    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )

            last_returncode = proc.returncode
            last_output = (
                (proc.stdout or "")
                + (proc.stderr or "")
            )

            if (
                proc.returncode == 0
                and "data: true" in last_output
            ):
                return True

        except subprocess.TimeoutExpired as exc:
            last_returncode = "subprocess-timeout"
            last_output = str(exc)

        if attempt < retries:
            delay = min(0.25 * attempt, 1.5)

            print(
                f"[WARN] set_pose attempt "
                f"{attempt}/{retries} failed; "
                f"retrying in {delay:.2f}s",
                flush=True,
            )

            time.sleep(delay)

    raise RuntimeError(
        "\n".join(
            [
                f"set_pose failed after {retries} attempts",
                f"world={world!r}",
                f"name={name!r}",
                f"pose={pose}",
                f"returncode={last_returncode}",
                f"output={last_output}",
            ]
        )
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json")
    ap.add_argument("--world", default="bus_real_data_camera_layout")
    ap.add_argument("--name", default="moving_calib_camera")
    ap.add_argument("--sleep", type=float, default=0.08)
    ap.add_argument("--start-frame", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(Path(args.route).read_text())
    frames = [
        frame
        for frame in data["frames"]
        if int(frame["frame"]) >= args.start_frame
    ]

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
