#!/usr/bin/env python3

import argparse
import json
import math
import re
import subprocess
from pathlib import Path


def quat_to_rpy(qx, qy, qz, qw):
    # roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def extract_block_after_name(text, name):
    idx = text.rfind(f'name: "{name}"')
    if idx < 0:
        return None
    return text[idx:idx + 2500]


def extract_float(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1))


def parse_pose(text, name):
    block = extract_block_after_name(text, name)
    if block is None:
        return None

    px = extract_float(r"position\s*\{[^}]*\bx:\s*([-+0-9.eE]+)", block)
    py = extract_float(r"position\s*\{[^}]*\by:\s*([-+0-9.eE]+)", block)
    pz = extract_float(r"position\s*\{[^}]*\bz:\s*([-+0-9.eE]+)", block)

    qx = extract_float(r"orientation\s*\{[^}]*\bx:\s*([-+0-9.eE]+)", block)
    qy = extract_float(r"orientation\s*\{[^}]*\by:\s*([-+0-9.eE]+)", block)
    qz = extract_float(r"orientation\s*\{[^}]*\bz:\s*([-+0-9.eE]+)", block)
    qw = extract_float(r"orientation\s*\{[^}]*\bw:\s*([-+0-9.eE]+)", block)

    if None in [px, py, pz, qx, qy, qz, qw]:
        return None

    roll, pitch, yaw = quat_to_rpy(qx, qy, qz, qw)
    return px, py, pz, roll, pitch, yaw


def run_ign_pose_dump(world, seconds):
    topic = f"/world/{world}/pose/info"
    cmd = ["timeout", str(seconds), "ign", "topic", "-e", "-t", topic]

    print("[INFO] reading Gazebo pose topic:")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)

    # timeout returns 124, but stdout is still useful.
    text = proc.stdout + "\n" + proc.stderr

    if not proc.stdout.strip():
        raise RuntimeError(
            f"No pose data received from {topic}. "
            "Check that Gazebo is still running and world name is correct."
        )

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="bus_real_data_camera_layout")
    ap.add_argument("--placements", default="src/calib_lab/bus_real_data/config/a4_marker_placements.json")
    ap.add_argument("--max-id", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()

    placements_path = Path(args.placements)
    data = json.loads(placements_path.read_text())

    by_name = {item["name"]: item for item in data}

    text = run_ign_pose_dump(args.world, args.seconds)

    saved = []
    missing = []

    for marker_id in range(args.max_id + 1):
        name = f"marker_{marker_id:03d}"
        pose = parse_pose(text, name)

        if pose is None:
            missing.append(name)
            continue

        pose_str = (
            f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} "
            f"{pose[3]:.8f} {pose[4]:.8f} {pose[5]:.8f}"
        )

        if name not in by_name:
            item = {
                "name": name,
                "model": f"a4_aruco_marker_{marker_id:03d}",
                "pose": pose_str,
            }
            data.append(item)
            by_name[name] = item
        else:
            by_name[name]["pose"] = pose_str
            by_name[name]["model"] = f"a4_aruco_marker_{marker_id:03d}"

        saved.append((name, pose_str))

    placements_path.write_text(json.dumps(data, indent=2) + "\n")

    print()
    print("[OK] saved live marker poses to:")
    print(placements_path)
    print()
    print("[SAVED]")
    for name, pose_str in saved:
        print(f"{name:12s} {pose_str}")

    if missing:
        print()
        print("[WARN] these markers were not found in the running Gazebo world:")
        for name in missing:
            print(name)

    print()
    print("[NEXT] rebuild marker world file:")
    print("python3 src/calib_lab/bus_real_data/scripts/03_build_world_with_a4_markers.py")


if __name__ == "__main__":
    main()
