#!/usr/bin/env python3

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


TOPIC = "/bus_real_data/moving_calib_camera/image"


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

def image_msg_to_bgr(msg):
    h = msg.height
    w = msg.width
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ["rgb8", "bgr8"]:
        arr = data.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3).copy()
        if enc == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    if enc in ["rgba8", "bgra8"]:
        arr = data.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4).copy()
        if enc == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if enc in ["mono8", "8uc1"]:
        arr = data.reshape(h, msg.step)[:, :w].reshape(h, w).copy()
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


class ImageGrabber(Node):
    def __init__(self):
        super().__init__("moving_camera_route_capture")
        self.last_msg = None
        self.counter = 0
        self.sub = self.create_subscription(
            Image,
            TOPIC,
            self.cb,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"subscribed: {TOPIC}")

    def cb(self, msg):
        self.last_msg = msg
        self.counter += 1


def wait_for_image(node, min_counter, timeout):
    start = time.time()
    while rclpy.ok() and time.time() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.last_msg is not None and node.counter > min_counter:
            return node.last_msg, node.counter
    return node.last_msg, node.counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json")
    ap.add_argument("--out", default="results/bus_real_data/01_marker_direct_relay_multimarker_multichain/03_moving_camera_sequence")
    ap.add_argument("--world", default="bus_real_data_camera_layout")
    ap.add_argument("--name", default="moving_calib_camera")
    ap.add_argument("--settle", type=float, default=0.80)
    ap.add_argument("--post-pose-skip", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    img_dir.mkdir(parents=True, exist_ok=True)

    route_data = json.loads(Path(args.route).read_text())
    frames = route_data["frames"]

    rclpy.init()
    node = ImageGrabber()

    startup_timeout = max(args.timeout, 30.0)
    print(
        f"[INFO] waiting for first image "
        f"(startup timeout={startup_timeout:.1f}s)..."
    )
    msg, cnt = wait_for_image(node, -1, startup_timeout)
    if msg is None:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("No image received. Is the bridge running?")

    # Important: move once to the first commanded route pose before recording frame_0000.
    # This flushes stale images from the default / previous moving-camera pose.
    if frames:
        first = frames[0]
        first_pose = [
            first["x"],
            first["y"],
            first["z"],
            first["roll"],
            first["pitch"],
            first["yaw"],
        ]
        print("[INFO] pre-positioning at first route frame and flushing stale images...")
        ok_first = set_pose(args.world, args.name, first_pose)
        print(f"[INFO] first route pose set_pose_ok={ok_first}")
        time.sleep(max(args.settle, 1.0))

        cnt = node.counter
        flush_n = max(args.post_pose_skip * 2, 10)
        for _ in range(flush_n):
            msg, cnt = wait_for_image(node, cnt, args.timeout)
            if msg is None:
                break
        print(f"[INFO] flushed up to counter={cnt}")

    rows = []

    for r in frames:
        frame_idx = int(r["frame"])
        pose = [r["x"], r["y"], r["z"], r["roll"], r["pitch"], r["yaw"]]

        ok = set_pose(args.world, args.name, pose)
        before = node.counter
        time.sleep(args.settle)

        msg = None
        cnt = before

        for _skip in range(args.post_pose_skip):
            msg, cnt = wait_for_image(node, cnt, args.timeout)
            if msg is None:
                break

        msg, cnt = wait_for_image(node, cnt, args.timeout)

        if msg is None:
            print(f"[WARN] frame {frame_idx:04d}: no image")
            continue

        bgr = image_msg_to_bgr(msg)
        img_path = img_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(img_path), bgr)

        rows.append({
            "frame": frame_idx,
            "segment": r["segment"],
            "x": r["x"],
            "y": r["y"],
            "z": r["z"],
            "roll": r["roll"],
            "pitch": r["pitch"],
            "yaw": r["yaw"],
            "image": str(img_path),
            "set_pose_ok": ok,
        })

        print(
            f"[CAPTURED] frame_{frame_idx:04d}.png "
            f"x={r['x']:.2f} y={r['y']:.2f} z={r['z']:.2f} yaw={r['yaw']:.2f}"
        )

    node.destroy_node()
    rclpy.shutdown()

    csv_path = out_dir / "route_commanded.csv"
    with csv_path.open("w", newline="") as f:
        fields = ["frame", "segment", "x", "y", "z", "roll", "pitch", "yaw", "image", "set_pose_ok"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    readme = out_dir / "README.txt"
    readme.write_text(
        "Moving camera route capture\n"
        "===========================\n\n"
        f"Route: {args.route}\n"
        f"Topic: {TOPIC}\n"
        f"Frames captured: {len(rows)}\n\n"
        "Files:\n"
        "- images/: captured moving camera frames\n"
        "- route_commanded.csv: commanded pose per frame\n"
    )

    print()
    print("[OK] wrote:", out_dir)
    print("[OK] images:", img_dir)
    print("[OK] route csv:", csv_path)


if __name__ == "__main__":
    main()
