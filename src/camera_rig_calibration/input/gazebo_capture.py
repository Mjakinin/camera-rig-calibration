#!/usr/bin/env python3
"""Capture the moving camera along a declared bus-world route."""

import argparse
import csv
import hashlib
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
            last_output = (proc.stdout or "") + (proc.stderr or "")

            if proc.returncode == 0 and "data: true" in last_output:
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
        arr = data.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3).copy()
        if enc == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    if enc in ["rgba8", "bgra8"]:
        arr = data.reshape(h, msg.step)[:, : w * 4].reshape(h, w, 4).copy()
        if enc == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if enc in ["mono8", "8uc1"]:
        arr = data.reshape(h, msg.step)[:, :w].reshape(h, w).copy()
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


class ImageGrabber(Node):
    def __init__(self, topic):
        super().__init__("moving_camera_route_capture")
        self.last_msg = None
        self.counter = 0
        self.sub = self.create_subscription(
            Image,
            topic,
            self.cb,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"subscribed: {topic}")

    def cb(self, msg):
        self.last_msg = msg
        self.counter += 1


def wait_for_image(node, min_counter, timeout):
    start = time.time()
    while rclpy.ok() and time.time() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.last_msg is not None and node.counter > min_counter:
            return node.last_msg, node.counter
    # Never relabel the previous message as a fresh frame after a timeout.
    return None, node.counter


def spin_for(node, duration):
    deadline = time.monotonic() + max(0.0, duration)
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(
            node,
            timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())),
        )


def wait_for_distinct_image(node, min_counter, timeout, seen_hashes):
    """Wait for a newly delivered image whose pixels were not captured before."""
    deadline = time.monotonic() + timeout
    counter = min_counter
    while rclpy.ok() and time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        msg, counter = wait_for_image(node, counter, remaining)
        if msg is None:
            return None, counter, None
        digest = hashlib.sha256(memoryview(msg.data)).hexdigest()
        if digest not in seen_hashes:
            return msg, counter, digest
    return None, counter, None


def capture_frame_at_pose(
    node,
    *,
    world,
    name,
    pose,
    settle,
    post_pose_skip,
    timeout,
    captured_hashes,
    frame_idx,
    retries,
):
    """Capture one distinct frame, retrying the same commanded pose if needed."""
    last_counter = node.counter

    for attempt in range(1, retries + 1):
        ok = set_pose(world, name, pose)

        # Keep consuming the subscription during settling. Otherwise DDS
        # backlog from the previous pose can be mistaken for the new pose.
        spin_for(node, settle)
        counter = node.counter
        fresh = True

        for _skip in range(post_pose_skip):
            msg, counter = wait_for_image(node, counter, timeout)
            if msg is None:
                fresh = False
                break

        msg = None
        image_sha256 = None
        if fresh:
            msg, counter, image_sha256 = wait_for_distinct_image(
                node,
                counter,
                timeout,
                captured_hashes,
            )

        last_counter = counter
        if msg is not None:
            return msg, counter, image_sha256, ok, attempt

        if attempt < retries:
            print(
                f"[WARN] frame {frame_idx:04d}: no fresh image after pose "
                f"(capture attempt {attempt}/{retries}); retrying the same pose",
                flush=True,
            )

    raise RuntimeError(
        f"frame {frame_idx:04d}: no fresh image after {retries} capture attempts "
        f"at the same commanded pose (last message counter={last_counter}). "
        "Stale data was not written."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--route",
        default=(
            "src/calib_lab/bus_real_data/config/"
            "moving_camera_route2_interpolated_final.json"
        ),
    )
    ap.add_argument(
        "--out",
        default="workspace/manual_gazebo_capture",
        help=(
            "Temporary capture directory. The rigcal pipeline supplies its "
            "own transaction path."
        ),
    )
    ap.add_argument("--world", default="bus_real_data_camera_layout")
    ap.add_argument("--name", default="moving_calib_camera")
    ap.add_argument("--topic", default="/bus_real_data/moving_calib_camera/image")
    ap.add_argument("--settle", type=float, default=0.80)
    ap.add_argument("--post-pose-skip", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument(
        "--frame-retries",
        type=int,
        default=4,
        help=(
            "Number of full same-pose capture attempts before failing a route frame. "
            "Stale images are never accepted."
        ),
    )
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.frame_retries < 1:
        raise ValueError("--frame-retries must be at least 1")

    out_dir = Path(args.out)
    img_dir = out_dir / "images"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    img_dir.mkdir(parents=True, exist_ok=True)

    route_data = json.loads(Path(args.route).read_text())
    frames = route_data["frames"]

    rclpy.init()
    node = ImageGrabber(args.topic)

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
    captured_hashes = set()
    progress_interval = max(1, len(frames) // 20)

    try:
        for r in frames:
            frame_idx = int(r["frame"])
            pose = [
                r["x"],
                r["y"],
                r["z"],
                r["roll"],
                r["pitch"],
                r["yaw"],
            ]

            msg, cnt, image_sha256, ok, capture_attempt = capture_frame_at_pose(
                node,
                world=args.world,
                name=args.name,
                pose=pose,
                settle=args.settle,
                post_pose_skip=args.post_pose_skip,
                timeout=args.timeout,
                captured_hashes=captured_hashes,
                frame_idx=frame_idx,
                retries=args.frame_retries,
            )

            bgr = image_msg_to_bgr(msg)
            captured_hashes.add(image_sha256)
            img_path = img_dir / f"frame_{frame_idx:04d}.png"
            if not cv2.imwrite(str(img_path), bgr):
                raise RuntimeError(f"Could not write captured frame: {img_path}")

            rows.append(
                {
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
                    "message_counter": cnt,
                    "image_sha256": image_sha256,
                    "capture_attempt": capture_attempt,
                }
            )

            print(
                f"[CAPTURED] frame_{frame_idx:04d}.png "
                f"x={r['x']:.2f} y={r['y']:.2f} z={r['z']:.2f} "
                f"yaw={r['yaw']:.2f} attempt={capture_attempt}/{args.frame_retries}"
            )

            captured_count = len(rows)
            if (
                captured_count == 1
                or captured_count % progress_interval == 0
                or captured_count == len(frames)
            ):
                print(
                    "RIGCAL_PROGRESS "
                    f"current={captured_count} total={len(frames)} "
                    "unit=frames label=Gazebo capture",
                    flush=True,
                )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    csv_path = out_dir / "route_commanded.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "frame",
            "segment",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "image",
            "set_pose_ok",
            "message_counter",
            "image_sha256",
            "capture_attempt",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    readme = out_dir / "README.txt"
    readme.write_text(
        "Moving camera route capture\n"
        "===========================\n\n"
        f"Route: {args.route}\n"
        f"Topic: {args.topic}\n"
        f"Frames captured: {len(rows)}\n"
        f"Same-pose capture retries: {args.frame_retries}\n\n"
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
