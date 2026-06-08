#!/usr/bin/env python3

import argparse
import csv
import subprocess
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageGrabber(Node):
    def __init__(self, topic):
        super().__init__("moving_camera_route_capture")
        self.bridge = CvBridge()
        self.latest = None
        self.create_subscription(Image, topic, self.callback, 10)

    def callback(self, msg):
        self.latest = msg

    def grab(self, timeout_sec=3.0):
        self.latest = None
        deadline = time.time() + timeout_sec

        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest is not None:
                return self.bridge.imgmsg_to_cv2(self.latest, desired_encoding="bgr8")

        raise RuntimeError("Timed out waiting for image.")


def set_pose(set_pose_script, entity_name, pose_values):
    cmd = [
        "python3",
        str(set_pose_script),
        "--name",
        entity_name,
        "--pose",
        *[str(v) for v in pose_values],
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route_csv", default="src/calib_lab/beintelli_bus_model/config/colmap/moving_camera_route_poc.csv")
    parser.add_argument("--dataset_dir", default="results/beintelli_bus_model/colmap/moving_route_poc")
    parser.add_argument("--topic", default="/moving_calib_camera/image")
    parser.add_argument("--settle_sec", type=float, default=0.35)
    parser.add_argument("--timeout_sec", type=float, default=3.0)
    parser.add_argument("--set_pose_script", default="src/calib_lab/beintelli_bus_model/scripts/tools/live_set_entity_pose.py")
    args = parser.parse_args()

    route_csv = Path(args.route_csv)
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    gt_out = dataset_dir / "route_gt.csv"

    with route_csv.open() as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] Loaded {len(rows)} route keyframes.")
    print(f"[INFO] Dataset: {dataset_dir}")
    print(f"[INFO] Images:  {images_dir}")

    rclpy.init()
    node = ImageGrabber(args.topic)

    written_rows = []

    try:
        for i, row in enumerate(rows):
            pose = [
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["roll"]),
                float(row["pitch"]),
                float(row["yaw"]),
            ]

            image_name = row["image_name"]
            out_path = images_dir / image_name

            print("")
            print("=" * 80)
            print(f"[INFO] {i+1}/{len(rows)} {image_name}")
            print(f"[INFO] pose: {' '.join(str(v) for v in pose)}")
            print("=" * 80)

            set_pose(Path(args.set_pose_script), "moving_calib_camera", pose)
            time.sleep(args.settle_sec)

            image = node.grab(timeout_sec=args.timeout_sec)
            ok = cv2.imwrite(str(out_path), image)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_path}")

            written = dict(row)
            written["image_path"] = str(out_path)
            written_rows.append(written)
            print(f"[OK] wrote {out_path}")

    finally:
        node.destroy_node()
        rclpy.shutdown()

    with gt_out.open("w", newline="") as f:
        fieldnames = ["image_name", "x", "y", "z", "roll", "pitch", "yaw", "tag", "image_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written_rows)

    # Store camera intrinsics used for COLMAP.
    # For current 640x480, 90deg horizontal FOV: fx=fy=320, cx=320, cy=240.
    (dataset_dir / "camera_intrinsics_used.txt").write_text(
        "camera_model: PINHOLE\n"
        "width: 640\n"
        "height: 480\n"
        "fx: 320\n"
        "fy: 320\n"
        "cx: 320\n"
        "cy: 240\n"
        "params_for_colmap: 320,320,320,240\n"
    )

    print("")
    print(f"[OK] Capture complete.")
    print(f"[OK] GT route written to: {gt_out}")
    print(f"[OK] Intrinsics written to: {dataset_dir / 'camera_intrinsics_used.txt'}")


if __name__ == "__main__":
    main()
