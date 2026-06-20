#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from ap02_common import (
    AP02_ROOT,
    STATIC_CAMERAS,
    MOVING_CAMERA,
    IMAGE_TOPICS,
    CAMERA_INFO_TOPICS,
    ensure_dir,
    write_json,
    camera_info_to_dict,
)


class AP02CaptureNode(Node):
    def __init__(self, out_root, moving_frames, moving_dt):
        super().__init__("ap02_capture_dataset")

        self.out_root = Path(out_root)
        self.moving_frames = moving_frames
        self.moving_dt = moving_dt

        self.bridge = CvBridge()
        self.latest_images = {}
        self.latest_infos = {}

        all_cams = STATIC_CAMERAS + [MOVING_CAMERA]

        for cam in all_cams:
            self.create_subscription(
                Image,
                IMAGE_TOPICS[cam],
                lambda msg, cam=cam: self.image_cb(cam, msg),
                10,
            )
            self.create_subscription(
                CameraInfo,
                CAMERA_INFO_TOPICS[cam],
                lambda msg, cam=cam: self.info_cb(cam, msg),
                10,
            )

    def image_cb(self, cam, msg):
        self.latest_images[cam] = msg

    def info_cb(self, cam, msg):
        self.latest_infos[cam] = msg

    def wait_for_all_static(self, timeout_s=10.0):
        start = time.time()
        while time.time() - start < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(cam in self.latest_images for cam in STATIC_CAMERAS) and all(cam in self.latest_infos for cam in STATIC_CAMERAS):
                return True
        return False

    def wait_for_moving(self, timeout_s=10.0):
        start = time.time()
        while time.time() - start < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
            if MOVING_CAMERA in self.latest_images and MOVING_CAMERA in self.latest_infos:
                return True
        return False

    def save_image_and_info(self, cam, image_path, info_path):
        msg = self.latest_images[cam]
        info = self.latest_infos[cam]

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ensure_dir(image_path.parent)
        cv2.imwrite(str(image_path), img)

        write_json(info_path, camera_info_to_dict(info))

    def capture_static(self):
        static_dir = ensure_dir(self.out_root / "static")
        info_dir = ensure_dir(self.out_root / "camera_info")

        ok = self.wait_for_all_static()
        if not ok:
            missing_images = [c for c in STATIC_CAMERAS if c not in self.latest_images]
            missing_infos = [c for c in STATIC_CAMERAS if c not in self.latest_infos]
            raise RuntimeError(f"Timeout waiting for static cameras. Missing images={missing_images}, infos={missing_infos}")

        for cam in STATIC_CAMERAS:
            self.save_image_and_info(
                cam,
                static_dir / f"{cam}.png",
                info_dir / f"{cam}.json",
            )
            self.get_logger().info(f"saved static {cam}")

    def capture_moving(self):
        moving_dir = ensure_dir(self.out_root / "moving")
        info_dir = ensure_dir(self.out_root / "camera_info")

        ok = self.wait_for_moving()
        if not ok:
            raise RuntimeError("Timeout waiting for moving camera image/camera_info")

        self.save_image_and_info(
            MOVING_CAMERA,
            moving_dir / "frame_000000.png",
            info_dir / f"{MOVING_CAMERA}.json",
        )

        for i in range(self.moving_frames):
            end = time.time() + self.moving_dt
            while time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.02)

            frame_path = moving_dir / f"frame_{i:06d}.png"
            self.save_image_and_info(MOVING_CAMERA, frame_path, info_dir / f"{MOVING_CAMERA}.json")

            if i % 25 == 0:
                self.get_logger().info(f"saved moving frame {i}/{self.moving_frames}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--moving-frames", type=int, default=204)
    ap.add_argument("--moving-dt", type=float, default=0.10)
    ap.add_argument("--static-only", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and any(out_path.rglob("*.png")) and not args.overwrite:
        raise RuntimeError(
            f"Raw dataset already exists and contains images: {out_path}\n"
            "Refusing to overwrite. Use --overwrite only if you intentionally want to recapture this dataset."
        )

    rclpy.init()
    node = AP02CaptureNode(args.out, args.moving_frames, args.moving_dt)

    try:
        node.capture_static()
        if not args.static_only:
            node.capture_moving()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("[OK] AP02 dataset captured:", args.out)


if __name__ == "__main__":
    main()
