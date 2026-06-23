#!/usr/bin/env python3
from pathlib import Path
import argparse

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageCapture(Node):
    def __init__(self, topic, output_dir, max_images, every):
        super().__init__("image_capture_node")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_images = max_images
        self.every = every
        self.received = 0
        self.saved = 0
        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(Image, topic, self.callback, qos)

        self.get_logger().info(f"Listening to {topic}")
        self.get_logger().info(f"Saving images to {self.output_dir}")
        self.get_logger().info(f"Saving max {self.max_images} images, every {self.every} frames")

    def callback(self, msg):
        self.received += 1

        if self.received % self.every != 0:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        filename = self.output_dir / f"frame_{self.saved:06d}.png"
        cv2.imwrite(str(filename), img)

        self.saved += 1
        self.get_logger().info(f"Saved {filename}")

        if self.saved >= self.max_images:
            self.get_logger().info("Done.")
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/bus_real_data/moving_calib_camera/image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=208)
    parser.add_argument("--every", type=int, default=3)
    args = parser.parse_args()

    rclpy.init()
    node = ImageCapture(args.topic, args.output, args.max_images, args.every)
    rclpy.spin(node)


if __name__ == "__main__":
    main()

