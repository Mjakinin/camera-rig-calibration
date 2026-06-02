#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CheckerboardLiveDetector(Node):
    def __init__(self):
        super().__init__("checkerboard_live_detector")

        self.declare_parameter("image_topic", "/camera_1/image")
        self.declare_parameter("corners_x", 9)
        self.declare_parameter("corners_y", 6)

        self.image_topic = self.get_parameter("image_topic").value
        self.pattern_size = (
            self.get_parameter("corners_x").value,
            self.get_parameter("corners_y").value,
        )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.success_count = 0

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(f"Listening on: {self.image_topic}")
        self.get_logger().info(f"Searching checkerboard inner corners: {self.pattern_size}")

    def image_callback(self, msg):
        self.frame_count += 1

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Newer, more robust OpenCV checkerboard detector
        found, corners = cv2.findChessboardCornersSB(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # Fallback to classic detector
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE
                    + cv2.CALIB_CB_FAST_CHECK,
            )

        if found:
            self.success_count += 1
            self.get_logger().info(
                f"FOUND checkerboard | frame={self.frame_count} "
                f"| corners={len(corners)} "
                f"| success={self.success_count}/{self.frame_count}"
            )
        else:
            if self.frame_count % 10 == 0:
                self.get_logger().warn(
                    f"not found | frame={self.frame_count} "
                    f"| success={self.success_count}/{self.frame_count}"
                )


def main():
    rclpy.init()
    node = CheckerboardLiveDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
