#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class CheckerboardPoseLive(Node):
    def __init__(self):
        super().__init__("checkerboard_pose_live")

        self.declare_parameter("image_topic", "/camera_1/image")
        self.declare_parameter("camera_info_topic", "/camera_1/camera_info")
        self.declare_parameter("corners_x", 9)
        self.declare_parameter("corners_y", 6)
        self.declare_parameter("square_size", 0.12)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.corners_x = int(self.get_parameter("corners_x").value)
        self.corners_y = int(self.get_parameter("corners_y").value)
        self.square_size = float(self.get_parameter("square_size").value)

        self.pattern_size = (self.corners_x, self.corners_y)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        self.frame_count = 0
        self.success_count = 0

        self.object_points = self.create_object_points()

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.camera_info_topic}")
        self.get_logger().info(f"Pattern: {self.pattern_size}, square_size={self.square_size} m")

    def create_object_points(self):
        objp = np.zeros((self.corners_x * self.corners_y, 3), np.float32)
        grid = np.mgrid[0:self.corners_x, 0:self.corners_y].T.reshape(-1, 2)
        objp[:, :2] = grid * self.square_size
        return objp

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def detect_checkerboard(self, gray):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            return True, corners, "SB"

        found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            return True, corners, "classic"

        return False, None, "none"

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        self.frame_count += 1

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners, method = self.detect_checkerboard(gray)

        if not found:
            if self.frame_count % 10 == 0:
                self.get_logger().warn(f"checkerboard not found | frame={self.frame_count}")
            return

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            self.get_logger().warn("solvePnP failed")
            return

        self.success_count += 1

        t = tvec.flatten()
        distance = float(np.linalg.norm(t))

        self.get_logger().info(
            f"POSE FOUND | method={method} | frame={self.frame_count} "
            f"| t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m "
            f"| dist={distance:.3f} m "
            f"| success={self.success_count}/{self.frame_count}"
        )


def main():
    rclpy.init()
    node = CheckerboardPoseLive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
