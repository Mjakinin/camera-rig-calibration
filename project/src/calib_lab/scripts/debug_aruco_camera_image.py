#!/usr/bin/env python3

import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DebugArucoCameraImage(Node):
    def __init__(self):
        super().__init__("debug_aruco_camera_image")

        self.declare_parameter("image_topic", "/camera_1/image")
        self.declare_parameter("marker_id", 23)

        self.image_topic = self.get_parameter("image_topic").value
        self.marker_id = int(self.get_parameter("marker_id").value)

        self.bridge = CvBridge()
        self.done = False

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250
        )

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.dictionary,
                self.parameters
            )
        else:
            self.detector = None

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(f"Waiting for one image on: {self.image_topic}")

    def detect_markers(self, gray):
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.parameters,
            )

        return corners, ids, rejected

    def image_callback(self, msg):
        if self.done:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        cv2.imwrite("debug_aruco_input.png", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detect_markers(gray)

        debug = img.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)
            detected_ids = ids.flatten().tolist()
        else:
            detected_ids = []

        if rejected is not None:
            for r in rejected:
                pts = r.reshape(-1, 2).astype(int)
                cv2.polylines(debug, [pts], True, (0, 0, 255), 2)

        cv2.imwrite("debug_aruco_result.png", debug)

        self.get_logger().info("Saved: debug_aruco_input.png")
        self.get_logger().info("Saved: debug_aruco_result.png")
        self.get_logger().info(f"Detected ids: {detected_ids}")
        self.get_logger().info(f"Rejected candidates: {0 if rejected is None else len(rejected)}")

        self.done = True


def main():
    rclpy.init()
    node = DebugArucoCameraImage()

    start = time.time()

    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)

        if time.time() - start > 10.0:
            node.get_logger().error("Timeout: no image received")
            break

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
