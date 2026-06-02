#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ArucoLiveDetector(Node):
    def __init__(self):
        super().__init__("aruco_live_detector")

        self.declare_parameter("image_topic", "/camera_1/image")
        self.declare_parameter("marker_id", 23)

        self.image_topic = self.get_parameter("image_topic").value
        self.target_marker_id = int(self.get_parameter("marker_id").value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.success_count = 0

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

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(f"Listening on: {self.image_topic}")
        self.get_logger().info("Searching ArUco dictionary: DICT_6X6_250")
        self.get_logger().info(f"Target marker ID: {self.target_marker_id}")

    def detect_markers_once(self, image):
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(image)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                image,
                self.dictionary,
                parameters=self.parameters,
            )

        return corners, ids, rejected

    def detect_markers(self, gray):
        variants = []

        variants.append(("gray", gray))

        normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )
        variants.append(("normalized", normalized))

        for threshold_value in [40, 60, 80, 100]:
            _, binary = cv2.threshold(
                gray,
                threshold_value,
                255,
                cv2.THRESH_BINARY
            )
            variants.append((f"binary_{threshold_value}", binary))

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        variants.append(("adaptive", adaptive))

        best_rejected = None

        for method_name, image_variant in variants:
            corners, ids, rejected = self.detect_markers_once(image_variant)

            if rejected is not None:
                if best_rejected is None or len(rejected) > len(best_rejected):
                    best_rejected = rejected

            if ids is not None and len(ids) > 0:
                return corners, ids, rejected, method_name

        return [], None, best_rejected, "none"

    def image_callback(self, msg):
        self.frame_count += 1

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected, preprocess_method = self.detect_markers(gray)

        if ids is None or len(ids) == 0:
            if self.frame_count % 10 == 0:
                rejected_count = 0 if rejected is None else len(rejected)
                self.get_logger().warn(
                    f"not found | frame={self.frame_count} "
                    f"| success={self.success_count}/{self.frame_count} "
                    f"| rejected={rejected_count}"
                )
            return

        detected_ids = ids.flatten().tolist()

        if self.target_marker_id not in detected_ids:
            if self.frame_count % 10 == 0:
                self.get_logger().warn(
                    f"marker found but not target | frame={self.frame_count} "
                    f"| ids={detected_ids} "
                    f"| target={self.target_marker_id}"
                )
            return

        marker_index = detected_ids.index(self.target_marker_id)
        marker_corners = corners[marker_index]

        self.success_count += 1

        self.get_logger().info(
            f"FOUND aruco | frame={self.frame_count} "
            f"| ids={detected_ids} "
            f"| target_id={self.target_marker_id} "
            f"| corners={len(marker_corners[0])} "
            f"| preprocess={preprocess_method} "
            f"| success={self.success_count}/{self.frame_count}"
        )


def main():
    rclpy.init()
    node = ArucoLiveDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
