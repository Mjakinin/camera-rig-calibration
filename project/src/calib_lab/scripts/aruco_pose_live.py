#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class ArucoPoseLive(Node):
    def __init__(self):
        super().__init__("aruco_pose_live")

        self.declare_parameter("image_topic", "/camera_1/image")
        self.declare_parameter("camera_info_topic", "/camera_1/camera_info")
        self.declare_parameter("marker_id", 23)
        self.declare_parameter("marker_size", 0.96)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_marker_id = int(self.get_parameter("marker_id").value)
        self.marker_size = float(self.get_parameter("marker_size").value)

        self.bridge = CvBridge()

        self.camera_matrix = None
        self.dist_coeffs = None

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

        self.object_points = self.create_object_points()

        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.camera_info_topic}")
        self.get_logger().info("ArUco dictionary: DICT_6X6_250")
        self.get_logger().info(f"Target marker ID: {self.target_marker_id}")
        self.get_logger().info(f"Marker size: {self.marker_size} m")

    def create_object_points(self):
        s = self.marker_size / 2.0

        # OpenCV ArUco corner order:
        # top-left, top-right, bottom-right, bottom-left
        objp = np.array(
            [
                [-s,  s, 0.0],
                [ s,  s, 0.0],
                [ s, -s, 0.0],
                [-s, -s, 0.0],
            ],
            dtype=np.float32,
        )

        return objp

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

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
        if self.camera_matrix is None:
            return

        self.frame_count += 1

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected, preprocess_method = self.detect_markers(gray)

        if ids is None or len(ids) == 0:
            if self.frame_count % 10 == 0:
                rejected_count = 0 if rejected is None else len(rejected)
                self.get_logger().warn(
                    f"aruco not found | frame={self.frame_count} "
                    f"| rejected={rejected_count}"
                )
            return

        detected_ids = ids.flatten().tolist()

        if self.target_marker_id not in detected_ids:
            if self.frame_count % 10 == 0:
                self.get_logger().warn(
                    f"target marker not found | frame={self.frame_count} "
                    f"| detected_ids={detected_ids}"
                )
            return

        marker_index = detected_ids.index(self.target_marker_id)
        marker_corners = corners[marker_index].reshape(4, 2).astype(np.float32)

        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
            solvepnp_flag = cv2.SOLVEPNP_IPPE_SQUARE
            method = "IPPE_SQUARE"
        else:
            solvepnp_flag = cv2.SOLVEPNP_ITERATIVE
            method = "ITERATIVE"

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            marker_corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=solvepnp_flag,
        )

        if not ok:
            self.get_logger().warn("solvePnP failed")
            return

        self.success_count += 1

        t = tvec.flatten()
        distance = float(np.linalg.norm(t))

        self.get_logger().info(
            f"ARUCO POSE FOUND | method={method} "
            f"| preprocess={preprocess_method} "
            f"| id={self.target_marker_id} "
            f"| frame={self.frame_count} "
            f"| t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m "
            f"| dist={distance:.3f} m "
            f"| success={self.success_count}/{self.frame_count}"
        )


def main():
    rclpy.init()
    node = ArucoPoseLive()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
