#!/usr/bin/env python3

import os
import time
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from transform_utils import (
    rvec_tvec_to_matrix,
    relative_transform_from_common_target,
    translation_norm,
    relative_rotation_angle_deg,
    format_vector,
)


class ArucoRigEstimator(Node):
    def __init__(self):
        super().__init__("aruco_rig_estimator")

        self.declare_parameter(
            "config_path",
            "src/calib_lab/config/ground_truth_aruco.yaml"
        )

        self.config_path = self.get_parameter("config_path").value
        self.config = self.load_config(self.config_path)

        aruco_cfg = self.config["aruco"]

        self.dictionary_name = aruco_cfg["dictionary"]
        self.marker_id = int(aruco_cfg["marker_id"])
        self.marker_size = float(aruco_cfg["marker_size"])

        self.camera_1_name = "camera_1"
        self.camera_2_name = "camera_2"

        cam1_cfg = self.config["cameras"][self.camera_1_name]
        cam2_cfg = self.config["cameras"][self.camera_2_name]

        self.cam1_image_topic = cam1_cfg["image_topic"]
        self.cam1_info_topic = cam1_cfg["camera_info_topic"]
        self.cam2_image_topic = cam2_cfg["image_topic"]
        self.cam2_info_topic = cam2_cfg["camera_info_topic"]

        self.expected_baseline_m = float(
            self.config["ground_truth"]["expected_baseline_m"]
        )

        self.expected_relative_rotation_deg = float(
            self.config["ground_truth"]["expected_relative_rotation_deg"]
        )

        self.bridge = CvBridge()

        self.latest_images = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.camera_matrices = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.dist_coeffs = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.frame_counter = 0
        self.success_counter = 0
        self.last_process_time = 0.0

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
                self.parameters,
            )
        else:
            self.detector = None

        self.object_points = self.create_object_points()

        self.create_subscription(
            CameraInfo,
            self.cam1_info_topic,
            lambda msg: self.camera_info_callback(self.camera_1_name, msg),
            10,
        )

        self.create_subscription(
            CameraInfo,
            self.cam2_info_topic,
            lambda msg: self.camera_info_callback(self.camera_2_name, msg),
            10,
        )

        self.create_subscription(
            Image,
            self.cam1_image_topic,
            lambda msg: self.image_callback(self.camera_1_name, msg),
            10,
        )

        self.create_subscription(
            Image,
            self.cam2_image_topic,
            lambda msg: self.image_callback(self.camera_2_name, msg),
            10,
        )

        self.create_timer(0.5, self.process_pair)

        self.get_logger().info("ArUco Rig Estimator started.")
        self.get_logger().info(f"Config: {self.config_path}")
        self.get_logger().info(f"Dictionary: {self.dictionary_name}")
        self.get_logger().info(f"Marker ID: {self.marker_id}")
        self.get_logger().info(f"Marker size: {self.marker_size:.3f} m")
        self.get_logger().info(
            f"{self.camera_1_name}: image={self.cam1_image_topic}, camera_info={self.cam1_info_topic}"
        )
        self.get_logger().info(
            f"{self.camera_2_name}: image={self.cam2_image_topic}, camera_info={self.cam2_info_topic}"
        )
        self.get_logger().info(
            f"Expected physical camera baseline: {self.expected_baseline_m:.3f} m"
        )

    def load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

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

    def camera_info_callback(self, camera_name, msg):
        self.camera_matrices[camera_name] = np.array(
            msg.k,
            dtype=np.float64,
        ).reshape(3, 3)

        self.dist_coeffs[camera_name] = np.array(
            msg.d,
            dtype=np.float64,
        )

    def image_callback(self, camera_name, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_images[camera_name] = img
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")

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

    def detect_aruco(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected, preprocess_method = self.detect_markers(gray)

        if ids is None or len(ids) == 0:
            rejected_count = 0 if rejected is None else len(rejected)
            return False, None, f"aruco_not_found_rejected_{rejected_count}"

        detected_ids = ids.flatten().tolist()

        if self.marker_id not in detected_ids:
            return False, None, f"target_id_not_found_detected_{detected_ids}"

        marker_index = detected_ids.index(self.marker_id)
        marker_corners = corners[marker_index].reshape(4, 2).astype(np.float32)

        return True, marker_corners, preprocess_method

    def estimate_target_pose_for_camera(self, camera_name):
        image = self.latest_images[camera_name]
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if image is None:
            return False, None, None, "no_image"

        if camera_matrix is None or dist_coeffs is None:
            return False, None, None, "no_camera_info"

        found, marker_corners, preprocess_method = self.detect_aruco(image)

        if not found:
            return False, None, None, preprocess_method

        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
            solvepnp_flag = cv2.SOLVEPNP_IPPE_SQUARE
            solvepnp_method = "IPPE_SQUARE"
        else:
            solvepnp_flag = cv2.SOLVEPNP_ITERATIVE
            solvepnp_method = "ITERATIVE"

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            marker_corners,
            camera_matrix,
            dist_coeffs,
            flags=solvepnp_flag,
        )

        if not ok:
            return False, None, None, "solvepnp_failed"

        T_cam_target = rvec_tvec_to_matrix(rvec, tvec)

        method = f"{solvepnp_method}/{preprocess_method}"

        return True, T_cam_target, tvec.flatten(), method

    def process_pair(self):
        now = time.time()

        if now - self.last_process_time < 0.4:
            return

        self.last_process_time = now
        self.frame_counter += 1

        ok1, T_cam1_target, tvec1, method1 = self.estimate_target_pose_for_camera(
            self.camera_1_name
        )

        ok2, T_cam2_target, tvec2, method2 = self.estimate_target_pose_for_camera(
            self.camera_2_name
        )

        if not ok1 or not ok2:
            if self.frame_counter % 5 == 0:
                self.get_logger().warn(
                    f"Waiting for valid pair | "
                    f"{self.camera_1_name}: {method1} | "
                    f"{self.camera_2_name}: {method2}"
                )
            return

        self.success_counter += 1

        T_cam1_cam2 = relative_transform_from_common_target(
            T_cam1_target,
            T_cam2_target,
        )

        estimated_translation = T_cam1_cam2[:3, 3]
        estimated_baseline = translation_norm(T_cam1_cam2)
        estimated_rotation_deg = relative_rotation_angle_deg(T_cam1_cam2)

        # Additional sanity check for the current synthetic setup:
        # both cameras are parallel, so the lateral baseline can also be checked
        # directly from the two target translation vectors.
        tvec_delta = np.asarray(tvec2) - np.asarray(tvec1)
        translation_only_baseline = float(np.linalg.norm(tvec_delta))
        translation_only_error = abs(
            translation_only_baseline - self.expected_baseline_m
        )

        baseline_error = abs(estimated_baseline - self.expected_baseline_m)

        rotation_error = abs(
            estimated_rotation_deg - self.expected_relative_rotation_deg
        )

        self.get_logger().info(
            "\n"
            "================ ARUCO RIG ESTIMATE ================\n"
            f"valid_pair={self.success_counter}/{self.frame_counter}\n"
            f"{self.camera_1_name}: method={method1}, t_cam_target={format_vector(tvec1)} m\n"
            f"{self.camera_2_name}: method={method2}, t_cam_target={format_vector(tvec2)} m\n"
            "\n"
            "Estimated relative transform from common ArUco target:\n"
            f"T_{self.camera_1_name}_{self.camera_2_name} translation = "
            f"{format_vector(estimated_translation)} m\n"
            f"full-transform baseline norm = {estimated_baseline:.4f} m\n"
            f"expected baseline norm       = {self.expected_baseline_m:.4f} m\n"
            f"full-transform baseline error = {baseline_error:.4f} m "
            f"({baseline_error * 100.0:.2f} cm)\n"
            f"translation-only baseline    = {translation_only_baseline:.4f} m\n"
            f"translation-only error       = {translation_only_error:.4f} m "
            f"({translation_only_error * 100.0:.2f} cm)\n"
            f"estimated relative rotation angle = {estimated_rotation_deg:.4f} deg\n"
            f"expected relative rotation angle  = {self.expected_relative_rotation_deg:.4f} deg\n"
            f"rotation angle error              = {rotation_error:.4f} deg\n"
            "====================================================\n"
        )


def main():
    rclpy.init()
    node = ArucoRigEstimator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
