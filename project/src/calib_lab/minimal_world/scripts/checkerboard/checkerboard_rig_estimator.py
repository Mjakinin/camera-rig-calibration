#!/usr/bin/env python3
# AUTO_IMPORT_COMMON_START
from pathlib import Path as _CalibLabPath
import sys as _CalibLabSys
for _p in _CalibLabPath(__file__).resolve().parents:
    if _p.name == "calib_lab":
        if str(_p) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_p))
        _common_scripts = _p / "common" / "scripts"
        if _common_scripts.exists() and str(_common_scripts) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_common_scripts))
        break
# AUTO_IMPORT_COMMON_END

import os
import time
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import sys
from pathlib import Path
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
from common.transform_utils import (
    rvec_tvec_to_matrix,
    relative_transform_from_common_target,
    translation_norm,
    relative_rotation_angle_deg,
    format_vector,
)


class CheckerboardRigEstimator(Node):
    def __init__(self):
        super().__init__("checkerboard_rig_estimator")

        self.declare_parameter(
            "config_path",
            "src/calib_lab/minimal_world/config/ground_truth_minimal.yaml"
        )

        self.config_path = self.get_parameter("config_path").value
        self.config = self.load_config(self.config_path)

        checker_cfg = self.config["checkerboard"]
        self.corners_x = int(checker_cfg["corners_x"])
        self.corners_y = int(checker_cfg["corners_y"])
        self.square_size = float(checker_cfg["square_size"])
        self.pattern_size = (self.corners_x, self.corners_y)

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
        self.object_points = self.create_object_points()

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

        # Process the latest available pair twice per second.
        # This is enough for the current static target test.
        self.create_timer(0.5, self.process_pair)

        self.get_logger().info("Checkerboard Rig Estimator started.")
        self.get_logger().info(f"Config: {self.config_path}")
        self.get_logger().info(
            f"Checkerboard pattern: {self.pattern_size}, square_size={self.square_size} m"
        )
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
        """
        Create known checkerboard 3D points in the local target frame.

        The board is planar, so z=0 for all points.
        """
        objp = np.zeros((self.corners_x * self.corners_y, 3), np.float32)
        grid = np.mgrid[0:self.corners_x, 0:self.corners_y].T.reshape(-1, 2)
        objp[:, :2] = grid * self.square_size
        return objp

    def camera_info_callback(self, camera_name, msg):
        self.camera_matrices[camera_name] = np.array(
            msg.k,
            dtype=np.float64
        ).reshape(3, 3)

        self.dist_coeffs[camera_name] = np.array(
            msg.d,
            dtype=np.float64
        )

    def image_callback(self, camera_name, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_images[camera_name] = img
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")

    def detect_checkerboard(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            return True, corners, "SB"

        found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
            return True, corners, "classic"

        return False, None, "none"

    def estimate_target_pose_for_camera(self, camera_name):
        image = self.latest_images[camera_name]
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if image is None:
            return False, None, None, "no_image"

        if camera_matrix is None or dist_coeffs is None:
            return False, None, None, "no_camera_info"

        found, corners, method = self.detect_checkerboard(image)

        if not found:
            return False, None, None, "checkerboard_not_found"

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return False, None, None, "solvepnp_failed"

        T_cam_target = rvec_tvec_to_matrix(rvec, tvec)
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

        baseline_error = abs(estimated_baseline - self.expected_baseline_m)
        rotation_error = abs(
            estimated_rotation_deg - self.expected_relative_rotation_deg
        )

        self.get_logger().info(
            "\n"
            "================ CHECKERBOARD RIG ESTIMATE ================\n"
            f"valid_pair={self.success_counter}/{self.frame_counter}\n"
            f"{self.camera_1_name}: method={method1}, t_cam_target={format_vector(tvec1)} m\n"
            f"{self.camera_2_name}: method={method2}, t_cam_target={format_vector(tvec2)} m\n"
            "\n"
            "Estimated relative transform from common target:\n"
            f"T_{self.camera_1_name}_{self.camera_2_name} translation = "
            f"{format_vector(estimated_translation)} m\n"
            f"estimated baseline norm = {estimated_baseline:.4f} m\n"
            f"expected baseline norm  = {self.expected_baseline_m:.4f} m\n"
            f"baseline error          = {baseline_error:.4f} m "
            f"({baseline_error * 100.0:.2f} cm)\n"
            f"estimated relative rotation angle = {estimated_rotation_deg:.4f} deg\n"
            f"expected relative rotation angle  = {self.expected_relative_rotation_deg:.4f} deg\n"
            f"rotation angle error              = {rotation_error:.4f} deg\n"
            "===========================================================\n"
        )


def main():
    rclpy.init()
    node = CheckerboardRigEstimator()

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
