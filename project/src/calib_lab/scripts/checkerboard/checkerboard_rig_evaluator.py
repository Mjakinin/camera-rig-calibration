#!/usr/bin/env python3

import os
import csv
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


class CheckerboardRigEvaluator(Node):
    def __init__(self):
        super().__init__("checkerboard_rig_evaluator")

        self.declare_parameter("config_path", "src/calib_lab/config/ground_truth_minimal.yaml")
        self.declare_parameter("scenario_name", "checkerboard_static")
        self.declare_parameter("output_csv", "results/checkerboard_sweep_results.csv")
        self.declare_parameter("debug_dir", "results/debug_images")
        self.declare_parameter("max_valid_samples", 1)
        self.declare_parameter("max_attempts", 1)
        self.declare_parameter("process_period_sec", 0.5)
        self.declare_parameter("ready_timeout_sec", 15.0)

        self.config_path = self.get_parameter("config_path").value
        self.scenario_name = self.get_parameter("scenario_name").value
        self.output_csv = self.get_parameter("output_csv").value
        self.debug_dir = self.get_parameter("debug_dir").value
        self.max_valid_samples = int(self.get_parameter("max_valid_samples").value)
        self.max_attempts = int(self.get_parameter("max_attempts").value)
        self.process_period_sec = float(self.get_parameter("process_period_sec").value)
        self.ready_timeout_sec = float(self.get_parameter("ready_timeout_sec").value)

        os.makedirs(self.debug_dir, exist_ok=True)

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

        self.expected_baseline_m = float(self.config["ground_truth"]["expected_baseline_m"])
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

        self.total_attempts = 0
        self.valid_samples = 0
        self.finished = False
        self.start_time = time.time()
        self.ready_log_counter = 0

        self.last_status = {
            self.camera_1_name: "not_started",
            self.camera_2_name: "not_started",
        }

        self.last_corners = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.last_num_points = {
            self.camera_1_name: 0,
            self.camera_2_name: 0,
        }

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

        self.create_timer(self.process_period_sec, self.process_pair)

        self.get_logger().info("Checkerboard Rig Evaluator started.")
        self.get_logger().info(f"Scenario: {self.scenario_name}")
        self.get_logger().info(f"Config: {self.config_path}")
        self.get_logger().info(f"Output CSV: {self.output_csv}")
        self.get_logger().info(f"Debug dir: {self.debug_dir}")
        self.get_logger().info(f"Max valid samples: {self.max_valid_samples}")
        self.get_logger().info(f"Max attempts: {self.max_attempts}")
        self.get_logger().info(f"Ready timeout: {self.ready_timeout_sec} sec")
        self.get_logger().info(
            f"Checkerboard pattern: {self.pattern_size}, square_size={self.square_size} m"
        )
        self.get_logger().info(f"Expected baseline: {self.expected_baseline_m:.4f} m")

    def load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def create_object_points(self):
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
        """
        Strict SB-only checkerboard detection.

        We intentionally do NOT mix findChessboardCornersSB and classic
        findChessboardCorners in the main experiment, because mixed detectors
        can return different corner ordering/orientation under difficult views.
        That can create wrong camera-to-camera transforms with ~180 deg rotation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            return True, corners, "SB"

        return False, None, "checkerboard_not_found"

    def estimate_target_pose_for_camera(self, camera_name):
        image = self.latest_images[camera_name]
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if image is None:
            self.last_status[camera_name] = "no_image"
            self.last_corners[camera_name] = None
            self.last_num_points[camera_name] = 0
            return False, None, None, "no_image"

        if camera_matrix is None or dist_coeffs is None:
            self.last_status[camera_name] = "no_camera_info"
            self.last_corners[camera_name] = None
            self.last_num_points[camera_name] = 0
            return False, None, None, "no_camera_info"

        found, corners, method = self.detect_checkerboard(image)

        self.last_status[camera_name] = method
        self.last_corners[camera_name] = corners
        self.last_num_points[camera_name] = int(len(corners)) if corners is not None else 0

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
            self.last_status[camera_name] = "solvepnp_failed"
            return False, None, None, "solvepnp_failed"

        T_cam_target = rvec_tvec_to_matrix(rvec, tvec)
        return True, T_cam_target, tvec.flatten(), method

    def save_debug_image(self, camera_name, success):
        image = self.latest_images[camera_name]
        status = self.last_status[camera_name]
        corners = self.last_corners[camera_name]

        if image is None:
            return ""

        img = image.copy()

        if corners is not None and status in ["SB", "classic"]:
            cv2.drawChessboardCorners(img, self.pattern_size, corners, True)

        label = f"{self.scenario_name} | {camera_name} | {status}"
        color = (0, 255, 0) if success else (0, 0, 255)

        cv2.putText(
            img,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        filename = f"{self.scenario_name}_{camera_name}_{status}.png"
        path = os.path.join(self.debug_dir, filename)
        cv2.imwrite(path, img)
        return path

    def inputs_ready(self):
        missing = []

        for camera_name in [self.camera_1_name, self.camera_2_name]:
            if self.latest_images[camera_name] is None:
                missing.append(f"{camera_name}:no_image")

            if self.camera_matrices[camera_name] is None or self.dist_coeffs[camera_name] is None:
                missing.append(f"{camera_name}:no_camera_info")

        return len(missing) == 0, missing

    def set_missing_statuses(self):
        for camera_name in [self.camera_1_name, self.camera_2_name]:
            if self.latest_images[camera_name] is None:
                self.last_status[camera_name] = "no_image"
                self.last_corners[camera_name] = None
                self.last_num_points[camera_name] = 0
            elif self.camera_matrices[camera_name] is None or self.dist_coeffs[camera_name] is None:
                self.last_status[camera_name] = "no_camera_info"
                self.last_corners[camera_name] = None
                self.last_num_points[camera_name] = 0

    def process_pair(self):
        if self.finished:
            return

        ready, missing = self.inputs_ready()

        if not ready:
            self.ready_log_counter += 1

            if self.ready_log_counter % 5 == 0:
                self.get_logger().warn(
                    f"Waiting for input topics before detection attempt | missing={missing}"
                )

            if time.time() - self.start_time >= self.ready_timeout_sec:
                self.set_missing_statuses()
                self.total_attempts = 0
                self.write_failure_and_exit()

            return

        # Count only real detection/PnP attempts after images and camera_info exist.
        self.total_attempts += 1

        ok1, T_cam1_target, tvec1, method1 = self.estimate_target_pose_for_camera(
            self.camera_1_name
        )

        ok2, T_cam2_target, tvec2, method2 = self.estimate_target_pose_for_camera(
            self.camera_2_name
        )

        if ok1 and ok2:
            self.valid_samples += 1

            T_cam1_cam2 = relative_transform_from_common_target(
                T_cam1_target,
                T_cam2_target,
            )

            estimated_translation = T_cam1_cam2[:3, 3]
            estimated_baseline = translation_norm(T_cam1_cam2)
            estimated_rotation_deg = relative_rotation_angle_deg(T_cam1_cam2)

            baseline_error_m = abs(estimated_baseline - self.expected_baseline_m)
            rotation_error_deg = abs(
                estimated_rotation_deg - self.expected_relative_rotation_deg
            )

            cam1_img = self.save_debug_image(self.camera_1_name, True)
            cam2_img = self.save_debug_image(self.camera_2_name, True)

            row = {
                "timestamp": time.time(),
                "scenario": self.scenario_name,
                "method": "checkerboard",
                "success": "true",
                "failure_reason": "",
                "total_attempts": self.total_attempts,
                "estimated_baseline_m": estimated_baseline,
                "expected_baseline_m": self.expected_baseline_m,
                "baseline_error_m": baseline_error_m,
                "rotation_error_deg": rotation_error_deg,
                "translation_x_m": float(estimated_translation[0]),
                "translation_y_m": float(estimated_translation[1]),
                "translation_z_m": float(estimated_translation[2]),
                "camera_1_status": self.last_status[self.camera_1_name],
                "camera_2_status": self.last_status[self.camera_2_name],
                "camera_1_points": self.last_num_points[self.camera_1_name],
                "camera_2_points": self.last_num_points[self.camera_2_name],
                "camera_1_image": cam1_img,
                "camera_2_image": cam2_img,
                "camera_1_target_x_m": float(tvec1[0]),
                "camera_1_target_y_m": float(tvec1[1]),
                "camera_1_target_z_m": float(tvec1[2]),
                "camera_2_target_x_m": float(tvec2[0]),
                "camera_2_target_y_m": float(tvec2[1]),
                "camera_2_target_z_m": float(tvec2[2]),
            }

            self.write_row(row)

            self.get_logger().info(
                "\n"
                "================ CHECKERBOARD EVALUATION SUCCESS ================\n"
                f"scenario: {self.scenario_name}\n"
                f"attempts: {self.total_attempts}\n"
                f"camera_1_status: {self.last_status[self.camera_1_name]}\n"
                f"camera_2_status: {self.last_status[self.camera_2_name]}\n"
                f"estimated baseline: {estimated_baseline:.4f} m\n"
                f"baseline error: {baseline_error_m:.4f} m ({baseline_error_m * 100.0:.2f} cm)\n"
                f"rotation error: {rotation_error_deg:.4f} deg\n"
                f"translation: {format_vector(estimated_translation)}\n"
                f"camera_1 image: {cam1_img}\n"
                f"camera_2 image: {cam2_img}\n"
                "=================================================================\n"
            )

            self.finish_process()

        else:
            if self.total_attempts % 5 == 0:
                self.get_logger().warn(
                    f"Waiting for valid pair | "
                    f"{self.camera_1_name}: {self.last_status[self.camera_1_name]} | "
                    f"{self.camera_2_name}: {self.last_status[self.camera_2_name]}"
                )

            if self.total_attempts >= self.max_attempts:
                self.write_failure_and_exit()

    def write_failure_and_exit(self):
        cam1_img = self.save_debug_image(self.camera_1_name, False)
        cam2_img = self.save_debug_image(self.camera_2_name, False)

        reason = (
            f"{self.camera_1_name}:{self.last_status[self.camera_1_name]};"
            f"{self.camera_2_name}:{self.last_status[self.camera_2_name]}"
        )

        row = {
            "timestamp": time.time(),
            "scenario": self.scenario_name,
            "method": "checkerboard",
            "success": "false",
            "failure_reason": reason,
            "total_attempts": self.total_attempts,
            "estimated_baseline_m": "",
            "expected_baseline_m": self.expected_baseline_m,
            "baseline_error_m": "",
            "rotation_error_deg": "",
            "translation_x_m": "",
            "translation_y_m": "",
            "translation_z_m": "",
            "camera_1_status": self.last_status[self.camera_1_name],
            "camera_2_status": self.last_status[self.camera_2_name],
            "camera_1_points": self.last_num_points[self.camera_1_name],
            "camera_2_points": self.last_num_points[self.camera_2_name],
            "camera_1_image": cam1_img,
            "camera_2_image": cam2_img,
            "camera_1_target_x_m": "",
            "camera_1_target_y_m": "",
            "camera_1_target_z_m": "",
            "camera_2_target_x_m": "",
            "camera_2_target_y_m": "",
            "camera_2_target_z_m": "",
        }

        self.write_row(row)

        self.get_logger().warn(
            "\n"
            "================ CHECKERBOARD EVALUATION FAILURE ================\n"
            f"scenario: {self.scenario_name}\n"
            f"attempts: {self.total_attempts}\n"
            f"failure_reason: {reason}\n"
            f"camera_1 image: {cam1_img}\n"
            f"camera_2 image: {cam2_img}\n"
            "=================================================================\n"
        )

        self.finish_process()

    def write_row(self, row):
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        fieldnames = [
            "timestamp",
            "scenario",
            "method",
            "success",
            "failure_reason",
            "total_attempts",
            "estimated_baseline_m",
            "expected_baseline_m",
            "baseline_error_m",
            "rotation_error_deg",
            "translation_x_m",
            "translation_y_m",
            "translation_z_m",
            "camera_1_status",
            "camera_2_status",
            "camera_1_points",
            "camera_2_points",
            "camera_1_image",
            "camera_2_image",
            "camera_1_target_x_m",
            "camera_1_target_y_m",
            "camera_1_target_z_m",
            "camera_2_target_x_m",
            "camera_2_target_y_m",
            "camera_2_target_z_m",
        ]

        file_exists = os.path.exists(self.output_csv)

        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def finish_process(self):
        self.finished = True
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main():
    rclpy.init()
    node = CheckerboardRigEvaluator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
