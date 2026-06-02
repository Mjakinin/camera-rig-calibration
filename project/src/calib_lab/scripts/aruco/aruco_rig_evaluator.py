#!/usr/bin/env python3

import os
import csv
import time
import yaml
from pathlib import Path
import sys

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from common.transform_utils import (
    rvec_tvec_to_matrix,
    relative_transform_from_common_target,
    translation_norm,
    relative_rotation_angle_deg,
    format_vector,
)


class ArucoRigEvaluator(Node):
    def __init__(self):
        super().__init__("aruco_rig_evaluator")

        self.declare_parameter("config_path", "src/calib_lab/config/ground_truth_minimal.yaml")
        self.declare_parameter("aruco_config_path", "src/calib_lab/config/aruco_target.yaml")
        self.declare_parameter("scenario_name", "aruco_static")
        self.declare_parameter("output_csv", "results/aruco/raw_results.csv")
        self.declare_parameter("debug_dir", "results/aruco/debug_images")
        self.declare_parameter("max_valid_samples", 1)
        self.declare_parameter("max_attempts", 1)
        self.declare_parameter("ready_timeout_sec", 20.0)
        self.declare_parameter("process_period_sec", 0.5)
        self.declare_parameter("min_markers", 1)

        self.config_path = self.get_parameter("config_path").value
        self.aruco_config_path = self.get_parameter("aruco_config_path").value
        self.scenario_name = self.get_parameter("scenario_name").value
        self.output_csv = self.get_parameter("output_csv").value
        self.debug_dir = self.get_parameter("debug_dir").value
        self.max_valid_samples = int(self.get_parameter("max_valid_samples").value)
        self.max_attempts = int(self.get_parameter("max_attempts").value)
        self.ready_timeout_sec = float(self.get_parameter("ready_timeout_sec").value)
        self.process_period_sec = float(self.get_parameter("process_period_sec").value)
        self.min_markers = int(self.get_parameter("min_markers").value)

        os.makedirs(self.debug_dir, exist_ok=True)

        self.config = self.load_yaml(self.config_path)
        self.aruco_cfg = self.load_yaml(self.aruco_config_path)["aruco"]

        self.dictionary_name = self.aruco_cfg["dictionary"]
        self.markers_x = int(self.aruco_cfg["markers_x"])
        self.markers_y = int(self.aruco_cfg["markers_y"])
        self.marker_length = float(self.aruco_cfg["marker_length"])
        self.marker_separation = float(self.aruco_cfg["marker_separation"])

        self.dictionary = self.get_dictionary(self.dictionary_name)
        self.detector = self.create_detector()

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

        self.last_status = {
            self.camera_1_name: "not_started",
            self.camera_2_name: "not_started",
        }

        self.last_corners = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.last_ids = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.last_used_ids = {
            self.camera_1_name: [],
            self.camera_2_name: [],
        }

        self.last_rvec = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.last_tvec = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.total_attempts = 0
        self.valid_samples = 0
        self.finished = False
        self.start_time = time.time()
        self.ready_log_counter = 0

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

        self.get_logger().info("ArUco Rig Evaluator started.")
        self.get_logger().info(f"Scenario: {self.scenario_name}")
        self.get_logger().info(f"Config: {self.config_path}")
        self.get_logger().info(f"ArUco config: {self.aruco_config_path}")
        self.get_logger().info(f"Output CSV: {self.output_csv}")
        self.get_logger().info(f"Debug dir: {self.debug_dir}")
        self.get_logger().info(
            f"Board: {self.markers_x}x{self.markers_y}, "
            f"marker_length={self.marker_length}, separation={self.marker_separation}, "
            f"dictionary={self.dictionary_name}"
        )
        self.get_logger().info(f"Expected baseline: {self.expected_baseline_m:.4f} m")
        self.get_logger().info(f"Max attempts: {self.max_attempts}")

    def load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_dictionary(self, dictionary_name):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco is not available.")

        if not hasattr(cv2.aruco, dictionary_name):
            raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")

        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    def create_detector(self):
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(self.dictionary, params)

        return None

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
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_images[camera_name] = image
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")

    def marker_object_corners(self, marker_id):
        """
        3D coordinates of one marker in board coordinates.

        Corner order matches OpenCV ArUco:
        top-left, top-right, bottom-right, bottom-left.

        Coordinate convention:
        x increases right.
        y increases upward.
        Texture rows go downward, therefore board y decreases with marker row.
        """
        if marker_id < 0 or marker_id >= self.markers_x * self.markers_y:
            return None

        row = marker_id // self.markers_x
        col = marker_id % self.markers_x

        step = self.marker_length + self.marker_separation

        x0 = col * step
        y0 = -row * step
        L = self.marker_length

        return np.array(
            [
                [x0, y0, 0.0],
                [x0 + L, y0, 0.0],
                [x0 + L, y0 - L, 0.0],
                [x0, y0 - L, 0.0],
            ],
            dtype=np.float32,
        )

    def detect_markers_once(self, gray):
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary)

        return corners, ids, rejected

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        variants = []

        variants.append(("gray", gray))

        normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )
        variants.append(("normalized", normalized))

        for threshold_value in [40, 60, 80, 100]:
            _, binary = cv2.threshold(
                gray,
                threshold_value,
                255,
                cv2.THRESH_BINARY,
            )
            variants.append((f"binary_{threshold_value}", binary))

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
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

    def build_correspondences(self, corners, ids):
        if ids is None or len(ids) == 0:
            return None, None, []

        object_points = []
        image_points = []
        used_ids = []

        ids_flat = ids.flatten().astype(int)

        for i, marker_id in enumerate(ids_flat):
            obj_corners = self.marker_object_corners(marker_id)

            if obj_corners is None:
                continue

            img_corners = corners[i].reshape(4, 2).astype(np.float32)

            object_points.append(obj_corners)
            image_points.append(img_corners)
            used_ids.append(int(marker_id))

        if not object_points:
            return None, None, []

        object_points = np.concatenate(object_points, axis=0).astype(np.float32)
        image_points = np.concatenate(image_points, axis=0).astype(np.float32)

        return object_points, image_points, used_ids

    def estimate_target_pose_for_camera(self, camera_name):
        image = self.latest_images[camera_name]
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        self.last_rvec[camera_name] = None
        self.last_tvec[camera_name] = None
        self.last_used_ids[camera_name] = []

        if image is None:
            self.last_status[camera_name] = "no_image"
            self.last_corners[camera_name] = None
            self.last_ids[camera_name] = None
            return False, None, None, "no_image"

        if camera_matrix is None or dist_coeffs is None:
            self.last_status[camera_name] = "no_camera_info"
            self.last_corners[camera_name] = None
            self.last_ids[camera_name] = None
            return False, None, None, "no_camera_info"

        corners, ids, rejected, preprocess_method = self.detect_markers(image)

        self.last_corners[camera_name] = corners
        self.last_ids[camera_name] = ids

        if ids is None or len(ids) == 0:
            self.last_status[camera_name] = "aruco_not_found"
            return False, None, None, f"aruco_not_found/{preprocess_method}"

        object_points, image_points, used_ids = self.build_correspondences(corners, ids)
        self.last_used_ids[camera_name] = used_ids

        if object_points is None or image_points is None:
            self.last_status[camera_name] = "no_known_marker_ids"
            return False, None, None, f"no_known_marker_ids/{preprocess_method}"

        if len(used_ids) < self.min_markers:
            self.last_status[camera_name] = "not_enough_markers"
            return False, None, None, f"not_enough_markers/{preprocess_method}"

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            self.last_status[camera_name] = "solvepnp_failed"
            return False, None, None, f"solvepnp_failed/{preprocess_method}"

        self.last_status[camera_name] = f"aruco/{preprocess_method}"
        self.last_rvec[camera_name] = rvec
        self.last_tvec[camera_name] = tvec

        T_cam_target = rvec_tvec_to_matrix(rvec, tvec)

        method = f"aruco/{preprocess_method}"
        return True, T_cam_target, tvec.flatten(), method

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
                self.last_used_ids[camera_name] = []
            elif self.camera_matrices[camera_name] is None or self.dist_coeffs[camera_name] is None:
                self.last_status[camera_name] = "no_camera_info"
                self.last_used_ids[camera_name] = []

    def save_debug_image(self, camera_name, success):
        image = self.latest_images[camera_name]

        if image is None:
            return ""

        annotated = image.copy()

        corners = self.last_corners[camera_name]
        ids = self.last_ids[camera_name]

        if ids is not None and corners is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        rvec = self.last_rvec[camera_name]
        tvec = self.last_tvec[camera_name]
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if success and rvec is not None and tvec is not None:
            cv2.drawFrameAxes(
                annotated,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                self.marker_length * 1.5,
            )

        status = self.last_status[camera_name]
        used_ids = self.last_used_ids[camera_name]

        label = f"{self.scenario_name} | {camera_name} | {status} | markers={len(used_ids)}"
        color = (0, 255, 0) if success else (0, 0, 255)

        cv2.putText(
            annotated,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

        safe_status = status.replace("/", "_").replace(" ", "_")
        filename = f"{self.scenario_name}_{camera_name}_{safe_status}.png"
        path = os.path.join(self.debug_dir, filename)

        cv2.imwrite(path, annotated)

        return path

    def make_base_row(
        self,
        success,
        failure_reason,
        estimated_baseline,
        baseline_error_m,
        rotation_error_deg,
        estimated_translation,
        cam1_img,
        cam2_img,
        tvec1,
        tvec2,
    ):
        if isinstance(estimated_translation, np.ndarray):
            tx = float(estimated_translation[0])
            ty = float(estimated_translation[1])
            tz = float(estimated_translation[2])
        else:
            tx = ty = tz = ""

        if isinstance(tvec1, np.ndarray):
            c1x = float(tvec1[0])
            c1y = float(tvec1[1])
            c1z = float(tvec1[2])
        else:
            c1x = c1y = c1z = ""

        if isinstance(tvec2, np.ndarray):
            c2x = float(tvec2[0])
            c2y = float(tvec2[1])
            c2z = float(tvec2[2])
        else:
            c2x = c2y = c2z = ""

        return {
            "timestamp": time.time(),
            "scenario": self.scenario_name,
            "method": "aruco",
            "success": success,
            "failure_reason": failure_reason,
            "total_attempts": self.total_attempts,
            "estimated_baseline_m": estimated_baseline,
            "expected_baseline_m": self.expected_baseline_m,
            "baseline_error_m": baseline_error_m,
            "rotation_error_deg": rotation_error_deg,
            "translation_x_m": tx,
            "translation_y_m": ty,
            "translation_z_m": tz,
            "camera_1_status": self.last_status[self.camera_1_name],
            "camera_2_status": self.last_status[self.camera_2_name],
            "camera_1_points": len(self.last_used_ids[self.camera_1_name]) * 4,
            "camera_2_points": len(self.last_used_ids[self.camera_2_name]) * 4,
            "camera_1_markers": len(self.last_used_ids[self.camera_1_name]),
            "camera_2_markers": len(self.last_used_ids[self.camera_2_name]),
            "camera_1_ids": " ".join(str(x) for x in self.last_used_ids[self.camera_1_name]),
            "camera_2_ids": " ".join(str(x) for x in self.last_used_ids[self.camera_2_name]),
            "camera_1_image": cam1_img,
            "camera_2_image": cam2_img,
            "camera_1_target_x_m": c1x,
            "camera_1_target_y_m": c1y,
            "camera_1_target_z_m": c1z,
            "camera_2_target_x_m": c2x,
            "camera_2_target_y_m": c2y,
            "camera_2_target_z_m": c2z,
        }

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
            "camera_1_markers",
            "camera_2_markers",
            "camera_1_ids",
            "camera_2_ids",
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

    def write_failure_and_exit(self):
        cam1_img = self.save_debug_image(self.camera_1_name, False)
        cam2_img = self.save_debug_image(self.camera_2_name, False)

        reason = (
            f"{self.camera_1_name}:{self.last_status[self.camera_1_name]};"
            f"{self.camera_2_name}:{self.last_status[self.camera_2_name]}"
        )

        row = self.make_base_row(
            success="false",
            failure_reason=reason,
            estimated_baseline="",
            baseline_error_m="",
            rotation_error_deg="",
            estimated_translation="",
            cam1_img=cam1_img,
            cam2_img=cam2_img,
            tvec1="",
            tvec2="",
        )

        self.write_row(row)

        self.get_logger().warn(
            "\n"
            "================ ARUCO EVALUATION FAILURE ================\n"
            f"scenario: {self.scenario_name}\n"
            f"attempts: {self.total_attempts}\n"
            f"failure_reason: {reason}\n"
            f"camera_1 markers: {len(self.last_used_ids[self.camera_1_name])}\n"
            f"camera_2 markers: {len(self.last_used_ids[self.camera_2_name])}\n"
            f"camera_1 image: {cam1_img}\n"
            f"camera_2 image: {cam2_img}\n"
            "==========================================================\n"
        )

        self.finish_process()

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

            row = self.make_base_row(
                success="true",
                failure_reason="",
                estimated_baseline=estimated_baseline,
                baseline_error_m=baseline_error_m,
                rotation_error_deg=rotation_error_deg,
                estimated_translation=estimated_translation,
                cam1_img=cam1_img,
                cam2_img=cam2_img,
                tvec1=tvec1,
                tvec2=tvec2,
            )

            self.write_row(row)

            self.get_logger().info(
                "\n"
                "================ ARUCO EVALUATION SUCCESS ================\n"
                f"scenario: {self.scenario_name}\n"
                f"attempts: {self.total_attempts}\n"
                f"{self.camera_1_name}: method={method1}, markers={len(self.last_used_ids[self.camera_1_name])}, "
                f"t_cam_target={format_vector(tvec1)} m\n"
                f"{self.camera_2_name}: method={method2}, markers={len(self.last_used_ids[self.camera_2_name])}, "
                f"t_cam_target={format_vector(tvec2)} m\n"
                "\n"
                "Estimated relative transform from common ArUco target:\n"
                f"T_{self.camera_1_name}_{self.camera_2_name} translation = "
                f"{format_vector(estimated_translation)} m\n"
                f"estimated baseline norm = {estimated_baseline:.4f} m\n"
                f"expected baseline norm  = {self.expected_baseline_m:.4f} m\n"
                f"baseline error          = {baseline_error_m:.4f} m "
                f"({baseline_error_m * 100.0:.2f} cm)\n"
                f"estimated relative rotation angle = {estimated_rotation_deg:.4f} deg\n"
                f"expected relative rotation angle  = {self.expected_relative_rotation_deg:.4f} deg\n"
                f"rotation angle error              = {rotation_error_deg:.4f} deg\n"
                f"camera_1 image: {cam1_img}\n"
                f"camera_2 image: {cam2_img}\n"
                "==========================================================\n"
            )

            self.finish_process()

        else:
            if self.total_attempts >= self.max_attempts:
                self.write_failure_and_exit()

    def finish_process(self):
        self.finished = True
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main():
    rclpy.init()
    node = ArucoRigEvaluator()

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
