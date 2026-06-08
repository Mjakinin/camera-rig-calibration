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


class ArucoRigEstimator(Node):
    def __init__(self):
        super().__init__("aruco_rig_estimator")

        self.declare_parameter("config_path", "src/calib_lab/minimal_world/config/ground_truth_minimal.yaml")
        self.declare_parameter("aruco_config_path", "src/calib_lab/config/aruco_target.yaml")
        self.declare_parameter("process_period_sec", 0.5)
        self.declare_parameter("min_markers", 1)
        self.declare_parameter("show_gui", True)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "results/minimal_world/aruco/rig_estimator/debug_images")
        self.declare_parameter("save_every_n_successes", 10)

        self.config_path = self.get_parameter("config_path").value
        self.aruco_config_path = self.get_parameter("aruco_config_path").value
        self.process_period_sec = float(self.get_parameter("process_period_sec").value)
        self.min_markers = int(self.get_parameter("min_markers").value)
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.save_debug = bool(self.get_parameter("save_debug").value)
        self.debug_dir = self.get_parameter("debug_dir").value
        self.save_every_n_successes = int(self.get_parameter("save_every_n_successes").value)

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

        self.last_rvec = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.last_tvec = {
            self.camera_1_name: None,
            self.camera_2_name: None,
        }

        self.last_used_ids = {
            self.camera_1_name: [],
            self.camera_2_name: [],
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

        self.create_timer(self.process_period_sec, self.process_pair)

        self.get_logger().info("ArUco Rig Estimator started.")
        self.get_logger().info(f"Config: {self.config_path}")
        self.get_logger().info(f"ArUco config: {self.aruco_config_path}")
        self.get_logger().info(
            f"Board: {self.markers_x}x{self.markers_y}, "
            f"marker_length={self.marker_length}, separation={self.marker_separation}, "
            f"dictionary={self.dictionary_name}"
        )
        self.get_logger().info(f"{self.camera_1_name}: image={self.cam1_image_topic}, camera_info={self.cam1_info_topic}")
        self.get_logger().info(f"{self.camera_2_name}: image={self.cam2_image_topic}, camera_info={self.cam2_info_topic}")
        self.get_logger().info(f"Expected physical camera baseline: {self.expected_baseline_m:.3f} m")
        self.get_logger().info(f"show_gui: {self.show_gui}")
        self.get_logger().info(f"save_debug: {self.save_debug}")
        self.get_logger().info(f"debug_dir: {self.debug_dir}")

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
        3D coordinates of one marker's four corners in the board coordinate frame.

        OpenCV ArUco corner order:
        top-left, top-right, bottom-right, bottom-left.

        Board convention:
        marker id 0 is top-left, then row-major order.
        """
        if marker_id < 0 or marker_id >= self.markers_x * self.markers_y:
            return None

        row = marker_id // self.markers_x
        col = marker_id % self.markers_x

        step = self.marker_length + self.marker_separation

        # Use an OpenCV-friendly board coordinate convention:
        # x increases to the right, y increases upward.
        # Image/texture rows go downward, therefore physical board y decreases with row.
        x0 = col * step
        y0 = -row * step
        L = self.marker_length

        # OpenCV ArUco corner order:
        # top-left, top-right, bottom-right, bottom-left.
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

    def annotate_image(self, camera_name, ok):
        image = self.latest_images[camera_name]

        if image is None:
            return None

        annotated = image.copy()

        corners = self.last_corners[camera_name]
        ids = self.last_ids[camera_name]

        if ids is not None and corners is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        rvec = self.last_rvec[camera_name]
        tvec = self.last_tvec[camera_name]
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if ok and rvec is not None and tvec is not None:
            cv2.drawFrameAxes(
                annotated,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                self.marker_length * 1.5,
            )

        used_ids = self.last_used_ids[camera_name]
        status = self.last_status[camera_name]

        label = f"{camera_name} | {status} | markers={len(used_ids)}"

        color = (0, 255, 0) if ok else (0, 0, 255)

        cv2.putText(
            annotated,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        return annotated

    def save_debug_images(self, ok1, ok2):
        if not self.save_debug:
            return

        if self.success_counter % self.save_every_n_successes != 0:
            return

        for camera_name, ok in [
            (self.camera_1_name, ok1),
            (self.camera_2_name, ok2),
        ]:
            annotated = self.annotate_image(camera_name, ok)

            if annotated is None:
                continue

            filename = (
                f"success_{self.success_counter:04d}_"
                f"{camera_name}_{self.last_status[camera_name].replace('/', '_')}.png"
            )

            path = os.path.join(self.debug_dir, filename)
            cv2.imwrite(path, annotated)

    def update_gui(self, ok1, ok2):
        if not self.show_gui:
            return

        img1 = self.annotate_image(self.camera_1_name, ok1)
        img2 = self.annotate_image(self.camera_2_name, ok2)

        if img1 is not None:
            cv2.imshow("aruco_rig_estimator_camera_1", img1)

        if img2 is not None:
            cv2.imshow("aruco_rig_estimator_camera_2", img2)

        cv2.waitKey(1)

    def process_pair(self):
        now = time.time()

        if now - self.last_process_time < self.process_period_sec:
            return

        self.last_process_time = now
        self.frame_counter += 1

        ok1, T_cam1_target, tvec1, method1 = self.estimate_target_pose_for_camera(
            self.camera_1_name
        )

        ok2, T_cam2_target, tvec2, method2 = self.estimate_target_pose_for_camera(
            self.camera_2_name
        )

        self.update_gui(ok1, ok2)

        if not ok1 or not ok2:
            if self.frame_counter % 5 == 0:
                self.get_logger().warn(
                    f"Waiting for valid ArUco pair | "
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

        tvec_delta = np.asarray(tvec2) - np.asarray(tvec1)
        translation_only_baseline = float(np.linalg.norm(tvec_delta))
        translation_only_error = abs(
            translation_only_baseline - self.expected_baseline_m
        )

        self.save_debug_images(ok1, ok2)

        self.get_logger().info(
            "\n"
            "================ ARUCO RIG ESTIMATE ================\n"
            f"valid_pair={self.success_counter}/{self.frame_counter}\n"
            f"{self.camera_1_name}: method={method1}, markers={len(self.last_used_ids[self.camera_1_name])}, "
            f"ids={self.last_used_ids[self.camera_1_name]}, t_cam_target={format_vector(tvec1)} m\n"
            f"{self.camera_2_name}: method={method2}, markers={len(self.last_used_ids[self.camera_2_name])}, "
            f"ids={self.last_used_ids[self.camera_2_name]}, t_cam_target={format_vector(tvec2)} m\n"
            "\n"
            "Estimated relative transform from common ArUco target:\n"
            f"T_{self.camera_1_name}_{self.camera_2_name} translation = "
            f"{format_vector(estimated_translation)} m\n"
            f"full-transform baseline norm  = {estimated_baseline:.4f} m\n"
            f"expected baseline norm        = {self.expected_baseline_m:.4f} m\n"
            f"full-transform baseline error = {baseline_error:.4f} m "
            f"({baseline_error * 100.0:.2f} cm)\n"
            f"translation-only baseline     = {translation_only_baseline:.4f} m\n"
            f"translation-only error        = {translation_only_error:.4f} m "
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
        if node.show_gui:
            cv2.destroyAllWindows()

        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
