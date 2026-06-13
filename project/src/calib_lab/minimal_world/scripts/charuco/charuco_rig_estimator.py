#!/usr/bin/env python3

from pathlib import Path
import sys

CALIB_LAB_DIR = Path(__file__).resolve().parents[3]
if str(CALIB_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_LAB_DIR))


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

try:
    from common.transform_utils import (
        rvec_tvec_to_matrix,
        relative_transform_from_common_target,
        translation_norm,
        relative_rotation_angle_deg,
        format_vector,
    )
except ImportError:
    from transform_utils import (
        rvec_tvec_to_matrix,
        relative_transform_from_common_target,
        translation_norm,
        relative_rotation_angle_deg,
        format_vector,
    )

try:
    from charuco_common import (
        get_dictionary,
        create_detector,
        create_charuco_board,
        detect_charuco,
        build_charuco_correspondences,
        draw_charuco_detection,
        estimate_charuco_pose_native,
    )
except ImportError:
    from charuco.charuco_common import (
        get_dictionary,
        create_detector,
        create_charuco_board,
        detect_charuco,
        build_charuco_correspondences,
        draw_charuco_detection,
        estimate_charuco_pose_native,
    )


class CharucoRigEstimator(Node):
    def __init__(self):
        super().__init__("charuco_rig_estimator")

        self.declare_parameter("config_path", "src/calib_lab/minimal_world/config/ground_truth_minimal.yaml")
        self.declare_parameter("charuco_config_path", "src/calib_lab/minimal_world/config/charuco_target.yaml")
        self.declare_parameter("process_period_sec", 0.5)
        self.declare_parameter("min_markers", 1)
        self.declare_parameter("min_charuco_corners", 4)
        self.declare_parameter("show_gui", True)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "results/charuco/rig_estimator/debug_images")
        self.declare_parameter("save_every_n_successes", 10)

        self.config_path = self.get_parameter("config_path").value
        self.charuco_config_path = self.get_parameter("charuco_config_path").value
        self.process_period_sec = float(self.get_parameter("process_period_sec").value)
        self.min_markers = int(self.get_parameter("min_markers").value)
        self.min_charuco_corners = int(self.get_parameter("min_charuco_corners").value)
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.save_debug = bool(self.get_parameter("save_debug").value)
        self.debug_dir = self.get_parameter("debug_dir").value
        self.save_every_n_successes = int(self.get_parameter("save_every_n_successes").value)

        os.makedirs(self.debug_dir, exist_ok=True)

        self.config = self.load_yaml(self.config_path)
        self.charuco_cfg = self.load_yaml(self.charuco_config_path)["charuco"]

        self.dictionary_name = self.charuco_cfg["dictionary"]
        self.squares_x = int(self.charuco_cfg["squares_x"])
        self.squares_y = int(self.charuco_cfg["squares_y"])
        self.square_length = float(self.charuco_cfg["square_length"])
        self.marker_length = float(self.charuco_cfg["marker_length"])
        self.min_markers = int(self.charuco_cfg.get("min_markers", self.min_markers))
        self.min_charuco_corners = int(self.charuco_cfg.get("min_charuco_corners", self.min_charuco_corners))

        self.dictionary = get_dictionary(self.dictionary_name)
        self.detector = create_detector(self.dictionary)
        self.board = create_charuco_board(
            self.dictionary,
            self.squares_x,
            self.squares_y,
            self.square_length,
            self.marker_length,
        )

        self.camera_1_name = "camera_1"
        self.camera_2_name = "camera_2"

        cam1_cfg = self.config["cameras"][self.camera_1_name]
        cam2_cfg = self.config["cameras"][self.camera_2_name]

        self.cam1_image_topic = cam1_cfg["image_topic"]
        self.cam1_info_topic = cam1_cfg["camera_info_topic"]
        self.cam2_image_topic = cam2_cfg["image_topic"]
        self.cam2_info_topic = cam2_cfg["camera_info_topic"]

        self.expected_baseline_m = float(self.config["ground_truth"]["expected_baseline_m"])
        self.expected_relative_rotation_deg = float(self.config["ground_truth"]["expected_relative_rotation_deg"])

        self.bridge = CvBridge()
        self.latest_images = {self.camera_1_name: None, self.camera_2_name: None}
        self.camera_matrices = {self.camera_1_name: None, self.camera_2_name: None}
        self.dist_coeffs = {self.camera_1_name: None, self.camera_2_name: None}

        self.last_status = {self.camera_1_name: "not_started", self.camera_2_name: "not_started"}
        self.last_detection = {self.camera_1_name: None, self.camera_2_name: None}
        self.last_marker_ids = {self.camera_1_name: [], self.camera_2_name: []}
        self.last_charuco_ids = {self.camera_1_name: [], self.camera_2_name: []}
        self.last_rvec = {self.camera_1_name: None, self.camera_2_name: None}
        self.last_tvec = {self.camera_1_name: None, self.camera_2_name: None}

        self.frame_counter = 0
        self.success_counter = 0
        self.last_process_time = 0.0

        self.create_subscription(CameraInfo, self.cam1_info_topic, lambda msg: self.camera_info_callback(self.camera_1_name, msg), 10)
        self.create_subscription(CameraInfo, self.cam2_info_topic, lambda msg: self.camera_info_callback(self.camera_2_name, msg), 10)
        self.create_subscription(Image, self.cam1_image_topic, lambda msg: self.image_callback(self.camera_1_name, msg), 10)
        self.create_subscription(Image, self.cam2_image_topic, lambda msg: self.image_callback(self.camera_2_name, msg), 10)
        self.create_timer(self.process_period_sec, self.process_pair)

        self.get_logger().info("ChArUco Rig Estimator started.")
        self.get_logger().info(f"Config: {self.config_path}")
        self.get_logger().info(f"ChArUco config: {self.charuco_config_path}")
        self.get_logger().info(
            f"Board: {self.squares_x}x{self.squares_y}, square_length={self.square_length}, "
            f"marker_length={self.marker_length}, dictionary={self.dictionary_name}"
        )
        self.get_logger().info(f"min_markers: {self.min_markers}, min_charuco_corners: {self.min_charuco_corners}")
        self.get_logger().info(f"{self.camera_1_name}: image={self.cam1_image_topic}, camera_info={self.cam1_info_topic}")
        self.get_logger().info(f"{self.camera_2_name}: image={self.cam2_image_topic}, camera_info={self.cam2_info_topic}")
        self.get_logger().info(f"Expected physical camera baseline: {self.expected_baseline_m:.3f} m")

    def load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def camera_info_callback(self, camera_name, msg):
        self.camera_matrices[camera_name] = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs[camera_name] = np.array(msg.d, dtype=np.float64)

    def image_callback(self, camera_name, msg):
        try:
            self.latest_images[camera_name] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")

    def estimate_target_pose_for_camera(self, camera_name):
        image = self.latest_images[camera_name]
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        self.last_rvec[camera_name] = None
        self.last_tvec[camera_name] = None
        self.last_marker_ids[camera_name] = []
        self.last_charuco_ids[camera_name] = []
        self.last_detection[camera_name] = None

        if image is None:
            self.last_status[camera_name] = "no_image"
            return False, None, None, "no_image"

        if camera_matrix is None or dist_coeffs is None:
            self.last_status[camera_name] = "no_camera_info"
            return False, None, None, "no_camera_info"

        detection = detect_charuco(
            image,
            self.dictionary,
            self.detector,
            self.board,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            use_preprocessing=True,
        )
        self.last_detection[camera_name] = detection

        marker_ids = [] if detection["ids"] is None else detection["ids"].flatten().astype(int).tolist()
        charuco_ids = [] if detection["charuco_ids"] is None else detection["charuco_ids"].flatten().astype(int).tolist()
        self.last_marker_ids[camera_name] = marker_ids
        self.last_charuco_ids[camera_name] = charuco_ids

        if detection["marker_count"] < self.min_markers:
            self.last_status[camera_name] = f"not_enough_markers/{detection['method']}"
            return False, None, None, self.last_status[camera_name]

        if detection["charuco_count"] < self.min_charuco_corners:
            self.last_status[camera_name] = f"not_enough_charuco_corners/{detection['method']}"
            return False, None, None, self.last_status[camera_name]

        object_points, image_points, used_charuco_ids = build_charuco_correspondences(
            self.board,
            detection["charuco_corners"],
            detection["charuco_ids"],
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            self.last_status[camera_name] = f"solvepnp_failed/{detection['method']}"
            return False, None, None, self.last_status[camera_name]

        self.last_status[camera_name] = f"charuco_matchpoints/{detection['method']}"
        self.last_rvec[camera_name] = rvec
        self.last_tvec[camera_name] = tvec
        T_cam_target = rvec_tvec_to_matrix(rvec, tvec)
        return True, T_cam_target, tvec.flatten(), self.last_status[camera_name]

    def annotate_image(self, camera_name, ok):
        image = self.latest_images[camera_name]
        if image is None:
            return None

        detection = self.last_detection[camera_name]
        if detection is None:
            annotated = image.copy()
        else:
            annotated = draw_charuco_detection(
                image,
                detection,
                camera_matrix=self.camera_matrices[camera_name],
                dist_coeffs=self.dist_coeffs[camera_name],
                rvec=self.last_rvec[camera_name] if ok else None,
                tvec=self.last_tvec[camera_name] if ok else None,
                axis_length=self.square_length * 1.5,
            )

        status = self.last_status[camera_name]
        marker_count = len(self.last_marker_ids[camera_name])
        charuco_count = len(self.last_charuco_ids[camera_name])
        label = f"{camera_name} | {status} | markers={marker_count} | charuco={charuco_count}"
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return annotated

    def save_debug_images(self, ok1, ok2):
        if not self.save_debug:
            return
        if self.success_counter % self.save_every_n_successes != 0:
            return
        for camera_name, ok in [(self.camera_1_name, ok1), (self.camera_2_name, ok2)]:
            annotated = self.annotate_image(camera_name, ok)
            if annotated is None:
                continue
            filename = f"success_{self.success_counter:04d}_{camera_name}_{self.last_status[camera_name].replace('/', '_')}.png"
            cv2.imwrite(os.path.join(self.debug_dir, filename), annotated)

    def update_gui(self, ok1, ok2):
        if not self.show_gui:
            return
        img1 = self.annotate_image(self.camera_1_name, ok1)
        img2 = self.annotate_image(self.camera_2_name, ok2)
        if img1 is not None:
            cv2.imshow("charuco_rig_estimator_camera_1", img1)
        if img2 is not None:
            cv2.imshow("charuco_rig_estimator_camera_2", img2)
        cv2.waitKey(1)

    def process_pair(self):
        now = time.time()
        if now - self.last_process_time < self.process_period_sec:
            return

        self.last_process_time = now
        self.frame_counter += 1

        ok1, T_cam1_target, tvec1, method1 = self.estimate_target_pose_for_camera(self.camera_1_name)
        ok2, T_cam2_target, tvec2, method2 = self.estimate_target_pose_for_camera(self.camera_2_name)

        self.update_gui(ok1, ok2)

        if not ok1 or not ok2:
            if self.frame_counter % 5 == 0:
                self.get_logger().warn(
                    f"Waiting for valid ChArUco pair | {self.camera_1_name}: {method1} | {self.camera_2_name}: {method2}"
                )
            return

        self.success_counter += 1

        T_cam1_cam2 = relative_transform_from_common_target(T_cam1_target, T_cam2_target)
        estimated_translation = T_cam1_cam2[:3, 3]
        estimated_baseline = translation_norm(T_cam1_cam2)
        estimated_rotation_deg = relative_rotation_angle_deg(T_cam1_cam2)
        baseline_error = abs(estimated_baseline - self.expected_baseline_m)
        rotation_error = abs(estimated_rotation_deg - self.expected_relative_rotation_deg)

        tvec_delta = np.asarray(tvec2) - np.asarray(tvec1)
        translation_only_baseline = float(np.linalg.norm(tvec_delta))
        translation_only_error = abs(translation_only_baseline - self.expected_baseline_m)

        self.save_debug_images(ok1, ok2)

        self.get_logger().info(
            "\n"
            "================ CHARUCO RIG ESTIMATE ================\n"
            f"valid_pair={self.success_counter}/{self.frame_counter}\n"
            f"{self.camera_1_name}: method={method1}, markers={len(self.last_marker_ids[self.camera_1_name])}, "
            f"charuco_corners={len(self.last_charuco_ids[self.camera_1_name])}, "
            f"t_cam_target={format_vector(tvec1)} m\n"
            f"{self.camera_2_name}: method={method2}, markers={len(self.last_marker_ids[self.camera_2_name])}, "
            f"charuco_corners={len(self.last_charuco_ids[self.camera_2_name])}, "
            f"t_cam_target={format_vector(tvec2)} m\n"
            "\n"
            "Estimated relative transform from common ChArUco target:\n"
            f"T_{self.camera_1_name}_{self.camera_2_name} translation = {format_vector(estimated_translation)} m\n"
            f"full-transform baseline norm  = {estimated_baseline:.4f} m\n"
            f"expected baseline norm        = {self.expected_baseline_m:.4f} m\n"
            f"full-transform baseline error = {baseline_error:.4f} m ({baseline_error * 100.0:.2f} cm)\n"
            f"translation-only baseline     = {translation_only_baseline:.4f} m\n"
            f"translation-only error        = {translation_only_error:.4f} m ({translation_only_error * 100.0:.2f} cm)\n"
            f"estimated relative rotation angle = {estimated_rotation_deg:.4f} deg\n"
            f"expected relative rotation angle  = {self.expected_relative_rotation_deg:.4f} deg\n"
            f"rotation angle error              = {rotation_error:.4f} deg\n"
            "======================================================\n"
        )


def main():
    rclpy.init()
    node = CharucoRigEstimator()
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
