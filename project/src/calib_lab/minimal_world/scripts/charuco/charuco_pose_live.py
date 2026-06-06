#!/usr/bin/env python3

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
    from charuco_common import (
        get_dictionary,
        create_detector,
        create_charuco_board,
        detect_charuco,
        build_charuco_correspondences,
        draw_charuco_detection,
    )
except ImportError:
    from charuco.charuco_common import (
        get_dictionary,
        create_detector,
        create_charuco_board,
        detect_charuco,
        build_charuco_correspondences,
        draw_charuco_detection,
    )


class CharucoPoseLive(Node):
    def __init__(self):
        super().__init__("charuco_pose_live")

        self.declare_parameter("charuco_config_path", "src/calib_lab/config/charuco_target.yaml")
        self.declare_parameter("camera_1_image_topic", "/camera_1/image")
        self.declare_parameter("camera_1_info_topic", "/camera_1/camera_info")
        self.declare_parameter("camera_2_image_topic", "/camera_2/image")
        self.declare_parameter("camera_2_info_topic", "/camera_2/camera_info")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "results/charuco/pose_live/debug_images")
        self.declare_parameter("save_every_n_frames", 30)
        self.declare_parameter("min_markers", 1)
        self.declare_parameter("min_charuco_corners", 4)

        self.charuco_config_path = self.get_parameter("charuco_config_path").value
        self.camera_1_image_topic = self.get_parameter("camera_1_image_topic").value
        self.camera_1_info_topic = self.get_parameter("camera_1_info_topic").value
        self.camera_2_image_topic = self.get_parameter("camera_2_image_topic").value
        self.camera_2_info_topic = self.get_parameter("camera_2_info_topic").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.save_debug = bool(self.get_parameter("save_debug").value)
        self.debug_dir = self.get_parameter("debug_dir").value
        self.save_every_n_frames = int(self.get_parameter("save_every_n_frames").value)
        self.min_markers = int(self.get_parameter("min_markers").value)
        self.min_charuco_corners = int(self.get_parameter("min_charuco_corners").value)

        os.makedirs(self.debug_dir, exist_ok=True)

        self.bridge = CvBridge()
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

        self.camera_matrices = {"camera_1": None, "camera_2": None}
        self.dist_coeffs = {"camera_1": None, "camera_2": None}
        self.frame_counter = {"camera_1": 0, "camera_2": 0}
        self.success_counter = {"camera_1": 0, "camera_2": 0}
        self.last_log_time = {"camera_1": 0.0, "camera_2": 0.0}

        self.create_subscription(CameraInfo, self.camera_1_info_topic, lambda msg: self.camera_info_callback("camera_1", msg), 10)
        self.create_subscription(CameraInfo, self.camera_2_info_topic, lambda msg: self.camera_info_callback("camera_2", msg), 10)
        self.create_subscription(Image, self.camera_1_image_topic, lambda msg: self.image_callback("camera_1", msg), 10)
        self.create_subscription(Image, self.camera_2_image_topic, lambda msg: self.image_callback("camera_2", msg), 10)

        self.get_logger().info("ChArUco Pose Live started.")
        self.get_logger().info(f"ChArUco config: {self.charuco_config_path}")
        self.get_logger().info(
            f"Board: {self.squares_x}x{self.squares_y}, square_length={self.square_length}, "
            f"marker_length={self.marker_length}, dictionary={self.dictionary_name}"
        )
        self.get_logger().info(f"min_markers: {self.min_markers}, min_charuco_corners: {self.min_charuco_corners}")
        self.get_logger().info(f"show_gui: {self.show_gui}")
        self.get_logger().info(f"debug_dir: {self.debug_dir}")

    def load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def camera_info_callback(self, camera_name, msg):
        self.camera_matrices[camera_name] = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs[camera_name] = np.array(msg.d, dtype=np.float64)

    def estimate_pose(self, camera_name, image):
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if camera_matrix is None or dist_coeffs is None:
            return False, None, None, None, [], [], "no_camera_info"

        detection = detect_charuco(
            image,
            self.dictionary,
            self.detector,
            self.board,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            use_preprocessing=True,
        )

        marker_ids = [] if detection["ids"] is None else detection["ids"].flatten().astype(int).tolist()
        if detection["marker_count"] < self.min_markers:
            return False, None, None, detection, marker_ids, [], f"not_enough_markers/{detection['method']}"

        if detection["charuco_count"] < self.min_charuco_corners:
            return False, None, None, detection, marker_ids, [], f"not_enough_charuco_corners/{detection['method']}"

        object_points, image_points, used_charuco_ids = build_charuco_correspondences(
            self.board,
            detection["charuco_corners"],
            detection["charuco_ids"],
        )

        if object_points is None or image_points is None:
            return False, None, None, detection, marker_ids, [], f"no_charuco_correspondences/{detection['method']}"

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return False, None, None, detection, marker_ids, used_charuco_ids, f"solvepnp_failed/{detection['method']}"

        return True, rvec, tvec, detection, marker_ids, used_charuco_ids, f"charuco/{detection['method']}"

    def image_callback(self, camera_name, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")
            return

        self.frame_counter[camera_name] += 1
        ok, rvec, tvec, detection, marker_ids, used_charuco_ids, status = self.estimate_pose(camera_name, image)

        annotated = image.copy()
        if detection is not None:
            annotated = draw_charuco_detection(
                image,
                detection,
                camera_matrix=self.camera_matrices[camera_name],
                dist_coeffs=self.dist_coeffs[camera_name],
                rvec=rvec if ok else None,
                tvec=tvec if ok else None,
                axis_length=self.square_length * 1.5,
            )

        marker_count = 0 if detection is None else detection["marker_count"]
        charuco_count = 0 if detection is None else detection["charuco_count"]
        label = f"{camera_name} | {status} | markers={marker_count} | charuco={charuco_count}"
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        now = time.time()
        if ok:
            self.success_counter[camera_name] += 1
            t = tvec.flatten()
            dist = float(np.linalg.norm(t))
            if now - self.last_log_time[camera_name] > 1.0:
                self.last_log_time[camera_name] = now
                self.get_logger().info(
                    f"POSE FOUND | {camera_name} | frame={self.frame_counter[camera_name]} | "
                    f"markers={marker_count} | charuco_corners={charuco_count} | "
                    f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m | dist={dist:.3f} m | "
                    f"success={self.success_counter[camera_name]}/{self.frame_counter[camera_name]}"
                )
        else:
            if now - self.last_log_time[camera_name] > 1.0:
                self.last_log_time[camera_name] = now
                self.get_logger().warn(
                    f"POSE NOT FOUND | {camera_name} | frame={self.frame_counter[camera_name]} | "
                    f"status={status} | markers={marker_count} | charuco_corners={charuco_count}"
                )

        if self.save_debug and self.frame_counter[camera_name] % self.save_every_n_frames == 0:
            safe_status = status.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(self.debug_dir, f"{camera_name}_frame_{self.frame_counter[camera_name]:06d}_{safe_status}.png")
            cv2.imwrite(out_path, annotated)

        if self.show_gui:
            cv2.imshow(f"charuco_pose_live_{camera_name}", annotated)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = CharucoPoseLive()
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
