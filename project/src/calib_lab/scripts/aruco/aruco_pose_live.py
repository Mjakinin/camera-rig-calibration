#!/usr/bin/env python3

import os
import time
import yaml
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class ArucoPoseLive(Node):
    def __init__(self):
        super().__init__("aruco_pose_live")

        self.declare_parameter("aruco_config_path", "src/calib_lab/config/aruco_target.yaml")
        self.declare_parameter("camera_1_image_topic", "/camera_1/image")
        self.declare_parameter("camera_1_info_topic", "/camera_1/camera_info")
        self.declare_parameter("camera_2_image_topic", "/camera_2/image")
        self.declare_parameter("camera_2_info_topic", "/camera_2/camera_info")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "results/aruco/pose_live/debug_images")
        self.declare_parameter("save_every_n_frames", 30)
        self.declare_parameter("min_markers", 1)

        self.aruco_config_path = self.get_parameter("aruco_config_path").value
        self.camera_1_image_topic = self.get_parameter("camera_1_image_topic").value
        self.camera_1_info_topic = self.get_parameter("camera_1_info_topic").value
        self.camera_2_image_topic = self.get_parameter("camera_2_image_topic").value
        self.camera_2_info_topic = self.get_parameter("camera_2_info_topic").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.save_debug = bool(self.get_parameter("save_debug").value)
        self.debug_dir = self.get_parameter("debug_dir").value
        self.save_every_n_frames = int(self.get_parameter("save_every_n_frames").value)
        self.min_markers = int(self.get_parameter("min_markers").value)

        os.makedirs(self.debug_dir, exist_ok=True)

        self.bridge = CvBridge()

        self.aruco_cfg = self.load_yaml(self.aruco_config_path)["aruco"]

        self.dictionary_name = self.aruco_cfg["dictionary"]
        self.markers_x = int(self.aruco_cfg["markers_x"])
        self.markers_y = int(self.aruco_cfg["markers_y"])
        self.marker_length = float(self.aruco_cfg["marker_length"])
        self.marker_separation = float(self.aruco_cfg["marker_separation"])

        self.dictionary = self.get_dictionary(self.dictionary_name)
        self.detector = self.create_detector()

        self.camera_matrices = {
            "camera_1": None,
            "camera_2": None,
        }

        self.dist_coeffs = {
            "camera_1": None,
            "camera_2": None,
        }

        self.frame_counter = {
            "camera_1": 0,
            "camera_2": 0,
        }

        self.success_counter = {
            "camera_1": 0,
            "camera_2": 0,
        }

        self.last_log_time = {
            "camera_1": 0.0,
            "camera_2": 0.0,
        }

        self.create_subscription(
            CameraInfo,
            self.camera_1_info_topic,
            lambda msg: self.camera_info_callback("camera_1", msg),
            10,
        )

        self.create_subscription(
            CameraInfo,
            self.camera_2_info_topic,
            lambda msg: self.camera_info_callback("camera_2", msg),
            10,
        )

        self.create_subscription(
            Image,
            self.camera_1_image_topic,
            lambda msg: self.image_callback("camera_1", msg),
            10,
        )

        self.create_subscription(
            Image,
            self.camera_2_image_topic,
            lambda msg: self.image_callback("camera_2", msg),
            10,
        )

        self.get_logger().info("ArUco Pose Live started.")
        self.get_logger().info(f"ArUco config: {self.aruco_config_path}")
        self.get_logger().info(
            f"Board: {self.markers_x}x{self.markers_y}, "
            f"marker_length={self.marker_length}, separation={self.marker_separation}, "
            f"dictionary={self.dictionary_name}"
        )
        self.get_logger().info(f"show_gui: {self.show_gui}")
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

    def marker_object_corners(self, marker_id):
        """
        Returns the 3D marker corner coordinates in the local ArUco board frame.

        Marker ID convention:
        IDs are generated by OpenCV GridBoard row-major:
        top-left marker = id 0
        next marker in row = id 1
        ...
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

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary)

        return corners, ids, rejected

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

    def estimate_pose(self, camera_name, image):
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        corners, ids, rejected = self.detect_markers(image)

        if ids is None or len(ids) == 0:
            return False, None, None, corners, ids, [], "aruco_not_found"

        if camera_matrix is None or dist_coeffs is None:
            return False, None, None, corners, ids, [], "no_camera_info"

        object_points, image_points, used_ids = self.build_correspondences(corners, ids)

        if object_points is None or image_points is None:
            return False, None, None, corners, ids, [], "no_known_marker_ids"

        if len(used_ids) < self.min_markers:
            return False, None, None, corners, ids, used_ids, "not_enough_markers"

        if len(object_points) < 4:
            return False, None, None, corners, ids, used_ids, "not_enough_points"

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return False, None, None, corners, ids, used_ids, "solvepnp_failed"

        return True, rvec, tvec, corners, ids, used_ids, "pose_found"

    def annotate(self, camera_name, image, ok, rvec, tvec, corners, ids, used_ids, status):
        annotated = image.copy()

        if ids is not None and corners is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if ok and camera_matrix is not None and dist_coeffs is not None:
            axis_length = self.marker_length * 1.5
            cv2.drawFrameAxes(
                annotated,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                axis_length,
            )

        marker_count = len(used_ids)
        ids_text = ",".join(str(x) for x in used_ids[:12]) if used_ids else "none"

        if ok:
            t = tvec.flatten()
            dist = float(np.linalg.norm(t))
            label = (
                f"{camera_name} | POSE | markers={marker_count} | "
                f"t=[{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}] m | dist={dist:.2f}"
            )
            color = (0, 255, 0)
        else:
            label = f"{camera_name} | {status} | markers={marker_count} | ids={ids_text}"
            color = (0, 0, 255)

        cv2.putText(
            annotated,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

        return annotated

    def image_callback(self, camera_name, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")
            return

        self.frame_counter[camera_name] += 1

        ok, rvec, tvec, corners, ids, used_ids, status = self.estimate_pose(camera_name, image)

        annotated = self.annotate(
            camera_name,
            image,
            ok,
            rvec,
            tvec,
            corners,
            ids,
            used_ids,
            status,
        )

        now = time.time()

        if ok:
            self.success_counter[camera_name] += 1
            t = tvec.flatten()
            dist = float(np.linalg.norm(t))

            if now - self.last_log_time[camera_name] > 1.0:
                self.last_log_time[camera_name] = now
                self.get_logger().info(
                    f"POSE FOUND | {camera_name} | "
                    f"frame={self.frame_counter[camera_name]} | "
                    f"markers={len(used_ids)} | "
                    f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m | "
                    f"dist={dist:.3f} m | "
                    f"success={self.success_counter[camera_name]}/{self.frame_counter[camera_name]}"
                )
        else:
            if now - self.last_log_time[camera_name] > 1.0:
                self.last_log_time[camera_name] = now
                self.get_logger().warn(
                    f"POSE NOT FOUND | {camera_name} | "
                    f"frame={self.frame_counter[camera_name]} | "
                    f"status={status} | markers={len(used_ids)}"
                )

        if self.save_debug and self.frame_counter[camera_name] % self.save_every_n_frames == 0:
            out_path = os.path.join(
                self.debug_dir,
                f"{camera_name}_frame_{self.frame_counter[camera_name]:06d}_{status}.png",
            )
            cv2.imwrite(out_path, annotated)

        if self.show_gui:
            cv2.imshow(f"aruco_pose_live_{camera_name}", annotated)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoPoseLive()

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
