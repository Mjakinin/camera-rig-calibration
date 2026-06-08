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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ArucoLiveDetector(Node):
    def __init__(self):
        super().__init__("aruco_live_detector")

        self.declare_parameter("aruco_config_path", "src/calib_lab/config/aruco_target.yaml")
        self.declare_parameter("camera_1_image_topic", "/camera_1/image")
        self.declare_parameter("camera_2_image_topic", "/camera_2/image")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "results/minimal_world/aruco/live_detector/debug_images")
        self.declare_parameter("save_every_n_frames", 30)

        self.aruco_config_path = self.get_parameter("aruco_config_path").value
        self.camera_1_image_topic = self.get_parameter("camera_1_image_topic").value
        self.camera_2_image_topic = self.get_parameter("camera_2_image_topic").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.save_debug = bool(self.get_parameter("save_debug").value)
        self.debug_dir = self.get_parameter("debug_dir").value
        self.save_every_n_frames = int(self.get_parameter("save_every_n_frames").value)

        os.makedirs(self.debug_dir, exist_ok=True)

        self.bridge = CvBridge()

        self.frame_counter = {
            "camera_1": 0,
            "camera_2": 0,
        }

        self.last_log_time = {
            "camera_1": 0.0,
            "camera_2": 0.0,
        }

        self.aruco_cfg = self.load_config(self.aruco_config_path)["aruco"]
        self.dictionary_name = self.aruco_cfg["dictionary"]
        self.dictionary = self.get_dictionary(self.dictionary_name)
        self.detector = self.create_detector()

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

        self.get_logger().info("ArUco Live Detector started.")
        self.get_logger().info(f"ArUco config: {self.aruco_config_path}")
        self.get_logger().info(f"Dictionary: {self.dictionary_name}")
        self.get_logger().info(f"camera_1 image topic: {self.camera_1_image_topic}")
        self.get_logger().info(f"camera_2 image topic: {self.camera_2_image_topic}")
        self.get_logger().info(f"show_gui: {self.show_gui}")
        self.get_logger().info(f"save_debug: {self.save_debug}")
        self.get_logger().info(f"debug_dir: {self.debug_dir}")

    def load_config(self, path):
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

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary)

        return corners, ids, rejected

    def annotate(self, camera_name, image, corners, ids):
        annotated = image.copy()

        num_markers = 0
        ids_text = "none"

        if ids is not None and len(ids) > 0:
            num_markers = len(ids)
            ids_flat = ids.flatten().astype(int).tolist()
            ids_text = ",".join(str(x) for x in ids_flat[:12])

            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        label = f"{camera_name} | markers={num_markers} | ids={ids_text}"

        color = (0, 255, 0) if num_markers > 0 else (0, 0, 255)

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

        return annotated, num_markers, ids_text

    def image_callback(self, camera_name, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")
            return

        self.frame_counter[camera_name] += 1

        corners, ids, rejected = self.detect_markers(image)
        annotated, num_markers, ids_text = self.annotate(camera_name, image, corners, ids)

        now = time.time()
        if now - self.last_log_time[camera_name] > 1.0:
            self.last_log_time[camera_name] = now
            self.get_logger().info(
                f"{camera_name}: frame={self.frame_counter[camera_name]} | "
                f"markers={num_markers} | ids={ids_text}"
            )

        if self.save_debug and self.frame_counter[camera_name] % self.save_every_n_frames == 0:
            out_path = os.path.join(
                self.debug_dir,
                f"{camera_name}_frame_{self.frame_counter[camera_name]:06d}_markers_{num_markers}.png",
            )
            cv2.imwrite(out_path, annotated)

        if self.show_gui:
            cv2.imshow(f"aruco_live_detector_{camera_name}", annotated)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoLiveDetector()

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
