#!/usr/bin/env python3

import os
import time
import yaml
from pathlib import Path
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from charuco_common import (
        get_dictionary,
        create_detector,
        create_charuco_board,
        detect_charuco,
        draw_charuco_detection,
    )
except ImportError:
    from charuco.charuco_common import (
        get_dictionary,
        create_detector,
        create_charuco_board,
        detect_charuco,
        draw_charuco_detection,
    )


class CharucoLiveDetector(Node):
    def __init__(self):
        super().__init__("charuco_live_detector")

        self.declare_parameter("charuco_config_path", "src/calib_lab/minimal_world/config/charuco_target.yaml")
        self.declare_parameter("camera_1_image_topic", "/camera_1/image")
        self.declare_parameter("camera_2_image_topic", "/camera_2/image")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "results/charuco/live_detector/debug_images")
        self.declare_parameter("save_every_n_frames", 30)

        self.charuco_config_path = self.get_parameter("charuco_config_path").value
        self.camera_1_image_topic = self.get_parameter("camera_1_image_topic").value
        self.camera_2_image_topic = self.get_parameter("camera_2_image_topic").value
        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.save_debug = bool(self.get_parameter("save_debug").value)
        self.debug_dir = self.get_parameter("debug_dir").value
        self.save_every_n_frames = int(self.get_parameter("save_every_n_frames").value)

        os.makedirs(self.debug_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.frame_counter = {"camera_1": 0, "camera_2": 0}
        self.last_log_time = {"camera_1": 0.0, "camera_2": 0.0}

        self.charuco_cfg = self.load_yaml(self.charuco_config_path)["charuco"]
        self.dictionary_name = self.charuco_cfg["dictionary"]
        self.squares_x = int(self.charuco_cfg["squares_x"])
        self.squares_y = int(self.charuco_cfg["squares_y"])
        self.square_length = float(self.charuco_cfg["square_length"])
        self.marker_length = float(self.charuco_cfg["marker_length"])

        self.dictionary = get_dictionary(self.dictionary_name)
        self.detector = create_detector(self.dictionary)
        self.board = create_charuco_board(
            self.dictionary,
            self.squares_x,
            self.squares_y,
            self.square_length,
            self.marker_length,
        )

        self.create_subscription(Image, self.camera_1_image_topic, lambda msg: self.image_callback("camera_1", msg), 10)
        self.create_subscription(Image, self.camera_2_image_topic, lambda msg: self.image_callback("camera_2", msg), 10)

        self.get_logger().info("ChArUco Live Detector started.")
        self.get_logger().info(f"ChArUco config: {self.charuco_config_path}")
        self.get_logger().info(
            f"Board: {self.squares_x}x{self.squares_y}, square_length={self.square_length}, "
            f"marker_length={self.marker_length}, dictionary={self.dictionary_name}"
        )
        self.get_logger().info(f"camera_1 image topic: {self.camera_1_image_topic}")
        self.get_logger().info(f"camera_2 image topic: {self.camera_2_image_topic}")
        self.get_logger().info(f"show_gui: {self.show_gui}")
        self.get_logger().info(f"save_debug: {self.save_debug}")
        self.get_logger().info(f"debug_dir: {self.debug_dir}")

    def load_yaml(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def image_callback(self, camera_name, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"{camera_name}: cv_bridge failed: {e}")
            return

        self.frame_counter[camera_name] += 1
        detection = detect_charuco(image, self.dictionary, self.detector, self.board, use_preprocessing=True)
        annotated = draw_charuco_detection(image, detection)

        marker_count = detection["marker_count"]
        charuco_count = detection["charuco_count"]
        marker_ids = [] if detection["ids"] is None else detection["ids"].flatten().astype(int).tolist()
        charuco_ids = [] if detection["charuco_ids"] is None else detection["charuco_ids"].flatten().astype(int).tolist()

        label = (
            f"{camera_name} | method={detection['method']} | markers={marker_count} | "
            f"charuco={charuco_count}"
        )
        color = (0, 255, 0) if charuco_count > 0 else (0, 0, 255)
        cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        now = time.time()
        if now - self.last_log_time[camera_name] > 1.0:
            self.last_log_time[camera_name] = now
            self.get_logger().info(
                f"{camera_name}: frame={self.frame_counter[camera_name]} | "
                f"method={detection['method']} | markers={marker_count} | "
                f"marker_ids={marker_ids[:12]} | charuco_corners={charuco_count} | "
                f"charuco_ids={charuco_ids[:12]}"
            )

        if self.save_debug and self.frame_counter[camera_name] % self.save_every_n_frames == 0:
            out_path = os.path.join(
                self.debug_dir,
                f"{camera_name}_frame_{self.frame_counter[camera_name]:06d}_markers_{marker_count}_charuco_{charuco_count}.png",
            )
            cv2.imwrite(out_path, annotated)

        if self.show_gui:
            cv2.imshow(f"charuco_live_detector_{camera_name}", annotated)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = CharucoLiveDetector()
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
