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

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not available. Install OpenCV contrib / ROS OpenCV with aruco support.")

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_detector(dictionary):
    # OpenCV newer API
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)

    # OpenCV older API
    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


def detect_markers(detector, gray):
    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, params = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None:
        return [], corners, rejected

    return [int(x) for x in ids.flatten().tolist()], corners, rejected


def remove_green_overlay_bgr(image):
    """Remove bright Gazebo/debug green overlay lines before ArUco detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green overlay is usually high-saturation green.
    lower = np.array([40, 80, 80], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Slightly expand mask to cover anti-aliased green line edges.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpaint masked line pixels from surrounding image.
    cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return cleaned, mask



def parse_expected_marker_ids(spec: str):
    """Parse marker id specification like '0-9' or '0,1,2,5'."""
    spec = spec.strip()
    if not spec:
        return list(range(10))

    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))

    return sorted(set(ids))


class BusArucoVisibilityDetector(Node):
    def __init__(self, args):
        super().__init__("bus_aruco_visibility_detector")

        self.args = args
        self.bridge = CvBridge()

        dictionary = get_aruco_dictionary(args.dictionary)
        self.detector = create_detector(dictionary)

        self.images = {
            "front_static_camera": None,
            "rear_static_camera": None,
        }

        self.sub_front = self.create_subscription(
            Image,
            args.front_topic,
            lambda msg: self.image_callback("front_static_camera", msg),
            10,
        )

        self.sub_rear = self.create_subscription(
            Image,
            args.rear_topic,
            lambda msg: self.image_callback("rear_static_camera", msg),
            10,
        )

        self.output_dir = Path(args.output_dir)
        self.debug_dir = self.output_dir / "debug_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info("Bus ArUco Visibility Detector started.")
        self.get_logger().info(f"Dictionary: {args.dictionary}")
        self.get_logger().info(f"Front topic: {args.front_topic}")
        self.get_logger().info(f"Rear topic:  {args.rear_topic}")
        self.get_logger().info(f"Output dir:  {self.output_dir}")

    def image_callback(self, camera_name, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image for {camera_name}: {exc}")
            return

        self.images[camera_name] = image

    def evaluate_once(self):
        rows = []
        detected_by_camera = {}

        for camera_name, image in self.images.items():
            if image is None:
                rows.append({
                    "camera": camera_name,
                    "status": "no_image",
                    "detected_ids": "",
                    "num_detected": 0,
                    "debug_image": "",
                })
                detected_by_camera[camera_name] = set()
                continue

            raw_path = self.debug_dir / f"{camera_name}_raw.png"
            cv2.imwrite(str(raw_path), image)

            if self.args.remove_green_overlay:
                detect_image, green_mask = remove_green_overlay_bgr(image)
                cleaned_path = self.debug_dir / f"{camera_name}_cleaned.png"
                mask_path = self.debug_dir / f"{camera_name}_green_mask.png"
                cv2.imwrite(str(cleaned_path), detect_image)
                cv2.imwrite(str(mask_path), green_mask)
            else:
                detect_image = image

            gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
            ids, corners, rejected = detect_markers(self.detector, gray)

            debug = detect_image.copy()

            if ids:
                ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
                cv2.aruco.drawDetectedMarkers(debug, corners, ids_np)
                status = "detected"
            else:
                status = "no_markers"

            cv2.putText(
                debug,
                f"{camera_name} | ids={ids}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if ids else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            debug_path = self.debug_dir / f"{camera_name}_aruco_visibility.png"
            cv2.imwrite(str(debug_path), debug)

            rows.append({
                "camera": camera_name,
                "status": status,
                "detected_ids": " ".join(str(i) for i in ids),
                "num_detected": len(ids),
                "debug_image": str(debug_path),
            })

            detected_by_camera[camera_name] = set(ids)

        front_ids = detected_by_camera.get("front_static_camera", set())
        rear_ids = detected_by_camera.get("rear_static_camera", set())
        overlap = sorted(front_ids.intersection(rear_ids))

        csv_path = self.output_dir / "bus_aruco_visibility.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["camera", "status", "detected_ids", "num_detected", "debug_image"],
            )
            writer.writeheader()
            writer.writerows(rows)

        expected_marker_ids = parse_expected_marker_ids(self.args.expected_marker_ids)
        matrix_rows = []
        for marker_id in expected_marker_ids:
            front_detected = marker_id in front_ids
            rear_detected = marker_id in rear_ids
            matrix_rows.append({
                "marker_id": marker_id,
                "front_detected": str(front_detected).lower(),
                "rear_detected": str(rear_detected).lower(),
                "overlap": str(front_detected and rear_detected).lower(),
            })

        matrix_path = self.output_dir / "marker_visibility_matrix.csv"
        with matrix_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["marker_id", "front_detected", "rear_detected", "overlap"],
            )
            writer.writeheader()
            writer.writerows(matrix_rows)

        summary_path = self.output_dir / "bus_aruco_visibility_summary.txt"
        summary_path.write_text(
            "BUS ARUCO VISIBILITY SUMMARY\n"
            "============================\n"
            f"front_static_camera detected: {sorted(front_ids)}\n"
            f"rear_static_camera detected:  {sorted(rear_ids)}\n"
            f"overlap IDs:                  {overlap}\n"
            f"front count:                  {len(front_ids)}\n"
            f"rear count:                   {len(rear_ids)}\n"
            f"overlap count:                {len(overlap)}\n"
            f"matrix CSV:                   {matrix_path}\n"
        )

        self.get_logger().info("")
        self.get_logger().info("================ BUS ARUCO VISIBILITY ================")
        self.get_logger().info(f"front_static_camera detected: {sorted(front_ids)}")
        self.get_logger().info(f"rear_static_camera detected:  {sorted(rear_ids)}")
        self.get_logger().info(f"overlap IDs:                  {overlap}")
        self.get_logger().info(f"CSV:                          {csv_path}")
        self.get_logger().info(f"Summary:                      {summary_path}")
        self.get_logger().info(f"Matrix CSV:                   {matrix_path}")
        self.get_logger().info(f"Debug dir:                    {self.debug_dir}")
        self.get_logger().info("======================================================")

        return rows, overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--front_topic", default="/front_static_camera/image")
    parser.add_argument("--rear_topic", default="/rear_static_camera/image")
    parser.add_argument("--output_dir", default="results/beintelli_bus_model/aruco_visibility")
    parser.add_argument("--wait_sec", type=float, default=5.0)
    parser.add_argument("--remove_green_overlay", action="store_true")
    parser.add_argument("--expected_marker_ids", default="0-9", help="Expected marker IDs, e.g. 0-9 or 0,1,2,5")
    args = parser.parse_args()

    rclpy.init()
    node = BusArucoVisibilityDetector(args)

    start = time.time()

    while rclpy.ok() and time.time() - start < args.wait_sec:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.evaluate_once()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
