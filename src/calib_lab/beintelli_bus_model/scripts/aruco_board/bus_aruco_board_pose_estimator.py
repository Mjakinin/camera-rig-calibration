#!/usr/bin/env python3

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


def get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not available. Install OpenCV contrib.")

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_detector(dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)

    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


def detect_markers(detector, gray):
    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, params = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None:
        return [], [], rejected

    return [int(x) for x in ids.flatten().tolist()], corners, rejected


def camera_matrix_from_hfov(width: int, height: int, horizontal_fov_deg: float):
    hfov_rad = math.radians(horizontal_fov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_board_object_points(args):
    """
    Board model:
      - physical board width is along local Y
      - physical board height is along local Z
      - board plane is X=0
      - texture origin is top-left

    Marker corner order follows OpenCV ArUco:
      top-left, top-right, bottom-right, bottom-left
    """

    marker_points = {}

    grid_w_px = args.cols * args.marker_px + (args.cols - 1) * args.gap_px
    grid_h_px = args.rows * args.marker_px + (args.rows - 1) * args.gap_px

    x0_px = (args.texture_width_px - grid_w_px) / 2.0
    y0_px = (args.texture_height_px - grid_h_px) / 2.0

    def px_to_board(u_px, v_px):
        y_m = (u_px / args.texture_width_px - 0.5) * args.board_width_m
        z_m = (0.5 - v_px / args.texture_height_px) * args.board_height_m
        return y_m, z_m

    for r in range(args.rows):
        for c in range(args.cols):
            marker_id = args.first_id + r * args.cols + c

            u_left = x0_px + c * (args.marker_px + args.gap_px)
            u_right = u_left + args.marker_px
            v_top = y0_px + r * (args.marker_px + args.gap_px)
            v_bottom = v_top + args.marker_px

            y_tl, z_tl = px_to_board(u_left, v_top)
            y_tr, z_tr = px_to_board(u_right, v_top)
            y_br, z_br = px_to_board(u_right, v_bottom)
            y_bl, z_bl = px_to_board(u_left, v_bottom)

            # X=0 board plane.
            marker_points[marker_id] = np.array(
                [
                    [0.0, y_tl, z_tl],
                    [0.0, y_tr, z_tr],
                    [0.0, y_br, z_br],
                    [0.0, y_bl, z_bl],
                ],
                dtype=np.float64,
            )

    return marker_points


def reprojection_rmse(obj_points, img_points, rvec, tvec, K, D):
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    projected = projected.reshape(-1, 2)
    img_points = img_points.reshape(-1, 2)
    err = projected - img_points
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


class BusArucoBoardPoseEstimator(Node):
    def __init__(self, args):
        super().__init__("bus_aruco_board_pose_estimator")

        self.args = args
        self.bridge = CvBridge()

        dictionary = get_aruco_dictionary(args.dictionary)
        self.detector = create_detector(dictionary)
        self.board_object_points = build_board_object_points(args)

        self.images = {
            "front_static_camera": None,
            "rear_static_camera": None,
            "moving_calib_camera": None,
        }

        self.camera_infos = {
            "front_static_camera": None,
            "rear_static_camera": None,
            "moving_calib_camera": None,
        }

        self.create_subscription(
            Image,
            args.front_topic,
            lambda msg: self.image_callback("front_static_camera", msg),
            10,
        )

        self.create_subscription(
            Image,
            args.rear_topic,
            lambda msg: self.image_callback("rear_static_camera", msg),
            10,
        )

        self.create_subscription(
            Image,
            args.moving_topic,
            lambda msg: self.image_callback("moving_calib_camera", msg),
            10,
        )

        self.create_subscription(
            CameraInfo,
            args.front_info_topic,
            lambda msg: self.camera_info_callback("front_static_camera", msg),
            10,
        )

        self.create_subscription(
            CameraInfo,
            args.rear_info_topic,
            lambda msg: self.camera_info_callback("rear_static_camera", msg),
            10,
        )

        self.create_subscription(
            CameraInfo,
            args.moving_info_topic,
            lambda msg: self.camera_info_callback("moving_calib_camera", msg),
            10,
        )

        self.output_dir = Path(args.output_dir)
        self.debug_dir = self.output_dir / "debug_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info("Bus ArUco Board Pose Estimator started.")
        self.get_logger().info(f"Dictionary:       {args.dictionary}")
        self.get_logger().info(f"Board layout:     {args.cols}x{args.rows}, ids {args.first_id}..{args.first_id + args.cols * args.rows - 1}")
        self.get_logger().info(f"Board size:       {args.board_width_m} x {args.board_height_m} m")
        self.get_logger().info(f"Output dir:       {self.output_dir}")

    def image_callback(self, camera_name, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image for {camera_name}: {exc}")
            return

        self.images[camera_name] = image

    def camera_info_callback(self, camera_name, msg):
        self.camera_infos[camera_name] = msg

    def camera_matrix_for_camera(self, camera_name, width, height):
        info = self.camera_infos.get(camera_name)

        if info is not None:
            K = np.array(info.k, dtype=np.float64).reshape(3, 3)
            D = np.array(info.d, dtype=np.float64).reshape(-1, 1) if info.d else np.zeros((5, 1), dtype=np.float64)
            return K, D, "camera_info"

        K = camera_matrix_from_hfov(width, height, self.args.horizontal_fov_deg)
        D = np.zeros((5, 1), dtype=np.float64)
        return K, D, "manual_hfov_fallback"

    def estimate_camera_pose(self, camera_name, image):
        h, w = image.shape[:2]
        K, D, intrinsics_source = self.camera_matrix_for_camera(camera_name, w, h)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ids, corners, rejected = detect_markers(self.detector, gray)

        object_points = []
        image_points = []
        used_ids = []

        for marker_id, marker_corners in zip(ids, corners):
            if marker_id not in self.board_object_points:
                continue

            obj = self.board_object_points[marker_id]
            img = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)

            object_points.append(obj)
            image_points.append(img)
            used_ids.append(marker_id)

        debug = image.copy()

        if ids:
            ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(debug, corners, ids_np)

        status = "failed_not_enough_markers"
        rvec = None
        tvec = None
        rmse = None

        if len(used_ids) >= self.args.min_markers_for_pose:
            obj_all = np.concatenate(object_points, axis=0).astype(np.float64)
            img_all = np.concatenate(image_points, axis=0).astype(np.float64)

            ok, rvec_tmp, tvec_tmp = cv2.solvePnP(
                obj_all,
                img_all,
                K,
                D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if ok:
                status = "pose_valid"
                rvec = rvec_tmp.reshape(3)
                tvec = tvec_tmp.reshape(3)
                rmse = reprojection_rmse(
                    obj_all,
                    img_all,
                    rvec.reshape(3, 1),
                    tvec.reshape(3, 1),
                    K,
                    D,
                )

                try:
                    cv2.drawFrameAxes(
                        debug,
                        K,
                        D,
                        rvec.reshape(3, 1),
                        tvec.reshape(3, 1),
                        self.args.axis_length_m,
                    )
                except Exception:
                    pass

        label = f"{camera_name} | {status} | ids={ids} | used={used_ids}"
        cv2.putText(
            debug,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if status == "pose_valid" else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        debug_path = self.debug_dir / f"{camera_name}_aruco_board_pose.png"
        cv2.imwrite(str(debug_path), debug)

        row = {
            "camera": camera_name,
            "status": status,
            "intrinsics_source": intrinsics_source,
            "detected_ids": str(sorted(ids)),
            "used_ids": str(sorted(used_ids)),
            "num_detected": len(ids),
            "num_used_markers": len(used_ids),
            "num_points": len(used_ids) * 4,
            "reprojection_rmse_px": "" if rmse is None else f"{rmse:.6f}",
            "tvec_x_m": "" if tvec is None else f"{tvec[0]:.8f}",
            "tvec_y_m": "" if tvec is None else f"{tvec[1]:.8f}",
            "tvec_z_m": "" if tvec is None else f"{tvec[2]:.8f}",
            "rvec_x": "" if rvec is None else f"{rvec[0]:.8f}",
            "rvec_y": "" if rvec is None else f"{rvec[1]:.8f}",
            "rvec_z": "" if rvec is None else f"{rvec[2]:.8f}",
            "debug_image": str(debug_path),
        }

        return row

    def evaluate_once(self):
        missing = [name for name, img in self.images.items() if img is None]
        if missing:
            raise RuntimeError(f"Missing images from cameras: {missing}")

        rows = []
        for camera_name, image in self.images.items():
            rows.append(self.estimate_camera_pose(camera_name, image))

        csv_path = self.output_dir / "aruco_board_pose_observations.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "camera",
                    "status",
                    "intrinsics_source",
                    "detected_ids",
                    "used_ids",
                    "num_detected",
                    "num_used_markers",
                    "num_points",
                    "reprojection_rmse_px",
                    "tvec_x_m",
                    "tvec_y_m",
                    "tvec_z_m",
                    "rvec_x",
                    "rvec_y",
                    "rvec_z",
                    "debug_image",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        summary_path = self.output_dir / "aruco_board_pose_summary.txt"

        summary_lines = [
            "BUS ARUCO BOARD POSE SUMMARY",
            "============================",
        ]

        for row in rows:
            summary_lines.append("")
            summary_lines.append(f"camera:                 {row['camera']}")
            summary_lines.append(f"status:                 {row['status']}")
            summary_lines.append(f"detected_ids:           {row['detected_ids']}")
            summary_lines.append(f"used_ids:               {row['used_ids']}")
            summary_lines.append(f"num_used_markers:       {row['num_used_markers']}")
            summary_lines.append(f"num_points:             {row['num_points']}")
            summary_lines.append(f"reprojection_rmse_px:   {row['reprojection_rmse_px']}")
            summary_lines.append(f"tvec_xyz_m:             {row['tvec_x_m']}, {row['tvec_y_m']}, {row['tvec_z_m']}")
            summary_lines.append(f"debug_image:            {row['debug_image']}")

        summary_lines.append("")
        summary_lines.append(f"observations_csv:       {csv_path}")
        summary_lines.append(f"debug_dir:              {self.debug_dir}")

        summary_text = "\n".join(summary_lines) + "\n"
        summary_path.write_text(summary_text)

        self.get_logger().info("")
        self.get_logger().info(summary_text)

        return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dictionary", default="DICT_4X4_50")

    parser.add_argument("--front_topic", default="/front_static_camera/image")
    parser.add_argument("--rear_topic", default="/rear_static_camera/image")
    parser.add_argument("--moving_topic", default="/moving_calib_camera/image")
    parser.add_argument("--front_info_topic", default="/front_static_camera/camera_info")
    parser.add_argument("--rear_info_topic", default="/rear_static_camera/camera_info")
    parser.add_argument("--moving_info_topic", default="/moving_calib_camera/camera_info")

    parser.add_argument("--output_dir", default="results/beintelli_bus_model/aruco_board_pose/current")
    parser.add_argument("--wait_sec", type=float, default=5.0)

    parser.add_argument("--horizontal_fov_deg", type=float, default=90.0)

    # Must match aruco_gridboard_target/board_layout.txt
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--first_id", type=int, default=0)
    parser.add_argument("--board_width_m", type=float, default=0.8)
    parser.add_argument("--board_height_m", type=float, default=0.6)
    parser.add_argument("--texture_width_px", type=int, default=1440)
    parser.add_argument("--texture_height_px", type=int, default=1080)
    parser.add_argument("--marker_px", type=int, default=300)
    parser.add_argument("--gap_px", type=int, default=80)

    parser.add_argument("--min_markers_for_pose", type=int, default=2)
    parser.add_argument("--axis_length_m", type=float, default=0.25)

    args = parser.parse_args()

    rclpy.init()
    node = BusArucoBoardPoseEstimator(args)

    start = time.time()
    while rclpy.ok() and time.time() - start < args.wait_sec:
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        node.evaluate_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
