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
        raise RuntimeError("cv2.aruco not available. Install OpenCV contrib / ROS OpenCV with aruco support.")

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


def marker_object_points(marker_size_m: float):
    s = marker_size_m / 2.0

    # OpenCV ArUco corner order:
    # top-left, top-right, bottom-right, bottom-left
    return np.array(
        [
            [-s, s, 0.0],
            [s, s, 0.0],
            [s, -s, 0.0],
            [-s, -s, 0.0],
        ],
        dtype=np.float64,
    )


def solve_marker_poses(ids, corners, camera_matrix, dist_coeffs, marker_size_m):
    obj_pts = marker_object_points(marker_size_m)
    poses = {}

    for marker_id, corner in zip(ids, corners):
        img_pts = np.asarray(corner, dtype=np.float64).reshape(4, 2)

        success = False
        rvec = None
        tvec = None

        # IPPE is usually good for square planar markers.
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
            ok, rvec_tmp, tvec_tmp = cv2.solvePnP(
                obj_pts,
                img_pts,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                success = True
                rvec = rvec_tmp
                tvec = tvec_tmp

        # Fallback.
        if not success:
            ok, rvec_tmp, tvec_tmp = cv2.solvePnP(
                obj_pts,
                img_pts,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                success = True
                rvec = rvec_tmp
                tvec = tvec_tmp

        if not success:
            continue

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        poses[marker_id] = {
            "R": R,
            "t": t,
            "rvec": rvec.reshape(3),
            "tvec": tvec.reshape(3),
            "corners": img_pts,
        }

    return poses


def make_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def rotation_angle_deg(R):
    value = (np.trace(R) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def mean_rotation(rotations):
    A = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        A += R

    U, _, Vt = np.linalg.svd(A)
    R_mean = U @ Vt

    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt

    return R_mean


class BusArucoDirectRigEstimator(Node):
    def __init__(self, args):
        super().__init__("bus_aruco_direct_rig_estimator")

        self.args = args
        self.bridge = CvBridge()

        dictionary = get_aruco_dictionary(args.dictionary)
        self.detector = create_detector(dictionary)

        self.images = {
            "front_static_camera": None,
            "rear_static_camera": None,
        }

        self.camera_infos = {
            "front_static_camera": None,
            "rear_static_camera": None,
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

        self.output_dir = Path(args.output_dir)
        self.debug_dir = self.output_dir / "debug_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info("Bus ArUco Direct Rig Estimator started.")
        self.get_logger().info(f"Dictionary:       {args.dictionary}")
        self.get_logger().info(f"Marker size:      {args.marker_size_m} m")
        self.get_logger().info(f"Horizontal FOV:   {args.horizontal_fov_deg} deg")
        self.get_logger().info(f"Front topic:      {args.front_topic}")
        self.get_logger().info(f"Rear topic:       {args.rear_topic}")
        self.get_logger().info(f"Front info topic: {args.front_info_topic}")
        self.get_logger().info(f"Rear info topic:  {args.rear_info_topic}")
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

    def evaluate_once(self):
        missing = [name for name, img in self.images.items() if img is None]
        if missing:
            raise RuntimeError(f"Missing images from cameras: {missing}")

        detections = {}
        poses_by_camera = {}

        for camera_name, image in self.images.items():
            h, w = image.shape[:2]
            K, dist, intrinsics_source = self.camera_matrix_for_camera(camera_name, w, h)

            self.get_logger().info(
                f"{camera_name} intrinsics source: {intrinsics_source}, "
                f"fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, cx={K[0,2]:.3f}, cy={K[1,2]:.3f}"
            )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ids, corners, rejected = detect_markers(self.detector, gray)

            poses = solve_marker_poses(
                ids,
                corners,
                K,
                dist,
                self.args.marker_size_m,
            )

            debug = image.copy()
            if ids:
                ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
                cv2.aruco.drawDetectedMarkers(debug, corners, ids_np)

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

            debug_path = self.debug_dir / f"{camera_name}_direct_aruco.png"
            cv2.imwrite(str(debug_path), debug)

            detections[camera_name] = set(ids)
            poses_by_camera[camera_name] = poses

        front_ids = detections["front_static_camera"]
        rear_ids = detections["rear_static_camera"]
        common_ids = sorted(front_ids.intersection(rear_ids))

        per_marker_rows = []
        transforms = []

        for marker_id in common_ids:
            front_pose = poses_by_camera["front_static_camera"].get(marker_id)
            rear_pose = poses_by_camera["rear_static_camera"].get(marker_id)

            if front_pose is None or rear_pose is None:
                continue

            # T_camera_marker maps marker coordinates into camera coordinates.
            T_front_marker = make_transform(front_pose["R"], front_pose["t"])
            T_rear_marker = make_transform(rear_pose["R"], rear_pose["t"])

            # Transform from rear camera frame into front camera frame.
            # X_front = T_front_rear * X_rear
            T_front_rear = T_front_marker @ np.linalg.inv(T_rear_marker)

            R = T_front_rear[:3, :3]
            t = T_front_rear[:3, 3]

            baseline = float(np.linalg.norm(t))
            rot_deg = rotation_angle_deg(R)

            transforms.append(T_front_rear)

            per_marker_rows.append({
                "marker_id": marker_id,
                "tx_m": t[0],
                "ty_m": t[1],
                "tz_m": t[2],
                "baseline_m": baseline,
                "relative_rotation_deg": rot_deg,
            })

        per_marker_csv = self.output_dir / "per_marker_transforms.csv"
        with per_marker_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "marker_id",
                    "tx_m",
                    "ty_m",
                    "tz_m",
                    "baseline_m",
                    "relative_rotation_deg",
                ],
            )
            writer.writeheader()
            writer.writerows(per_marker_rows)

        summary_path = self.output_dir / "direct_aruco_rig_summary.txt"

        if not transforms:
            summary_path.write_text(
                "BUS DIRECT ARUCO RIG SUMMARY\n"
                "============================\n"
                f"front detected IDs: {sorted(front_ids)}\n"
                f"rear detected IDs:  {sorted(rear_ids)}\n"
                f"common IDs:         {common_ids}\n"
                "status:             failed_no_common_pose_markers\n"
            )
            raise RuntimeError("No common markers with valid poses.")

        rotations = [T[:3, :3] for T in transforms]
        translations = np.array([T[:3, 3] for T in transforms], dtype=np.float64)

        R_mean = mean_rotation(rotations)
        t_mean = translations.mean(axis=0)
        baseline_mean = float(np.linalg.norm(t_mean))

        translation_spread = translations.std(axis=0)
        baseline_values = np.array([np.linalg.norm(t) for t in translations], dtype=np.float64)
        baseline_std = float(baseline_values.std())

        rot_spread_values = []
        for R in rotations:
            dR = R_mean.T @ R
            rot_spread_values.append(rotation_angle_deg(dR))

        rot_spread_mean = float(np.mean(rot_spread_values))
        rot_spread_max = float(np.max(rot_spread_values))

        mean_transform_csv = self.output_dir / "mean_transform_front_T_rear.csv"
        with mean_transform_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            for row in make_transform(R_mean, t_mean.reshape(3, 1)):
                writer.writerow([f"{x:.10f}" for x in row])

        summary_text = (
            "BUS DIRECT ARUCO RIG SUMMARY\n"
            "============================\n"
            f"front detected IDs:          {sorted(front_ids)}\n"
            f"rear detected IDs:           {sorted(rear_ids)}\n"
            f"common IDs:                  {common_ids}\n"
            f"valid common pose markers:   {len(transforms)}\n"
            "\n"
            "Estimated transform convention:\n"
            "  T_front_rear maps points from rear camera frame into front camera frame.\n"
            "\n"
            f"mean translation x/y/z [m]:  {t_mean[0]:.6f}, {t_mean[1]:.6f}, {t_mean[2]:.6f}\n"
            f"mean baseline norm [m]:      {baseline_mean:.6f}\n"
            f"baseline std [m]:            {baseline_std:.6f}\n"
            f"translation std x/y/z [m]:   {translation_spread[0]:.6f}, {translation_spread[1]:.6f}, {translation_spread[2]:.6f}\n"
            f"rotation spread mean [deg]:  {rot_spread_mean:.6f}\n"
            f"rotation spread max [deg]:   {rot_spread_max:.6f}\n"
            "\n"
            f"per-marker CSV:              {per_marker_csv}\n"
            f"mean transform CSV:          {mean_transform_csv}\n"
            f"debug dir:                   {self.debug_dir}\n"
        )

        summary_path.write_text(summary_text)

        self.get_logger().info("")
        self.get_logger().info(summary_text)

        return transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--front_topic", default="/front_static_camera/image")
    parser.add_argument("--rear_topic", default="/rear_static_camera/image")
    parser.add_argument("--front_info_topic", default="/front_static_camera/camera_info")
    parser.add_argument("--rear_info_topic", default="/rear_static_camera/camera_info")
    parser.add_argument("--output_dir", default="results/beintelli_bus_model/aruco_direct_rig/current")
    parser.add_argument("--wait_sec", type=float, default=5.0)
    parser.add_argument("--marker_size_m", type=float, default=0.60)
    parser.add_argument("--horizontal_fov_deg", type=float, default=90.0)
    args = parser.parse_args()

    rclpy.init()
    node = BusArucoDirectRigEstimator(args)

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
