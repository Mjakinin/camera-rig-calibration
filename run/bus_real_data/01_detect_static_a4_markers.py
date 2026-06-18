#!/usr/bin/env python3

import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


CAMERA_TOPICS = {
    "cam_edge_0": "/bus_real_data/cam_edge_0/image",
    "cam_edge_1": "/bus_real_data/cam_edge_1/image",
    "cam_edge_3": "/bus_real_data/cam_edge_3/image",
    "cam_edge_5": "/bus_real_data/cam_edge_5/image",
}

MARKER_LENGTH_M = 0.170
ARUCO_DICT_NAME = "DICT_4X4_50"


def image_msg_to_bgr(msg: Image):
    h = msg.height
    w = msg.width
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ["rgb8", "bgr8"]:
        arr = data.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3).copy()
        if enc == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    if enc in ["rgba8", "bgra8"]:
        arr = data.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4).copy()
        if enc == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if enc in ["mono8", "8uc1"]:
        arr = data.reshape(h, msg.step)[:, :w].reshape(h, w).copy()
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


def get_aruco_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT_NAME))

    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        return ("new", aruco.ArucoDetector(dictionary, params), dictionary, params)

    return ("old", None, dictionary, params)


def detect_markers(gray, detector_pack):
    mode, detector, dictionary, params = detector_pack

    if mode == "new":
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None:
        ids = np.empty((0, 1), dtype=np.int32)
        corners = []

    return corners, ids


def marker_object_points(marker_length):
    s = marker_length / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float32)


class OneShotImageCollector(Node):
    def __init__(self):
        super().__init__("a4_aruco_static_camera_detector")
        self.images = {}
        self.subs = []

        for cam, topic in CAMERA_TOPICS.items():
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, cam=cam: self.cb(msg, cam),
                10,
            )
            self.subs.append(sub)
            self.get_logger().info(f"subscribed {cam}: {topic}")

    def cb(self, msg, cam):
        if cam not in self.images:
            self.images[cam] = msg
            self.get_logger().info(f"received image from {cam}")


def load_intrinsics(path):
    data = yaml.safe_load(Path(path).read_text())
    out = {}

    for cam, intr in data.items():
        K = np.array([
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        D = np.array(intr.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)

        out[cam] = {
            **intr,
            "K": K,
            "D": D,
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", default="src/calib_lab/bus_real_data/config/camera_intrinsics_by_camera.yaml")
    ap.add_argument("--out", default="results/bus_real_data/01_marker_direct_relay_multimarker_multichain/01_static_a4_marker_detection")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw_images"
    dbg_dir = out_dir / "debug_images"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = load_intrinsics(args.intrinsics)
    detector_pack = get_aruco_detector()
    obj_pts = marker_object_points(MARKER_LENGTH_M)

    rclpy.init()
    node = OneShotImageCollector()

    start = time.time()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if set(node.images.keys()) == set(CAMERA_TOPICS.keys()):
            break
        if time.time() - start > args.timeout:
            break

    images = dict(node.images)
    node.destroy_node()
    rclpy.shutdown()

    missing = sorted(set(CAMERA_TOPICS.keys()) - set(images.keys()))
    if missing:
        print("[WARN] missing images:", ", ".join(missing))

    detection_rows = []
    summary_rows = []

    for cam in sorted(CAMERA_TOPICS.keys()):
        if cam not in images:
            summary_rows.append({
                "camera": cam,
                "source_camera_info": intrinsics.get(cam, {}).get("source_camera_info", ""),
                "num_detected": 0,
                "detected_ids": "",
                "raw_image": "",
                "debug_image": "",
                "status": "missing_image",
            })
            continue

        msg = images[cam]
        bgr = image_msg_to_bgr(msg)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        raw_path = raw_dir / f"{cam}.png"
        dbg_path = dbg_dir / f"{cam}_debug.png"

        cv2.imwrite(str(raw_path), bgr)

        corners, ids = detect_markers(gray, detector_pack)
        debug = bgr.copy()

        detected_ids = []

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)

        K = intrinsics[cam]["K"]
        D = intrinsics[cam]["D"]

        for idx, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])
            c = corners[idx].reshape(4, 2).astype(np.float32)
            center = c.mean(axis=0)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                c,
                K,
                D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if ok:
                try:
                    cv2.drawFrameAxes(debug, K, D, rvec, tvec, MARKER_LENGTH_M * 0.5)
                except Exception:
                    pass

                tx, ty, tz = [float(v) for v in tvec.reshape(3)]
                rx, ry, rz = [float(v) for v in rvec.reshape(3)]
                distance = math.sqrt(tx * tx + ty * ty + tz * tz)
            else:
                tx = ty = tz = rx = ry = rz = distance = float("nan")

            detected_ids.append(marker_id)

            detection_rows.append({
                "camera": cam,
                "source_camera_info": intrinsics[cam].get("source_camera_info", ""),
                "image_width": int(intrinsics[cam]["width"]),
                "image_height": int(intrinsics[cam]["height"]),
                "fx": float(intrinsics[cam]["fx"]),
                "fy": float(intrinsics[cam]["fy"]),
                "cx": float(intrinsics[cam]["cx"]),
                "cy": float(intrinsics[cam]["cy"]),
                "hfov_deg": float(intrinsics[cam]["horizontal_fov_deg"]),
                "vfov_deg": float(intrinsics[cam]["vertical_fov_deg"]),
                "marker_length_m": MARKER_LENGTH_M,
                "marker_id": marker_id,
                "center_u": float(center[0]),
                "center_v": float(center[1]),
                "corner0_u": float(c[0, 0]),
                "corner0_v": float(c[0, 1]),
                "corner1_u": float(c[1, 0]),
                "corner1_v": float(c[1, 1]),
                "corner2_u": float(c[2, 0]),
                "corner2_v": float(c[2, 1]),
                "corner3_u": float(c[3, 0]),
                "corner3_v": float(c[3, 1]),
                "pnp_success": bool(ok),
                "tvec_x_m": tx,
                "tvec_y_m": ty,
                "tvec_z_m": tz,
                "distance_m": distance,
                "rvec_x": rx,
                "rvec_y": ry,
                "rvec_z": rz,
            })

        cv2.imwrite(str(dbg_path), debug)

        detected_ids_sorted = sorted(set(detected_ids))
        summary_rows.append({
            "camera": cam,
            "source_camera_info": intrinsics[cam].get("source_camera_info", ""),
            "num_detected": len(detected_ids_sorted),
            "detected_ids": ";".join(str(v) for v in detected_ids_sorted),
            "raw_image": str(raw_path),
            "debug_image": str(dbg_path),
            "status": "ok",
        })

        print(f"[{cam}] detected IDs: {detected_ids_sorted}")

    detections_csv = out_dir / "detections.csv"
    summary_csv = out_dir / "summary_by_camera.csv"

    detection_fields = [
        "camera", "source_camera_info",
        "image_width", "image_height",
        "fx", "fy", "cx", "cy", "hfov_deg", "vfov_deg",
        "marker_length_m", "marker_id",
        "center_u", "center_v",
        "corner0_u", "corner0_v", "corner1_u", "corner1_v",
        "corner2_u", "corner2_v", "corner3_u", "corner3_v",
        "pnp_success", "tvec_x_m", "tvec_y_m", "tvec_z_m",
        "distance_m", "rvec_x", "rvec_y", "rvec_z",
    ]

    with detections_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detection_fields)
        writer.writeheader()
        writer.writerows(detection_rows)

    summary_fields = ["camera", "source_camera_info", "num_detected", "detected_ids", "raw_image", "debug_image", "status"]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    readme = out_dir / "README.txt"
    readme.write_text(
        "A4 ArUco static-camera detection results\n"
        "=========================================\n\n"
        f"Dictionary: {ARUCO_DICT_NAME}\n"
        f"Marker length: {MARKER_LENGTH_M} m\n"
        f"Intrinsics: {args.intrinsics}\n\n"
        "Files:\n"
        "- raw_images/: one captured image per static camera\n"
        "- debug_images/: detected markers drawn on image\n"
        "- detections.csv: per-marker detection and PnP rows\n"
        "- summary_by_camera.csv: detected marker IDs per camera\n"
    )

    print()
    print("[OK] wrote", out_dir)
    print("[OK] summary:", summary_csv)
    print("[OK] detections:", detections_csv)


if __name__ == "__main__":
    main()
