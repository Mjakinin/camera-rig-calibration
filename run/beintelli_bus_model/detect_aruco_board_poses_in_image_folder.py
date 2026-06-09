#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


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
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    return K, D, "manual_hfov_fallback"


def parse_intrinsics_file(path: Path, width: int, height: int):
    if not path.exists():
        return None

    text = path.read_text()

    vals = {}
    for key in ["fx", "fy", "cx", "cy"]:
        m = re.search(rf"^\s*{key}\s*:\s*([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
        if m:
            vals[key] = float(m.group(1))

    if all(k in vals for k in ["fx", "fy", "cx", "cy"]):
        K = np.array(
            [
                [vals["fx"], 0.0, vals["cx"]],
                [0.0, vals["fy"], vals["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        D = np.zeros((5, 1), dtype=np.float64)
        return K, D, str(path)

    return None


def build_board_object_points(args):
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


def camera_center_in_board_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    t = tvec.reshape(3)
    C_board = -R.T @ t
    return C_board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--horizontal_fov_deg", type=float, default=90.0)

    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--first_id", type=int, required=True)

    parser.add_argument("--board_width_m", type=float, default=0.8)
    parser.add_argument("--board_height_m", type=float, default=0.6)
    parser.add_argument("--texture_width_px", type=int, default=1440)
    parser.add_argument("--texture_height_px", type=int, default=1080)
    parser.add_argument("--marker_px", type=int, default=300)
    parser.add_argument("--gap_px", type=int, default=80)

    parser.add_argument("--min_markers_for_pose", type=int, default=2)
    parser.add_argument("--max_reprojection_rmse_px", type=float, default=2.0)
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--debug_every", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    images_dir = Path(args.images_dir) if args.images_dir else dataset_dir / "images"
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    dictionary = get_aruco_dictionary(args.dictionary)
    detector = create_detector(dictionary)
    board_points = build_board_object_points(args)

    print(f"[INFO] dataset_dir: {dataset_dir}")
    print(f"[INFO] images_dir:  {images_dir}")
    print(f"[INFO] output_csv:  {output_csv}")
    print(f"[INFO] board ids:   {args.first_id}..{args.first_id + args.cols * args.rows - 1}")
    print(f"[INFO] images:      {len(image_paths)}")

    rows = []

    K_cache = None
    D_cache = None
    intrinsics_source = None

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        h, w = image.shape[:2]

        if K_cache is None:
            parsed = parse_intrinsics_file(dataset_dir / "camera_intrinsics_used.txt", w, h)
            if parsed:
                K_cache, D_cache, intrinsics_source = parsed
            else:
                K_cache, D_cache, intrinsics_source = camera_matrix_from_hfov(w, h, args.horizontal_fov_deg)

        K, D = K_cache, D_cache

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ids, corners, rejected = detect_markers(detector, gray)

        object_points = []
        image_points = []
        used_ids = []

        for marker_id, marker_corners in zip(ids, corners):
            if marker_id not in board_points:
                continue
            object_points.append(board_points[marker_id])
            image_points.append(np.asarray(marker_corners, dtype=np.float64).reshape(4, 2))
            used_ids.append(marker_id)

        status = "failed_not_enough_markers"
        rvec = None
        tvec = None
        rmse = None
        C_board = None

        if len(used_ids) >= args.min_markers_for_pose:
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
                rvec = rvec_tmp.reshape(3)
                tvec = tvec_tmp.reshape(3)
                rmse = reprojection_rmse(obj_all, img_all, rvec.reshape(3, 1), tvec.reshape(3, 1), K, D)
                C_board = camera_center_in_board_from_rvec_tvec(rvec, tvec)

                if rmse <= args.max_reprojection_rmse_px:
                    status = "pose_valid"
                else:
                    status = "failed_high_rmse"

        if debug_dir and args.debug_every > 0 and idx % args.debug_every == 0:
            debug = image.copy()
            if ids:
                ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
                cv2.aruco.drawDetectedMarkers(debug, corners, ids_np)
            if status == "pose_valid":
                try:
                    cv2.drawFrameAxes(debug, K, D, rvec.reshape(3, 1), tvec.reshape(3, 1), 0.25)
                except Exception:
                    pass
            cv2.imwrite(str(debug_dir / image_path.name), debug)

        rows.append({
            "image_name": image_path.name,
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
            "camera_center_board_x_m": "" if C_board is None else f"{C_board[0]:.8f}",
            "camera_center_board_y_m": "" if C_board is None else f"{C_board[1]:.8f}",
            "camera_center_board_z_m": "" if C_board is None else f"{C_board[2]:.8f}",
        })

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["status"] == "pose_valid"]
    print("")
    print("[OK] detection complete")
    print(f"[OK] valid poses: {len(valid)} / {len(rows)}")
    print(f"[OK] output:      {output_csv}")

    if valid:
        best = sorted(
            valid,
            key=lambda r: (-int(r["num_used_markers"]), float(r["reprojection_rmse_px"]))
        )[0]
        print(f"[OK] best image:  {best['image_name']}")
        print(f"[OK] best used:   {best['used_ids']}")
        print(f"[OK] best rmse:   {best['reprojection_rmse_px']} px")


if __name__ == "__main__":
    main()
