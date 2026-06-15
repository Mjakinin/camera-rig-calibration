#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
MARKER_LENGTH_M = 0.170
ARUCO_DICT_NAME = "DICT_4X4_50"


def degrade_image(bgr, mode, value):
    if mode == "gaussian_blur":
        k = int(value)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(bgr, (k, k), 0)

    if mode == "motion_blur":
        k = int(value)
        if k < 1:
            return bgr.copy()
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(bgr, -1, kernel)

    if mode == "brightness":
        alpha = float(value) / 100.0
        return cv2.convertScaleAbs(bgr, alpha=alpha, beta=0)

    if mode == "contrast":
        alpha = float(value)
        return cv2.convertScaleAbs(bgr, alpha=alpha, beta=0)

    if mode == "gamma":
        gamma = float(value)
        inv = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(bgr, table)

    raise ValueError(f"unknown degradation mode: {mode}")


def load_intrinsics(path):
    data = yaml.safe_load(Path(path).read_text())
    out = {}

    for cam, intr in data.items():
        K = np.array([
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        D = np.array(
            intr.get("distortion", [0, 0, 0, 0, 0]),
            dtype=np.float64,
        ).reshape(-1, 1)

        out[cam] = {
            **intr,
            "K": K,
            "D": D,
        }

    return out


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

    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    if mode == "new":
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=params,
        )

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


def load_baseline_detections(path):
    out = {}

    path = Path(path)
    if not path.exists():
        return out

    with path.open() as f:
        for row in csv.DictReader(f):
            cam = row["camera"]
            marker_id = int(row["marker_id"])
            out.setdefault(cam, {})[marker_id] = row

    return out


def row_corners(row):
    return np.array([
        [float(row["corner0_u"]), float(row["corner0_v"])],
        [float(row["corner1_u"]), float(row["corner1_v"])],
        [float(row["corner2_u"]), float(row["corner2_v"])],
        [float(row["corner3_u"]), float(row["corner3_v"])],
    ], dtype=np.float32)


def draw_label(img, text, x, y, color, scale=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = max(0, min(int(x), img.shape[1] - tw - 6))
    y = max(th + 6, min(int(y), img.shape[0] - 6))

    cv2.rectangle(
        img,
        (x, y - th - baseline - 5),
        (x + tw + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img,
        text,
        (x + 2, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_polygon_from_row(img, row, color, text):
    pts = row_corners(row).astype(np.int32)
    cv2.polylines(
        img,
        [pts],
        isClosed=True,
        color=color,
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    center = pts.mean(axis=0)
    draw_label(img, text, center[0], center[1], color)


def make_contact_sheet(image_paths, out_path, thumb_w=480, cols=2):
    thumbs = []

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_w / float(w)
        thumb_h = int(h * scale)
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)

    if not thumbs:
        return

    thumb_h = thumbs[0].shape[0]
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, img in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y0 = r * thumb_h
        x0 = c * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-static",
        default="results/bus_real_data/01_static_a4_marker_detection",
        help="Baseline static detection folder containing raw_images/ and detections.csv",
    )
    ap.add_argument(
        "--intrinsics",
        default="src/calib_lab/bus_real_data/config/camera_intrinsics_by_camera.yaml",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        required=True,
        choices=["gaussian_blur", "motion_blur", "brightness", "contrast", "gamma"],
    )
    ap.add_argument("--value", required=True)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--no-axes", action="store_true")
    args = ap.parse_args()

    input_static = Path(args.input_static)
    input_raw_dir = input_static / "raw_images"
    input_det_csv = input_static / "detections.csv"

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
    baseline = load_baseline_detections(input_det_csv)

    detection_rows = []
    summary_rows = []
    report_rows = []
    debug_paths = []

    for cam in CAMERAS:
        src_path = input_raw_dir / f"{cam}.png"
        bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)

        if bgr is None:
            summary_rows.append({
                "camera": cam,
                "source_camera_info": intrinsics.get(cam, {}).get("source_camera_info", ""),
                "num_detected": 0,
                "detected_ids": "",
                "raw_image": "",
                "debug_image": "",
                "status": "missing_input_image",
            })
            report_rows.append({
                "camera": cam,
                "baseline_ids": sorted(baseline.get(cam, {}).keys()),
                "degraded_ids": [],
                "missed_ids": sorted(baseline.get(cam, {}).keys()),
            })
            continue

        degraded = degrade_image(bgr, args.mode, args.value)
        gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)

        raw_path = raw_dir / f"{cam}.png"
        dbg_path = dbg_dir / f"{cam}_debug.png"

        cv2.imwrite(str(raw_path), degraded)

        corners, ids = detect_markers(gray, detector_pack)

        debug = degraded.copy()
        detected_ids = []

        K = intrinsics[cam]["K"]
        D = intrinsics[cam]["D"]

        # First draw missed baseline markers in red.
        degraded_id_set = set(int(v[0]) for v in ids)
        baseline_id_set = set(baseline.get(cam, {}).keys())
        missed_ids = sorted(baseline_id_set - degraded_id_set)

        for marker_id in missed_ids:
            row = baseline[cam][marker_id]
            draw_polygon_from_row(
                debug,
                row,
                color=(0, 0, 255),
                text=f"missed {marker_id}",
            )

        # Then draw detected degraded markers in green.
        for idx, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])
            c = corners[idx].reshape(4, 2).astype(np.float32)
            center = c.mean(axis=0)

            cv2.polylines(
                debug,
                [c.astype(np.int32)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            draw_label(
                debug,
                f"id {marker_id}",
                center[0],
                center[1],
                color=(0, 255, 0),
            )

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                c,
                K,
                D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if ok:
                if not args.no_axes:
                    try:
                        cv2.drawFrameAxes(
                            debug,
                            K,
                            D,
                            rvec,
                            tvec,
                            MARKER_LENGTH_M * 0.5,
                        )
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

        detected_ids_sorted = sorted(set(detected_ids))

        # Header overlay.
        cv2.rectangle(debug, (0, 0), (debug.shape[1], 95), (0, 0, 0), -1)
        draw_label(debug, cam, 12, 28, color=(255, 255, 255), scale=0.7)
        draw_label(
            debug,
            f"detected degraded: {detected_ids_sorted}",
            12,
            58,
            color=(0, 255, 0),
        )
        draw_label(
            debug,
            f"missed vs baseline: {missed_ids}",
            12,
            84,
            color=(0, 0, 255),
        )

        cv2.imwrite(str(dbg_path), debug)
        debug_paths.append(dbg_path)

        summary_rows.append({
            "camera": cam,
            "source_camera_info": intrinsics[cam].get("source_camera_info", ""),
            "num_detected": len(detected_ids_sorted),
            "detected_ids": ";".join(str(v) for v in detected_ids_sorted),
            "raw_image": str(raw_path),
            "debug_image": str(dbg_path),
            "status": "ok",
        })

        report_rows.append({
            "camera": cam,
            "baseline_ids": sorted(baseline_id_set),
            "degraded_ids": detected_ids_sorted,
            "missed_ids": missed_ids,
        })

        print(f"[{cam}] detected degraded IDs: {detected_ids_sorted}")
        print(f"[{cam}] missed vs baseline: {missed_ids}")

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

    detections_csv = out_dir / "detections.csv"
    with detections_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detection_fields)
        writer.writeheader()
        writer.writerows(detection_rows)

    summary_fields = [
        "camera",
        "source_camera_info",
        "num_detected",
        "detected_ids",
        "raw_image",
        "debug_image",
        "status",
    ]

    summary_csv = out_dir / "summary_by_camera.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path = out_dir / "static_detection_report.txt"
    with report_path.open("w") as f:
        f.write("Static-camera degradation detection report\n")
        f.write("=========================================\n\n")
        f.write(f"Input static folder: {input_static}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Value: {args.value}\n")
        f.write(f"Dictionary: {ARUCO_DICT_NAME}\n")
        f.write(f"Marker length: {MARKER_LENGTH_M} m\n\n")

        for row in report_rows:
            f.write(f"{row['camera']}\n")
            f.write(f"  baseline IDs: {row['baseline_ids']}\n")
            f.write(f"  degraded IDs: {row['degraded_ids']}\n")
            f.write(f"  missed IDs: {row['missed_ids']}\n\n")

        all_baseline = sorted(set().union(*[set(r["baseline_ids"]) for r in report_rows]) if report_rows else set())
        all_degraded = sorted(set().union(*[set(r["degraded_ids"]) for r in report_rows]) if report_rows else set())
        all_missed = sorted(set(all_baseline) - set(all_degraded))

        f.write("Global summary\n")
        f.write(f"  baseline IDs across static cameras: {all_baseline}\n")
        f.write(f"  degraded IDs across static cameras: {all_degraded}\n")
        f.write(f"  globally missed IDs: {all_missed}\n\n")
        f.write("Legend for debug images:\n")
        f.write("  green polygon: detected in degraded static image\n")
        f.write("  red polygon: detected in baseline static image but missed after degradation\n")

    metadata_path = out_dir / "degradation_metadata.json"
    metadata_path.write_text(json.dumps({
        "input_static": str(input_static),
        "mode": args.mode,
        "value": args.value,
        "raw_images": str(raw_dir),
        "debug_images": str(dbg_dir),
        "detections_csv": str(detections_csv),
        "summary_csv": str(summary_csv),
        "report": str(report_path),
    }, indent=2))

    make_contact_sheet(
        debug_paths,
        out_dir / "static_detection_contact_sheet.png",
    )

    readme = out_dir / "README.txt"
    readme.write_text(
        "Degraded static-camera A4 ArUco detection results\n"
        "================================================\n\n"
        f"Input static folder: {input_static}\n"
        f"Mode: {args.mode}\n"
        f"Value: {args.value}\n"
        f"Dictionary: {ARUCO_DICT_NAME}\n"
        f"Marker length: {MARKER_LENGTH_M} m\n\n"
        "Files:\n"
        "- raw_images/: degraded static camera images\n"
        "- debug_images/: green detected markers, red missed baseline markers\n"
        "- detections.csv: per-marker detection and PnP rows\n"
        "- summary_by_camera.csv: detected marker IDs per camera\n"
        "- static_detection_report.txt: baseline-vs-degraded ID comparison\n"
        "- static_detection_contact_sheet.png: overview of all static cameras\n"
    )

    print()
    print("[OK] wrote", out_dir)
    print("[OK] detections:", detections_csv)
    print("[OK] summary:", summary_csv)
    print("[OK] report:", report_path)
    print("[OK] contact sheet:", out_dir / "static_detection_contact_sheet.png")


if __name__ == "__main__":
    main()
