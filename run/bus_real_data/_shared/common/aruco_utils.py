#!/usr/bin/env python3
import cv2
import numpy as np


def get_aruco_dict(dict_name="DICT_4X4_50"):
    if not hasattr(cv2.aruco, dict_name):
        raise RuntimeError(f"OpenCV has no aruco dictionary {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def make_aruco_detector(dict_name="DICT_4X4_50"):
    dictionary = get_aruco_dict(dict_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    params = cv2.aruco.DetectorParameters_create()

    def detect(gray):
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    return detect


def marker_object_points(marker_length_m):
    s = float(marker_length_m) / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)


def detect_markers_in_image(image_path, dict_name="DICT_4X4_50"):
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect = make_aruco_detector(dict_name)
    corners, ids, rejected = detect(gray)

    if ids is None:
        return img, []

    ids = ids.reshape(-1)
    detections = []
    for idx, marker_id in enumerate(ids.tolist()):
        pts = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
        detections.append({
            "marker_id": int(marker_id),
            "corners_px": pts,
            "area_px2": float(cv2.contourArea(pts.astype(np.float32))),
        })
    return img, detections


def solve_marker_pnp(corners_px, marker_length_m, K, D=None):
    if D is None:
        D = np.zeros(5, dtype=np.float64)

    obj = marker_object_points(marker_length_m)
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not ok:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.reshape(3).astype(np.float64), rvec.reshape(3).astype(np.float64)


def draw_marker_debug(image, detections):
    out = image.copy()
    for det in detections:
        pts = det["corners_px"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        c = det["corners_px"].mean(axis=0).astype(int)
        cv2.putText(out, str(det["marker_id"]), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out
