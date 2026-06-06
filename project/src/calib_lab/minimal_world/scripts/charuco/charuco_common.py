#!/usr/bin/env python3

import cv2
import numpy as np


def get_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is not available. Install/use OpenCV contrib.")
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def create_detector(dictionary):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None


def create_charuco_board(dictionary, squares_x, squares_y, square_length, marker_length):
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_length,
            marker_length,
            dictionary,
        )
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        dictionary,
    )


def detect_markers_once(gray, dictionary, detector):
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
    return corners, ids, rejected


def image_variants(gray):
    variants = [("gray", gray)]

    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    variants.append(("normalized", normalized))

    for threshold_value in [40, 60, 80, 100]:
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        variants.append((f"binary_{threshold_value}", binary))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    variants.append(("adaptive", adaptive))
    return variants


def detect_charuco(image, dictionary, detector, board, camera_matrix=None, dist_coeffs=None, use_preprocessing=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = image_variants(gray) if use_preprocessing else [("gray", gray)]

    best = None

    for method_name, image_variant in variants:
        corners, ids, rejected = detect_markers_once(image_variant, dictionary, detector)
        marker_count = 0 if ids is None else len(ids)

        if ids is None or len(ids) == 0:
            candidate = {
                "method": method_name,
                "corners": corners,
                "ids": ids,
                "rejected": rejected,
                "charuco_corners": None,
                "charuco_ids": None,
                "marker_count": marker_count,
                "charuco_count": 0,
                "gray": image_variant,
            }
            if best is None:
                best = candidate
            continue

        kwargs = {}
        if camera_matrix is not None:
            kwargs["cameraMatrix"] = camera_matrix
        if dist_coeffs is not None:
            kwargs["distCoeffs"] = dist_coeffs

        try:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                image_variant,
                board,
                **kwargs,
            )
        except TypeError:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                image_variant,
                board,
            )

        charuco_count = 0 if charuco_ids is None else len(charuco_ids)
        candidate = {
            "method": method_name,
            "corners": corners,
            "ids": ids,
            "rejected": rejected,
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
            "marker_count": marker_count,
            "charuco_count": charuco_count,
            "gray": image_variant,
        }

        if best is None or charuco_count > best["charuco_count"]:
            best = candidate

        if charuco_count > 0:
            return candidate

    if best is None:
        return {
            "method": "none",
            "corners": [],
            "ids": None,
            "rejected": None,
            "charuco_corners": None,
            "charuco_ids": None,
            "marker_count": 0,
            "charuco_count": 0,
            "gray": gray,
        }
    return best

def get_board_chessboard_corners(board):
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        corners = board.chessboardCorners
    else:
        raise RuntimeError("Could not access ChArUco board chessboard corners from OpenCV board object.")

    return np.asarray(corners, dtype=np.float32).reshape(-1, 3)


def build_charuco_correspondences(board, charuco_corners, charuco_ids):
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return None, None, []

    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )

        object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        used_ids = charuco_ids.flatten().astype(int).tolist()

        return object_points, image_points, used_ids

    board_corners = get_board_chessboard_corners(board)
    ids_flat = charuco_ids.flatten().astype(int)

    object_points = []
    image_points = []
    used_ids = []

    img_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)

    for i, corner_id in enumerate(ids_flat):
        if corner_id < 0 or corner_id >= len(board_corners):
            continue

        object_points.append(board_corners[corner_id])
        image_points.append(img_points[i])
        used_ids.append(int(corner_id))

    if not object_points:
        return None, None, []

    return (
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        used_ids,
    )

def draw_charuco_detection(image, detection, camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None, axis_length=0.15):
    annotated = image.copy()
    corners = detection.get("corners")
    ids = detection.get("ids")
    charuco_corners = detection.get("charuco_corners")
    charuco_ids = detection.get("charuco_ids")

    if ids is not None and corners is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

    if charuco_ids is not None and charuco_corners is not None and len(charuco_ids) > 0:
        try:
            cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
        except Exception:
            pts = np.asarray(charuco_corners).reshape(-1, 2).astype(int)
            for p in pts:
                cv2.circle(annotated, tuple(p), 3, (0, 255, 255), -1)

    if rvec is not None and tvec is not None and camera_matrix is not None and dist_coeffs is not None:
        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

    return annotated

def estimate_charuco_pose_native(board, detection, camera_matrix, dist_coeffs):
    charuco_corners = detection.get("charuco_corners")
    charuco_ids = detection.get("charuco_ids")

    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return False, None, None, "no_charuco_corners"

    if not hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        return False, None, None, "estimatePoseCharucoBoard_not_available"

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
    )

    if not ok:
        return False, None, None, "estimatePoseCharucoBoard_failed"

    return True, rvec, tvec, "charuco_native"
