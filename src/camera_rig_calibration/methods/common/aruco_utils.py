"""Shared, versioned ArUco detection and pose helpers."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DETECTOR_CONTRACT = "rigcal_aruco_detector_v2"
DETECTION_MODES = ("baseline", "subpixel_refined", "high_sensitivity")
HIGH_SENSITIVITY_GAMMAS = (0.60, 0.65)


def get_aruco_dict(dict_name="DICT_4X4_50"):
    if not hasattr(cv2.aruco, dict_name):
        raise RuntimeError(f"OpenCV has no aruco dictionary {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def _new_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def _set_parameter(parameters, name: str, value: Any) -> None:
    if hasattr(parameters, name):
        setattr(parameters, name, value)


def _parameters(mode: str):
    parameters = _new_parameters()
    if mode in {"subpixel_refined", "high_sensitivity"}:
        _set_parameter(
            parameters,
            "cornerRefinementMethod",
            cv2.aruco.CORNER_REFINE_SUBPIX,
        )
    if mode == "high_sensitivity":
        for name, value in {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 53,
            "adaptiveThreshWinSizeStep": 4,
            "minMarkerPerimeterRate": 0.008,
            "minCornerDistanceRate": 0.01,
            "minMarkerDistanceRate": 0.01,
            "minDistanceToBorder": 1,
            "polygonalApproxAccuracyRate": 0.05,
            "errorCorrectionRate": 0.8,
        }.items():
            _set_parameter(parameters, name, value)
    return parameters


def effective_detector_config(
    mode: str,
    dictionary: str = "DICT_4X4_50",
) -> dict[str, Any]:
    if mode not in DETECTION_MODES:
        raise ValueError(
            f"Unknown ArUco detection mode '{mode}'; choose "
            + ", ".join(DETECTION_MODES)
        )
    result: dict[str, Any] = {
        "contract": DETECTOR_CONTRACT,
        "mode": mode,
        "dictionary": dictionary,
        "opencv_version": cv2.__version__,
    }
    if mode == "baseline":
        result["parameters"] = "opencv_defaults"
    elif mode == "subpixel_refined":
        result["parameters"] = {
            "base": "opencv_defaults",
            "cornerRefinementMethod": "CORNER_REFINE_SUBPIX",
        }
    else:
        result.update(
            {
                "parameters": {
                    "adaptiveThreshWinSizeMin": 3,
                    "adaptiveThreshWinSizeMax": 53,
                    "adaptiveThreshWinSizeStep": 4,
                    "minMarkerPerimeterRate": 0.008,
                    "minCornerDistanceRate": 0.01,
                    "minMarkerDistanceRate": 0.01,
                    "minDistanceToBorder": 1,
                    "polygonalApproxAccuracyRate": 0.05,
                    "errorCorrectionRate": 0.8,
                    "cornerRefinementMethod": "CORNER_REFINE_SUBPIX",
                },
                "gamma_passes": list(HIGH_SENSITIVITY_GAMMAS),
                "support_rule": {
                    "same_marker_id": True,
                    "maximum_center_distance_image_diagonal_fraction": 0.01,
                    "maximum_area_ratio": 2.0,
                    "minimum_area_image_fraction": 1e-4,
                },
            }
        )
    return result


def _opencv_detector(dictionary, parameters):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    def detect(gray):
        return cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    return detect


def _raw_detection(gray, dictionary, mode: str):
    return _opencv_detector(dictionary, _parameters(mode))(gray)


def _candidate_rows(corners, ids, *, pass_name: str) -> list[dict[str, Any]]:
    if ids is None:
        return []
    result = []
    for index, marker_id in enumerate(np.asarray(ids).reshape(-1).tolist()):
        points = np.asarray(corners[index], dtype=np.float64).reshape(4, 2)
        result.append(
            {
                "marker_id": int(marker_id),
                "corners_px": points,
                "area_px2": abs(
                    float(cv2.contourArea(points.astype(np.float32)))
                ),
                "center_px": points.mean(axis=0),
                "pass": pass_name,
            }
        )
    return result


def _gamma_image(gray: np.ndarray, gamma: float) -> np.ndarray:
    table = np.asarray(
        [round(((value / 255.0) ** gamma) * 255.0) for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, table)


def _same_candidate(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    maximum_center_distance: float,
) -> bool:
    if left["marker_id"] != right["marker_id"]:
        return False
    center_distance = float(
        np.linalg.norm(left["center_px"] - right["center_px"])
    )
    if center_distance > maximum_center_distance:
        return False
    small_area = min(left["area_px2"], right["area_px2"])
    large_area = max(left["area_px2"], right["area_px2"])
    return small_area > 0 and large_area / small_area <= 2.0


def detect_markers_with_diagnostics(
    image: np.ndarray,
    *,
    dict_name: str = "DICT_4X4_50",
    detection_mode: str = "baseline",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect markers and return accepted detections plus candidate evidence."""
    effective_detector_config(detection_mode, dict_name)
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    dictionary = get_aruco_dict(dict_name)
    base_mode = (
        "subpixel_refined"
        if detection_mode in {"subpixel_refined", "high_sensitivity"}
        else "baseline"
    )
    corners, ids, _ = _raw_detection(gray, dictionary, base_mode)
    baseline = _candidate_rows(corners, ids, pass_name="original")
    accepted = [dict(item, detection_support=1) for item in baseline]
    diagnostics = [
        {
            "pass": "original",
            "marker_id": item["marker_id"],
            "area_px2": item["area_px2"],
            "center_u": float(item["center_px"][0]),
            "center_v": float(item["center_px"][1]),
            "accepted": True,
            "reason": "original_pass",
        }
        for item in baseline
    ]
    if detection_mode != "high_sensitivity":
        for item in accepted:
            item["detection_source"] = "original"
        return accepted, diagnostics

    pass_candidates: list[list[dict[str, Any]]] = []
    for gamma in HIGH_SENSITIVITY_GAMMAS:
        gamma_corners, gamma_ids, _ = _raw_detection(
            _gamma_image(gray, gamma),
            dictionary,
            "high_sensitivity",
        )
        pass_candidates.append(
            _candidate_rows(
                gamma_corners,
                gamma_ids,
                pass_name=f"gamma_{gamma:.2f}",
            )
        )

    image_area = float(gray.shape[0] * gray.shape[1])
    maximum_center_distance = 0.01 * float(
        np.hypot(gray.shape[0], gray.shape[1])
    )
    consumed_right: set[int] = set()
    supported_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for left in pass_candidates[0]:
        best_index = None
        best_distance = float("inf")
        for index, right in enumerate(pass_candidates[1]):
            if index in consumed_right or not _same_candidate(
                left,
                right,
                maximum_center_distance=maximum_center_distance,
            ):
                continue
            distance = float(
                np.linalg.norm(left["center_px"] - right["center_px"])
            )
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is not None:
            consumed_right.add(best_index)
            supported_pairs.append((left, pass_candidates[1][best_index]))

    supported_objects = {id(item) for pair in supported_pairs for item in pair}
    for pass_items in pass_candidates:
        for item in pass_items:
            large_enough = item["area_px2"] >= 1e-4 * image_area
            supported = id(item) in supported_objects
            diagnostics.append(
                {
                    "pass": item["pass"],
                    "marker_id": item["marker_id"],
                    "area_px2": item["area_px2"],
                    "center_u": float(item["center_px"][0]),
                    "center_v": float(item["center_px"][1]),
                    "accepted": bool(supported and large_enough),
                    "reason": (
                        "confirmed_by_both_gamma_passes"
                        if supported and large_enough
                        else "area_below_minimum"
                        if supported
                        else "not_confirmed_by_both_gamma_passes"
                    ),
                }
            )

    for left, right in supported_pairs:
        area = min(left["area_px2"], right["area_px2"])
        if area < 1e-4 * image_area:
            continue
        merged_corners = (
            left["corners_px"] + right["corners_px"]
        ) / 2.0
        merged = {
            "marker_id": left["marker_id"],
            "corners_px": merged_corners,
            "area_px2": abs(
                float(cv2.contourArea(merged_corners.astype(np.float32)))
            ),
            "center_px": merged_corners.mean(axis=0),
            "pass": "gamma_consensus",
            "detection_source": "gamma_consensus",
            "detection_support": 2,
        }
        duplicate = any(
            _same_candidate(
                existing,
                merged,
                maximum_center_distance=maximum_center_distance,
            )
            for existing in accepted
        )
        if not duplicate:
            accepted.append(merged)

    for item in accepted:
        item.setdefault("detection_source", "original")
    accepted.sort(
        key=lambda item: (
            int(item["marker_id"]),
            float(item["center_px"][0]),
            float(item["center_px"][1]),
        )
    )
    return accepted, diagnostics


def make_aruco_detector(
    dict_name="DICT_4X4_50",
    detection_mode: str = "baseline",
):
    def detect(gray):
        detections, _ = detect_markers_with_diagnostics(
            gray,
            dict_name=dict_name,
            detection_mode=detection_mode,
        )
        corners = [
            np.asarray(item["corners_px"], dtype=np.float32).reshape(1, 4, 2)
            for item in detections
        ]
        ids = (
            np.asarray(
                [[int(item["marker_id"])] for item in detections],
                dtype=np.int32,
            )
            if detections
            else None
        )
        return corners, ids, []

    return detect


def marker_object_points(marker_length_m):
    s = float(marker_length_m) / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)


def detect_markers_in_image(
    image_path,
    dict_name="DICT_4X4_50",
    detection_mode: str = "baseline",
    *,
    return_diagnostics: bool = False,
):
    image_path = str(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    detections, diagnostics = detect_markers_with_diagnostics(
        img,
        dict_name=dict_name,
        detection_mode=detection_mode,
    )
    result = []
    for item in detections:
        result.append(
            {
                "marker_id": int(item["marker_id"]),
                "corners_px": np.asarray(
                    item["corners_px"], dtype=np.float64
                ).reshape(4, 2),
                "area_px2": float(item["area_px2"]),
                "detection_source": item["detection_source"],
                "detection_support": int(item["detection_support"]),
            }
        )
    if return_diagnostics:
        return img, result, diagnostics
    return img, result


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
