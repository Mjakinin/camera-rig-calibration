#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


TOPIC_MAP = {
    "/edge_0/camera/color/image_raw/compressed": {
        "camera": "cam_edge_5",
        "physical": "back_right",
    },
    "/edge_1/camera/color/image_raw/compressed": {
        "camera": "cam_edge_1",
        "physical": "front_right",
    },
    "/edge_3/camera/color/image_raw/compressed": {
        "camera": "cam_edge_3",
        "physical": "front_left",
    },
    "/edge_5/camera/color/image_raw/compressed": {
        "camera": "cam_edge_0",
        "physical": "center_left",
    },
}

CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]

FIELDS = [
    "observer_type",
    "observer_id",
    "camera_name",
    "frame_id",
    "image_path",
    "marker_id",
    "marker_length_m",
    "fx",
    "fy",
    "cx",
    "cy",
    "pnp_success",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
    "distance_m",
    "center_u",
    "center_v",
    "area_px2",
    "corner0_u",
    "corner0_v",
    "corner1_u",
    "corner1_v",
    "corner2_u",
    "corner2_v",
    "corner3_u",
    "corner3_v",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan every static-camera frame in the MCAP, select a compact "
            "multi-frame set that covers every observed marker, and combine "
            "the selected static observations with the existing moving-camera "
            "observations."
        )
    )
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--moving-observations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker-length-m", type=float, default=0.17)
    parser.add_argument("--min-marker-id", type=int, default=0)
    parser.add_argument("--max-marker-id", type=int, default=20)
    parser.add_argument(
        "--target-observations-per-marker",
        type=int,
        default=2,
        help=(
            "Try to retain this many independent static frames per marker. "
            "Every available marker is covered once before redundancy is added."
        ),
    )
    parser.add_argument(
        "--max-frames-per-camera",
        type=int,
        default=8,
    )
    return parser.parse_args()


def load_camera_info(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())

    K_values = data.get("K", data.get("k"))
    D_values = data.get("D", data.get("d"))

    if K_values is None or D_values is None:
        raise RuntimeError(f"Missing K/D in {path}")

    K = np.asarray(K_values, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D_values, dtype=np.float64).reshape(-1)
    return K, D


def dictionary_from_name(name: str):
    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown OpenCV ArUco dictionary: {name}")

    dictionary_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def make_detector(dictionary_name: str):
    dictionary = dictionary_from_name(dictionary_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    parameters = cv2.aruco.DetectorParameters_create()
    return (dictionary, parameters)


def detect(image: np.ndarray, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        return [], [], rejected

    ids_flat = [int(value) for value in ids.reshape(-1)]
    return corners, ids_flat, rejected


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError("Could not decode CompressedImage")

    return image


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    return abs(float(cv2.contourArea(points.astype(np.float32))))


def marker_object_points(length_m: float) -> np.ndarray:
    half = length_m / 2.0

    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing CSV: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_frames(
    candidates: list[dict],
    target_per_marker: int,
    maximum: int,
) -> tuple[list[dict], set[int], dict[int, int]]:
    marker_union = set()

    for candidate in candidates:
        marker_union.update(candidate["marker_ids"])

    selected = []
    selected_indices = set()
    counts = defaultdict(int)

    while len(selected) < maximum:
        deficits_exist = any(
            counts[marker_id] < target_per_marker
            for marker_id in marker_union
        )

        if not deficits_exist:
            break

        best_index = None
        best_score = None

        for index, candidate in enumerate(candidates):
            if index in selected_indices:
                continue

            marker_ids = candidate["marker_ids"]

            new_unique = sum(
                1
                for marker_id in marker_ids
                if counts[marker_id] == 0
            )

            deficit_gain = sum(
                1
                for marker_id in marker_ids
                if counts[marker_id] < target_per_marker
            )

            if deficit_gain == 0:
                continue

            deficit_area = sum(
                candidate["areas_by_marker"].get(marker_id, 0.0)
                for marker_id in marker_ids
                if counts[marker_id] < target_per_marker
            )

            score = (
                new_unique,
                deficit_gain,
                deficit_area,
                candidate["sharpness"],
            )

            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            break

        selected_indices.add(best_index)
        chosen = candidates[best_index]
        selected.append(chosen)

        for marker_id in chosen["marker_ids"]:
            counts[marker_id] += 1

    return selected, marker_union, dict(counts)


def observation_rows(
    image: np.ndarray,
    image_path: Path,
    camera: str,
    frame_id: str,
    detections: list[dict],
    K: np.ndarray,
    D: np.ndarray,
    marker_length_m: float,
) -> list[dict]:
    rows = []
    object_points = marker_object_points(marker_length_m)

    for detection in detections:
        marker_id = detection["marker_id"]
        points = detection["corners"].astype(np.float32).reshape(4, 2)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            points,
            K,
            D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                points,
                K,
                D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if ok:
            rvec = rvec.reshape(3)
            tvec = tvec.reshape(3)
            distance = float(np.linalg.norm(tvec))
        else:
            rvec = np.full(3, np.nan)
            tvec = np.full(3, np.nan)
            distance = float("nan")

        center = points.mean(axis=0)
        area = polygon_area(points)

        row = {
            "observer_type": "static",
            "observer_id": camera,
            "camera_name": camera,
            "frame_id": frame_id,
            "image_path": str(image_path),
            "marker_id": marker_id,
            "marker_length_m": marker_length_m,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "pnp_success": bool(ok),
            "rvec_x": float(rvec[0]) if ok else "",
            "rvec_y": float(rvec[1]) if ok else "",
            "rvec_z": float(rvec[2]) if ok else "",
            "tvec_x_m": float(tvec[0]) if ok else "",
            "tvec_y_m": float(tvec[1]) if ok else "",
            "tvec_z_m": float(tvec[2]) if ok else "",
            "distance_m": distance,
            "center_u": float(center[0]),
            "center_v": float(center[1]),
            "area_px2": area,
        }

        for corner_index in range(4):
            row[f"corner{corner_index}_u"] = float(points[corner_index, 0])
            row[f"corner{corner_index}_v"] = float(points[corner_index, 1])

        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()

    mcap = Path(args.mcap).resolve()
    dataset = Path(args.dataset).resolve()
    moving_csv = Path(args.moving_observations).resolve()
    out = Path(args.out).resolve()

    if not mcap.is_file():
        raise RuntimeError(f"MCAP not found: {mcap}")

    camera_info_dir = dataset / "raw_images" / "camera_info"
    selected_image_root = dataset / "raw_images" / "static_multi"

    shutil.rmtree(selected_image_root, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)

    selected_image_root.mkdir(parents=True)
    out.mkdir(parents=True)

    debug_root = out / "debug_images" / "static_multi"
    debug_root.mkdir(parents=True)

    detector = make_detector(args.dictionary)

    camera_models = {
        camera: load_camera_info(
            camera_info_dir / f"{camera}.json"
        )
        for camera in CAMERAS
    }

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(mcap),
            storage_id="mcap",
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="",
            output_serialization_format="",
        ),
    )

    topic_types = {
        item.name: item.type
        for item in reader.get_all_topics_and_types()
    }

    for topic in TOPIC_MAP:
        actual_type = topic_types.get(topic)

        if actual_type != "sensor_msgs/msg/CompressedImage":
            raise RuntimeError(
                f"Required topic missing or wrong type: {topic}; "
                f"found={actual_type!r}"
            )

    candidates_by_camera = {
        camera: []
        for camera in CAMERAS
    }

    unexpected_ids = defaultdict(set)

    print("[INFO] scanning every static-camera frame")

    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()

        mapping = TOPIC_MAP.get(topic)

        if mapping is None:
            continue

        camera = mapping["camera"]
        frame_index = len(candidates_by_camera[camera])

        message = deserialize_message(
            serialized,
            CompressedImage,
        )

        compressed = bytes(message.data)
        image = decode_image(compressed)

        corners, marker_ids, _ = detect(
            image,
            detector,
        )

        accepted_detections = []
        areas_by_marker = {}

        for marker_corners, marker_id in zip(corners, marker_ids):
            marker_id = int(marker_id)

            if not (
                args.min_marker_id
                <= marker_id
                <= args.max_marker_id
            ):
                unexpected_ids[camera].add(marker_id)
                continue

            points = np.asarray(
                marker_corners,
                dtype=np.float64,
            ).reshape(4, 2)

            area = polygon_area(points)

            accepted_detections.append(
                {
                    "marker_id": marker_id,
                    "corners": points,
                    "area_px2": area,
                }
            )

            areas_by_marker[marker_id] = max(
                areas_by_marker.get(marker_id, 0.0),
                area,
            )

        candidates_by_camera[camera].append(
            {
                "camera": camera,
                "physical": mapping["physical"],
                "topic": topic,
                "frame_index": frame_index,
                "timestamp_ns": int(timestamp_ns),
                "compressed": compressed,
                "format": str(message.format),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "sharpness": sharpness(image),
                "marker_ids": set(areas_by_marker),
                "areas_by_marker": areas_by_marker,
                "detections": accepted_detections,
            }
        )

    moving_rows = read_csv(moving_csv)
    moving_marker_union = {
        int(float(row["marker_id"]))
        for row in moving_rows
        if row.get("pnp_success") == "True"
        and args.min_marker_id
        <= int(float(row["marker_id"]))
        <= args.max_marker_id
    }

    selected_static_rows = []
    selection_rows = []
    report_lines = [
        "MULTI-FRAME STATIC ARUCO SELECTION",
        "=" * 78,
        "",
        f"Source MCAP: {mcap}",
        f"Dataset: {dataset}",
        f"Dictionary: {args.dictionary}",
        f"Marker length [m]: {args.marker_length_m}",
        (
            "Allowed marker IDs: "
            f"{args.min_marker_id}..{args.max_marker_id}"
        ),
        (
            "Target observations per static marker: "
            f"{args.target_observations_per_marker}"
        ),
        (
            "Maximum selected frames per camera: "
            f"{args.max_frames_per_camera}"
        ),
        "",
        f"Moving-camera marker union: {sorted(moving_marker_union)}",
        "",
    ]

    for camera in CAMERAS:
        candidates = candidates_by_camera[camera]

        selected, marker_union, selected_counts = select_frames(
            candidates,
            target_per_marker=(
                args.target_observations_per_marker
            ),
            maximum=args.max_frames_per_camera,
        )

        camera_dir = selected_image_root / camera
        camera_debug_dir = debug_root / camera

        camera_dir.mkdir(parents=True, exist_ok=True)
        camera_debug_dir.mkdir(parents=True, exist_ok=True)

        K, D = camera_models[camera]

        report_lines.extend(
            [
                camera,
                f"  physical camera: {candidates[0]['physical']}",
                f"  available frames: {len(candidates)}",
                f"  marker union across all frames: {sorted(marker_union)}",
                (
                    "  shared with moving camera: "
                    f"{sorted(marker_union & moving_marker_union)}"
                ),
                f"  selected frames: {len(selected)}",
            ]
        )

        for selection_index, candidate in enumerate(selected):
            image = decode_image(candidate["compressed"])

            filename = (
                f"frame_{candidate['frame_index']:04d}"
                f"_t{candidate['timestamp_ns']}.png"
            )

            image_path = camera_dir / filename
            debug_path = camera_debug_dir / filename

            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(
                    f"Could not write {image_path}"
                )

            debug = image.copy()

            debug_corners = [
                detection["corners"].astype(np.float32).reshape(1, 4, 2)
                for detection in candidate["detections"]
            ]

            debug_ids = np.asarray(
                [
                    detection["marker_id"]
                    for detection in candidate["detections"]
                ],
                dtype=np.int32,
            ).reshape(-1, 1)

            if len(debug_ids):
                cv2.aruco.drawDetectedMarkers(
                    debug,
                    debug_corners,
                    debug_ids,
                )

            cv2.putText(
                debug,
                (
                    f"{camera} source_frame={candidate['frame_index']} "
                    f"markers={sorted(candidate['marker_ids'])}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(str(debug_path), debug)

            frame_id = f"static_{candidate['frame_index']:04d}"

            rows = observation_rows(
                image=image,
                image_path=image_path,
                camera=camera,
                frame_id=frame_id,
                detections=candidate["detections"],
                K=K,
                D=D,
                marker_length_m=args.marker_length_m,
            )

            selected_static_rows.extend(rows)

            selection_row = {
                "camera": camera,
                "physical_camera": candidate["physical"],
                "source_topic": candidate["topic"],
                "source_frame_index": candidate["frame_index"],
                "timestamp_ns": candidate["timestamp_ns"],
                "selected_order": selection_index,
                "width": candidate["width"],
                "height": candidate["height"],
                "sharpness": candidate["sharpness"],
                "marker_ids": " ".join(
                    str(value)
                    for value in sorted(candidate["marker_ids"])
                ),
                "saved_image": str(image_path),
                "debug_image": str(debug_path),
            }

            selection_rows.append(selection_row)

            report_lines.append(
                (
                    f"    frame {candidate['frame_index']:4d}: "
                    f"markers={sorted(candidate['marker_ids'])}, "
                    f"sharpness={candidate['sharpness']:.2f}"
                )
            )

        report_lines.extend(
            [
                (
                    "  retained observations per marker: "
                    + str(
                        {
                            marker_id: selected_counts.get(marker_id, 0)
                            for marker_id in sorted(marker_union)
                        }
                    )
                ),
                (
                    "  unexpected IDs rejected: "
                    f"{sorted(unexpected_ids[camera])}"
                ),
                "",
            ]
        )

    filtered_moving_rows = [
        row
        for row in moving_rows
        if args.min_marker_id
        <= int(float(row["marker_id"]))
        <= args.max_marker_id
    ]

    all_rows = selected_static_rows + filtered_moving_rows

    write_csv(
        out / "shared_static_aruco_observations.csv",
        selected_static_rows,
        FIELDS,
    )

    write_csv(
        out / "shared_moving_aruco_observations.csv",
        filtered_moving_rows,
        FIELDS,
    )

    write_csv(
        out / "shared_all_aruco_observations.csv",
        all_rows,
        FIELDS,
    )

    if selection_rows:
        write_csv(
            out / "SELECTED_STATIC_FRAMES.csv",
            selection_rows,
            list(selection_rows[0].keys()),
        )

    static_marker_union = {
        int(row["marker_id"])
        for row in selected_static_rows
        if str(row["pnp_success"]) == "True"
    }

    report_lines.extend(
        [
            "Combined observation summary",
            "-" * 78,
            f"Selected static observations: {len(selected_static_rows)}",
            f"Moving observations retained: {len(filtered_moving_rows)}",
            f"Combined observations: {len(all_rows)}",
            f"Selected static marker union: {sorted(static_marker_union)}",
            f"Moving marker union: {sorted(moving_marker_union)}",
            (
                "Static/moving shared marker union: "
                f"{sorted(static_marker_union & moving_marker_union)}"
            ),
            "",
            (
                "Note: all selected frames from one static camera use the "
                "same observer_id. They add repeated observations of one fixed pose; "
                "they do not create additional cameras."
            ),
            "",
            "[OK] multi-frame static observation set prepared",
        ]
    )

    report_path = out / "MULTI_STATIC_ARUCO_SUMMARY.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(report_path.read_text())


if __name__ == "__main__":
    main()
