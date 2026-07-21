#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


# Preserve canonical physical camera IDs used by the project.
# Current rosbag device assignments edge_0 and edge_5 are swapped:
#   /edge_0/ contains the previous cam_edge_5 device (back_right)
#   /edge_5/ contains the previous cam_edge_0 device (center_left)
TOPIC_MAP = {
    "/edge_0/camera/color/image_raw/compressed": {
        "canonical_camera": "cam_edge_5",
        "physical_camera": "back_right",
    },
    "/edge_1/camera/color/image_raw/compressed": {
        "canonical_camera": "cam_edge_1",
        "physical_camera": "front_right",
    },
    "/edge_3/camera/color/image_raw/compressed": {
        "canonical_camera": "cam_edge_3",
        "physical_camera": "front_left",
    },
    "/edge_5/camera/color/image_raw/compressed": {
        "canonical_camera": "cam_edge_0",
        "physical_camera": "center_left",
    },
}

CANONICAL_ORDER = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)

    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


def detect_markers(image: np.ndarray, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, params = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=params,
        )

    marker_ids = []
    total_area = 0.0

    if ids is not None:
        marker_ids = [int(value) for value in ids.reshape(-1)]

        for corner in corners:
            points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
            total_area += abs(float(cv2.contourArea(points.astype(np.float32))))

    return corners, ids, sorted(marker_ids), total_area


def image_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError("Could not decode CompressedImage")

    return image


def make_thumbnail(
    image: np.ndarray,
    title: str,
    details: list[str],
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)

    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    x = (width - resized_width) // 2
    y = (height - resized_height) // 2

    canvas[y:y + resized_height, x:x + resized_width] = resized

    lines = [title, *details]
    text_y = 35

    for line in lines:
        cv2.putText(
            canvas,
            line,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        text_y += 32

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--save-all-candidates",
        action="store_true",
        help="Save each decoded static frame as a JPEG for manual inspection.",
    )
    args = parser.parse_args()

    mcap = Path(args.mcap).resolve()
    dataset = Path(args.dataset).resolve()

    if not mcap.is_file():
        raise RuntimeError(f"MCAP not found: {mcap}")

    static_dir = dataset / "raw_images" / "static"
    preview_root = dataset / "preview" / "static_candidates"
    selected_preview_dir = dataset / "preview" / "static_selected"
    metadata_dir = dataset / "metadata" / "static_extraction"

    static_dir.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(preview_root, ignore_errors=True)
    shutil.rmtree(selected_preview_dir, ignore_errors=True)
    shutil.rmtree(metadata_dir, ignore_errors=True)

    preview_root.mkdir(parents=True)
    selected_preview_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    detector = make_detector()

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

    available = {
        item.name: item.type
        for item in reader.get_all_topics_and_types()
    }

    for topic in TOPIC_MAP:
        actual_type = available.get(topic)

        if actual_type != "sensor_msgs/msg/CompressedImage":
            raise RuntimeError(
                f"Required topic missing or wrong type: {topic}; "
                f"found={actual_type!r}"
            )

    records = {camera: [] for camera in CANONICAL_ORDER}
    best = {}

    print("[INFO] topic remapping")

    for topic, cfg in TOPIC_MAP.items():
        print(
            f"  {topic} -> {cfg['canonical_camera']} "
            f"({cfg['physical_camera']})"
        )

    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()

        if topic not in TOPIC_MAP:
            continue

        cfg = TOPIC_MAP[topic]
        camera = cfg["canonical_camera"]
        physical = cfg["physical_camera"]

        message = deserialize_message(serialized, CompressedImage)
        compressed = bytes(message.data)
        image = decode_image(compressed)

        corners, ids, marker_ids, marker_area = detect_markers(
            image,
            detector,
        )

        sharpness = image_sharpness(image)
        frame_index = len(records[camera])

        score = (
            len(marker_ids),
            marker_area,
            sharpness,
        )

        candidate_path = (
            preview_root
            / camera
            / f"frame_{frame_index:04d}_t{int(timestamp_ns)}.jpg"
        )

        if args.save_all_candidates:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)

            debug = image.copy()

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(debug, corners, ids)

            cv2.putText(
                debug,
                (
                    f"{camera} physical={physical} "
                    f"markers={marker_ids} sharp={sharpness:.1f}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(
                str(candidate_path),
                debug,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

        record = {
            "canonical_camera": camera,
            "physical_camera": physical,
            "source_topic": topic,
            "frame_index": frame_index,
            "timestamp_ns": int(timestamp_ns),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "marker_count": len(marker_ids),
            "marker_ids": " ".join(str(value) for value in marker_ids),
            "marker_total_area_px2": marker_area,
            "sharpness": sharpness,
            "candidate_preview": str(candidate_path) if args.save_all_candidates else "",
        }

        records[camera].append(record)

        current_best = best.get(camera)

        if current_best is None or score > current_best["score"]:
            best[camera] = {
                "score": score,
                "record": record,
                "compressed": compressed,
                "format": str(message.format),
            }

    for camera in CANONICAL_ORDER:
        if not records[camera]:
            raise RuntimeError(f"No images read for {camera}")

    report_lines = [
        "REAL STATIC IMAGE EXTRACTION",
        "=" * 72,
        "",
        f"Source MCAP: {mcap}",
        "",
        "Canonical remapping:",
        "  /edge_0/ -> cam_edge_5 (back_right)",
        "  /edge_1/ -> cam_edge_1 (front_right)",
        "  /edge_3/ -> cam_edge_3 (front_left)",
        "  /edge_5/ -> cam_edge_0 (center_left)",
        "",
        "Important:",
        "  No static-camera intrinsics were read from the rosbag.",
        "  Static intrinsics must be installed separately from camera_info.zip.",
        "",
    ]

    selected_rows = []
    thumbnails = []

    for camera in CANONICAL_ORDER:
        selected = best[camera]
        record = selected["record"]
        image = decode_image(selected["compressed"])

        corners, ids, marker_ids, _ = detect_markers(image, detector)

        output_path = static_dir / f"{camera}.png"

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not write {output_path}")

        preview_path = selected_preview_dir / f"{camera}.jpg"
        debug = image.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)

        cv2.putText(
            debug,
            (
                f"{camera} physical={record['physical_camera']} "
                f"source={record['source_topic']}"
            ),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(preview_path),
            debug,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        selected_row = dict(record)
        selected_row["pipeline_image"] = str(output_path)
        selected_row["selected_preview"] = str(preview_path)
        selected_rows.append(selected_row)

        thumbnails.append(
            make_thumbnail(
                debug,
                camera,
                [
                    f"physical: {record['physical_camera']}",
                    f"topic: {record['source_topic']}",
                    f"markers: {marker_ids}",
                    f"sharpness: {record['sharpness']:.1f}",
                ],
            )
        )

        report_lines.extend(
            [
                camera,
                f"  physical camera: {record['physical_camera']}",
                f"  source topic: {record['source_topic']}",
                f"  frames available: {len(records[camera])}",
                f"  selected frame index: {record['frame_index']}",
                f"  resolution: {record['width']}x{record['height']}",
                f"  marker IDs: {record['marker_ids'] or 'none'}",
                f"  sharpness: {record['sharpness']:.3f}",
                f"  pipeline image: {output_path}",
                "",
            ]
        )

    contact_sheet = np.vstack(
        [
            np.hstack(thumbnails[:2]),
            np.hstack(thumbnails[2:]),
        ]
    )

    contact_path = selected_preview_dir / "STATIC_CAMERA_CONTACT_SHEET.jpg"
    cv2.imwrite(str(contact_path), contact_sheet)

    selected_csv = metadata_dir / "selected_static_frames.csv"

    with selected_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(selected_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    all_csv = metadata_dir / "all_static_candidates.csv"
    all_rows = []

    for camera in CANONICAL_ORDER:
        all_rows.extend(records[camera])

    with all_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    mapping = {
        "policy": (
            "Preserve canonical physical camera IDs for comparability "
            "with simulation and measured reference distances."
        ),
        "current_topic_to_canonical_camera": {
            topic: cfg["canonical_camera"]
            for topic, cfg in TOPIC_MAP.items()
        },
        "current_topic_to_physical_camera": {
            topic: cfg["physical_camera"]
            for topic, cfg in TOPIC_MAP.items()
        },
        "static_intrinsics_source": "camera_info.zip, not rosbag CameraInfo",
        "moving_intrinsics_source": (
            "0.5x iPhone checkerboard calibration; applies only to moving frames"
        ),
    }

    (metadata_dir / "STATIC_CAMERA_MAPPING.json").write_text(
        json.dumps(mapping, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines.extend(
        [
            f"All candidate previews: {preview_root}",
            f"Selected previews: {selected_preview_dir}",
            f"Contact sheet: {contact_path}",
            f"Selected-frame CSV: {selected_csv}",
            f"All-candidate CSV: {all_csv}",
            "",
            "[OK] static images extracted without using rosbag intrinsics",
        ]
    )

    report_path = metadata_dir / "STATIC_EXTRACTION_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(report_path.read_text())


if __name__ == "__main__":
    main()
