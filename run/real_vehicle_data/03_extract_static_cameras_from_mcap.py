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
from sensor_msgs.msg import CameraInfo, CompressedImage


CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]


def choose_topic(
    topic_types: dict[str, str],
    camera: str,
    message_type: str,
) -> str:
    topic_camera = camera.removeprefix("cam_")
    topic_token = f"/{topic_camera}/"

    candidates = [
        topic
        for topic, topic_type in topic_types.items()
        if topic_token in topic
        and topic_type == message_type
    ]

    candidates = [
        topic
        for topic in candidates
        if "depth" not in topic.lower()
    ]

    if not candidates:
        raise RuntimeError(
            f"No {message_type} topic found for {camera}"
        )

    def score(topic: str) -> tuple[int, int]:
        name = topic.lower()
        value = 0

        if "color" in name:
            value += 20

        if "image_raw" in name:
            value += 10

        if "compressed" in name:
            value += 5

        if "camera_info" in name:
            value += 10

        return value, -len(topic)

    return max(candidates, key=score)


def decode_compressed(data: bytes) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)

    image = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "Could not decode compressed image"
        )

    return image


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()

        return cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

    return dictionary


def detect_marker_ids(
    image: np.ndarray,
    detector,
) -> list[int]:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    if hasattr(cv2.aruco, "ArucoDetector"):
        _, ids, _ = detector.detectMarkers(gray)
    else:
        parameters = (
            cv2.aruco.DetectorParameters_create()
        )

        _, ids, _ = cv2.aruco.detectMarkers(
            gray,
            detector,
            parameters=parameters,
        )

    if ids is None:
        return []

    return sorted(
        int(value)
        for value in ids.reshape(-1)
    )


def nearest_record(records, timestamp_ns: int):
    return min(
        records,
        key=lambda record: abs(
            record[0] - timestamp_ns
        ),
    )


def camera_info_dict(
    camera: str,
    message: CameraInfo,
    mcap: Path,
    image_topic: str,
    info_topic: str,
    image_timestamp_ns: int,
    info_timestamp_ns: int,
    image_width: int,
    image_height: int,
) -> dict:
    K = [float(value) for value in message.k]
    D = [float(value) for value in message.d]
    R = [float(value) for value in message.r]
    P = [float(value) for value in message.p]

    return {
        "camera_name": camera,
        "width": int(message.width),
        "height": int(message.height),
        "image_width": int(message.width),
        "image_height": int(message.height),
        "distortion_model": str(
            message.distortion_model
        ),
        "K": K,
        "k": K,
        "D": D,
        "d": D,
        "R": R,
        "r": R,
        "P": P,
        "p": P,
        "fx": float(K[0]),
        "fy": float(K[4]),
        "cx": float(K[2]),
        "cy": float(K[5]),
        "source_mcap": str(mcap),
        "image_topic": image_topic,
        "camera_info_topic": info_topic,
        "selected_image_timestamp_ns": (
            int(image_timestamp_ns)
        ),
        "camera_info_timestamp_ns": (
            int(info_timestamp_ns)
        ),
        "image_camera_info_offset_ms": (
            abs(
                image_timestamp_ns
                - info_timestamp_ns
            )
            / 1e6
        ),
        "decoded_image_width": image_width,
        "decoded_image_height": image_height,
    }


def thumbnail(
    image: np.ndarray,
    label_lines: list[str],
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    source_height, source_width = image.shape[:2]

    scale = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(
        1,
        int(round(source_width * scale)),
    )

    resized_height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    x = (width - resized_width) // 2
    y = (height - resized_height) // 2

    canvas[
        y:y + resized_height,
        x:x + resized_width,
    ] = resized

    text_y = 35

    for line in label_lines:
        cv2.putText(
            canvas,
            line,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        text_y += 35

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mcap",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--candidate-times",
        type=int,
        default=21,
    )

    args = parser.parse_args()

    mcap = Path(args.mcap).resolve()
    dataset = Path(args.dataset).resolve()

    if not mcap.is_file():
        raise RuntimeError(
            f"MCAP not found: {mcap}"
        )

    static_dir = dataset / "raw_images/static"
    info_dir = dataset / "raw_images/camera_info"
    metadata_dir = dataset / "metadata/static_extraction"

    static_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(
        metadata_dir,
        ignore_errors=True,
    )

    metadata_dir.mkdir(parents=True)

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

    image_topics = {
        camera: choose_topic(
            topic_types,
            camera,
            "sensor_msgs/msg/CompressedImage",
        )
        for camera in CAMERAS
    }

    info_topics = {
        camera: choose_topic(
            topic_types,
            camera,
            "sensor_msgs/msg/CameraInfo",
        )
        for camera in CAMERAS
    }

    print("[INFO] selected topics")

    for camera in CAMERAS:
        print(
            camera,
            "\n  image:",
            image_topics[camera],
            "\n  info: ",
            info_topics[camera],
        )

    image_topic_to_camera = {
        topic: camera
        for camera, topic in image_topics.items()
    }

    info_topic_to_camera = {
        topic: camera
        for camera, topic in info_topics.items()
    }

    image_records = {
        camera: []
        for camera in CAMERAS
    }

    info_records = {
        camera: []
        for camera in CAMERAS
    }

    while reader.has_next():
        topic, serialized, timestamp_ns = (
            reader.read_next()
        )

        camera = image_topic_to_camera.get(topic)

        if camera is not None:
            message = deserialize_message(
                serialized,
                CompressedImage,
            )

            image_records[camera].append(
                (
                    int(timestamp_ns),
                    bytes(message.data),
                )
            )

            continue

        camera = info_topic_to_camera.get(topic)

        if camera is not None:
            message = deserialize_message(
                serialized,
                CameraInfo,
            )

            info_records[camera].append(
                (
                    int(timestamp_ns),
                    message,
                )
            )

    for camera in CAMERAS:
        print(
            f"[INFO] {camera}: "
            f"{len(image_records[camera])} images, "
            f"{len(info_records[camera])} camera_info"
        )

        if not image_records[camera]:
            raise RuntimeError(
                f"No images read for {camera}"
            )

        if not info_records[camera]:
            raise RuntimeError(
                f"No CameraInfo read for {camera}"
            )

    overlap_start = max(
        records[0][0]
        for records in image_records.values()
    )

    overlap_end = min(
        records[-1][0]
        for records in image_records.values()
    )

    if overlap_end <= overlap_start:
        raise RuntimeError(
            "Static camera streams have no common "
            "timestamp interval"
        )

    margin = int(
        0.10 * (overlap_end - overlap_start)
    )

    candidate_start = overlap_start + margin
    candidate_end = overlap_end - margin

    candidate_timestamps = np.linspace(
        candidate_start,
        candidate_end,
        max(args.candidate_times, 3),
        dtype=np.int64,
    )

    aruco_detector = make_aruco_detector()
    candidate_results = []

    for target_timestamp in candidate_timestamps:
        target_timestamp = int(target_timestamp)

        camera_results = {}

        for camera in CAMERAS:
            image_timestamp, compressed = nearest_record(
                image_records[camera],
                target_timestamp,
            )

            image = decode_compressed(compressed)
            marker_ids = detect_marker_ids(
                image,
                aruco_detector,
            )

            camera_results[camera] = {
                "timestamp_ns": image_timestamp,
                "image": image,
                "sharpness": sharpness(image),
                "marker_ids": marker_ids,
            }

        marker_counts = [
            len(camera_results[camera]["marker_ids"])
            for camera in CAMERAS
        ]

        sharpness_values = np.asarray(
            [
                camera_results[camera]["sharpness"]
                for camera in CAMERAS
            ],
            dtype=np.float64,
        )

        log_sharpness = np.log1p(
            sharpness_values
        )

        score = (
            min(marker_counts),
            sum(marker_counts),
            float(np.min(log_sharpness)),
            float(np.median(log_sharpness)),
        )

        candidate_results.append(
            {
                "target_timestamp_ns": (
                    target_timestamp
                ),
                "camera_results": camera_results,
                "score": score,
            }
        )

        print(
            "[CANDIDATE]",
            target_timestamp,
            "markers=",
            marker_counts,
            "score=",
            score,
        )

    selected = max(
        candidate_results,
        key=lambda result: result["score"],
    )

    selected_target = selected[
        "target_timestamp_ns"
    ]

    print()
    print(
        "[SELECTED] target timestamp:",
        selected_target,
    )

    metadata_rows = []
    contact_images = []

    for camera in CAMERAS:
        result = selected[
            "camera_results"
        ][camera]

        image_timestamp = result["timestamp_ns"]
        image = result["image"]
        image_height, image_width = image.shape[:2]

        info_timestamp, info_message = nearest_record(
            info_records[camera],
            image_timestamp,
        )

        image_path = static_dir / f"{camera}.png"
        info_path = info_dir / f"{camera}.json"

        if not cv2.imwrite(
            str(image_path),
            image,
        ):
            raise RuntimeError(
                f"Could not write {image_path}"
            )

        info = camera_info_dict(
            camera=camera,
            message=info_message,
            mcap=mcap,
            image_topic=image_topics[camera],
            info_topic=info_topics[camera],
            image_timestamp_ns=image_timestamp,
            info_timestamp_ns=info_timestamp,
            image_width=image_width,
            image_height=image_height,
        )

        info_path.write_text(
            json.dumps(info, indent=2) + "\n"
        )

        marker_ids = result["marker_ids"]

        metadata_rows.append(
            {
                "camera": camera,
                "target_timestamp_ns": selected_target,
                "image_timestamp_ns": image_timestamp,
                "image_target_offset_ms": (
                    abs(
                        image_timestamp
                        - selected_target
                    )
                    / 1e6
                ),
                "camera_info_timestamp_ns": (
                    info_timestamp
                ),
                "image_info_offset_ms": (
                    abs(
                        image_timestamp
                        - info_timestamp
                    )
                    / 1e6
                ),
                "image_width": image_width,
                "image_height": image_height,
                "camera_info_width": int(
                    info_message.width
                ),
                "camera_info_height": int(
                    info_message.height
                ),
                "sharpness": result["sharpness"],
                "marker_count": len(marker_ids),
                "marker_ids": " ".join(
                    str(marker_id)
                    for marker_id in marker_ids
                ),
                "image_topic": image_topics[camera],
                "camera_info_topic": info_topics[
                    camera
                ],
            }
        )

        contact_images.append(
            thumbnail(
                image,
                [
                    camera,
                    (
                        "markers: "
                        + (
                            ",".join(
                                str(marker_id)
                                for marker_id in marker_ids
                            )
                            if marker_ids
                            else "none"
                        )
                    ),
                    (
                        "sharpness: "
                        f"{result['sharpness']:.1f}"
                    ),
                ],
            )
        )

        print(
            f"[WRITE] {camera}: "
            f"{image_width}x{image_height}, "
            f"markers={marker_ids}, "
            f"sharpness={result['sharpness']:.1f}"
        )

    contact_sheet = np.vstack(
        [
            np.hstack(contact_images[:2]),
            np.hstack(contact_images[2:]),
        ]
    )

    contact_path = (
        metadata_dir
        / "STATIC_CAMERA_CONTACT_SHEET.jpg"
    )

    cv2.imwrite(
        str(contact_path),
        contact_sheet,
    )

    csv_path = (
        metadata_dir
        / "selected_static_frames.csv"
    )

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                metadata_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    report_lines = [
        "REAL STATIC CAMERA EXTRACTION",
        "=" * 72,
        "",
        f"Source MCAP: {mcap}",
        f"Selected common target timestamp: {selected_target}",
        "",
    ]

    for row in metadata_rows:
        report_lines.extend(
            [
                row["camera"],
                (
                    f"  resolution: "
                    f"{row['image_width']}x"
                    f"{row['image_height']}"
                ),
                (
                    f"  markers: "
                    f"{row['marker_ids'] or 'none'}"
                ),
                (
                    f"  marker count: "
                    f"{row['marker_count']}"
                ),
                (
                    f"  sharpness: "
                    f"{row['sharpness']:.3f}"
                ),
                (
                    f"  image/info offset: "
                    f"{row['image_info_offset_ms']:.3f} ms"
                ),
                "",
            ]
        )

    report_lines.extend(
        [
            f"Contact sheet: {contact_path}",
            f"Metadata CSV: {csv_path}",
            "",
            "[OK] static cameras extracted",
        ]
    )

    report_path = (
        metadata_dir
        / "STATIC_EXTRACTION_REPORT.txt"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n"
    )

    print()
    print(report_path.read_text())


if __name__ == "__main__":
    main()
