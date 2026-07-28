from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..methods.common.aruco_utils import (
    effective_detector_config,
    make_aruco_detector,
)
from .topics import resolve_rosbag_source


@dataclass
class Candidate:
    score: tuple[int, float, float]
    image: Any
    timestamp_ns: int
    marker_ids: list[int]


def _raw_image(message: Any, cv2: Any, np: Any) -> Any:
    encoding = str(message.encoding).lower()
    channels = {
        "mono8": 1,
        "8uc1": 1,
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
    }.get(encoding)
    if channels is None:
        raise RuntimeError(f"Unsupported ROS Image encoding: {message.encoding}")
    expected = int(message.height) * int(message.step)
    buffer = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if buffer.size < expected:
        raise RuntimeError("ROS Image data is shorter than height * step")
    rows = buffer[:expected].reshape(int(message.height), int(message.step))
    image = rows[:, : int(message.width) * channels]
    image = image.reshape(int(message.height), int(message.width), channels)
    if channels == 1:
        return cv2.cvtColor(image.reshape(int(message.height), int(message.width)), cv2.COLOR_GRAY2BGR)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _decode(message: Any, message_type: str, cv2: Any, np: Any) -> Any:
    if message_type == "sensor_msgs/msg/CompressedImage":
        image = cv2.imdecode(np.frombuffer(bytes(message.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not decode CompressedImage")
        return image
    if message_type == "sensor_msgs/msg/Image":
        return _raw_image(message, cv2, np)
    raise RuntimeError(f"Topic is not a supported image type: {message_type}")


def _camera_info(message: Any, camera_id: str) -> dict[str, Any]:
    return {
        "camera_name": camera_id,
        "width": int(message.width),
        "height": int(message.height),
        "image_width": int(message.width),
        "image_height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "K": [float(value) for value in message.k],
        "k": [float(value) for value in message.k],
        "D": [float(value) for value in message.d],
        "d": [float(value) for value in message.d],
        "R": [float(value) for value in message.r],
        "r": [float(value) for value in message.r],
        "P": [float(value) for value in message.p],
        "p": [float(value) for value in message.p],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract generic static-camera inputs from an MCAP or ROS bag."
    )
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mapping", required=True)
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "MCAP extraction requires a sourced ROS 2 environment, rosbag2_py, "
            "sensor_msgs and OpenCV. Run rigcal inside the ros2humble container."
        ) from exc

    source = resolve_rosbag_source(Path(args.mcap))
    dataset = Path(args.dataset).resolve()
    mapping_path = Path(args.mapping).resolve()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    cameras = mapping.get("cameras", [])
    moving_camera = mapping.get("moving_camera")
    if not cameras and not moving_camera:
        raise RuntimeError("MCAP mapping contains no cameras")
    if any(not camera.get("image_topic") for camera in cameras):
        missing = [camera.get("id") for camera in cameras if not camera.get("image_topic")]
        raise RuntimeError(f"MCAP image topics are missing for: {missing}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(source.uri),
            storage_id=source.storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="", output_serialization_format=""
        ),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    requested = {
        value
        for camera in cameras
        for value in (camera.get("image_topic"), camera.get("camera_info_topic"))
        if value
    }
    if moving_camera:
        requested.update(
            value
            for value in (
                moving_camera.get("image_topic"),
                moving_camera.get("camera_info_topic"),
            )
            if value
        )
    absent = sorted(requested - topic_types.keys())
    if absent:
        raise RuntimeError(f"Configured MCAP topics do not exist: {absent}")

    camera_by_image_topic = {camera["image_topic"]: camera for camera in cameras}
    camera_by_info_topic = {
        camera["camera_info_topic"]: camera
        for camera in cameras
        if camera.get("camera_info_topic")
    }
    moving_image_topic = moving_camera.get("image_topic") if moving_camera else None
    moving_info_topic = (
        moving_camera.get("camera_info_topic") if moving_camera else None
    )
    message_classes = {
        topic: get_message(topic_types[topic]) for topic in requested
    }
    dictionary_name = str(mapping.get("marker_dictionary", "DICT_4X4_50"))
    # Raw acquisition remains mode-independent. The canonical shared detector
    # is used in baseline mode only to choose a representative MCAP frame;
    # the queue-selected mode is applied later to immutable raw images.
    selection_detection_mode = "baseline"
    detect_markers = make_aruco_detector(
        dictionary_name, selection_detection_mode
    )

    best: dict[str, Candidate] = {}
    infos: dict[str, dict[str, Any]] = {}
    counts = {camera["id"]: 0 for camera in cameras}
    preview_root = dataset / "preview" / "static_candidates"
    save_all = bool(mapping.get("save_all_candidates", False))
    moving_root = dataset / "raw_images" / "moving"
    moving_root.mkdir(parents=True, exist_ok=True)
    moving_count = 0
    moving_info: dict[str, Any] | None = None
    last_moving_timestamp_ns: int | None = None
    moving_hz = float(mapping.get("moving_sampling_hz") or 0.0)
    moving_period_ns = int(1_000_000_000 / moving_hz) if moving_hz > 0 else 0

    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic not in requested:
            continue
        message = deserialize_message(serialized, message_classes[topic])
        if topic == moving_info_topic and moving_camera:
            moving_info = _camera_info(message, moving_camera["id"])
            continue
        if topic == moving_image_topic and moving_camera:
            if (
                last_moving_timestamp_ns is None
                or moving_period_ns == 0
                or int(timestamp_ns) - last_moving_timestamp_ns >= moving_period_ns
            ):
                image = _decode(message, topic_types[topic], cv2, np)
                destination = moving_root / f"frame_{moving_count:06d}.png"
                if not cv2.imwrite(str(destination), image):
                    raise RuntimeError(f"Could not write moving image: {destination}")
                moving_count += 1
                last_moving_timestamp_ns = int(timestamp_ns)
            continue
        if topic in camera_by_info_topic:
            camera = camera_by_info_topic[topic]
            infos.setdefault(camera["id"], _camera_info(message, camera["id"]))
            continue
        camera = camera_by_image_topic[topic]
        camera_id = camera["id"]
        image = _decode(message, topic_types[topic], cv2, np)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray)
        marker_ids = [] if ids is None else [int(value) for value in ids.reshape(-1)]
        total_area = sum(
            abs(float(cv2.contourArea(np.asarray(corner).reshape(4, 2).astype(np.float32))))
            for corner in corners
        )
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score = (len(marker_ids), total_area, sharpness)
        candidate = Candidate(score, image, int(timestamp_ns), sorted(marker_ids))
        if camera_id not in best or candidate.score > best[camera_id].score:
            best[camera_id] = candidate
        if save_all:
            directory = preview_root / camera_id
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / f"frame_{counts[camera_id]:06d}.jpg"), image)
        counts[camera_id] += 1

    static_root = dataset / "raw_images" / "static"
    info_root = dataset / "raw_images" / "camera_info"
    metadata_root = dataset / "metadata" / "static_extraction"
    static_root.mkdir(parents=True, exist_ok=True)
    info_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    if moving_camera:
        if moving_count == 0:
            raise RuntimeError(
                f"No moving images found on {moving_camera.get('image_topic')}"
            )
        if moving_info is not None:
            (info_root / f"{moving_camera['id']}.json").write_text(
                json.dumps(moving_info, indent=2) + "\n", encoding="utf-8"
            )
        elif moving_camera.get("camera_info_topic"):
            raise RuntimeError(
                f"No moving CameraInfo found on {moving_camera['camera_info_topic']}"
            )
    rows = []
    for camera in cameras:
        camera_id = camera["id"]
        if camera_id not in best:
            raise RuntimeError(
                f"No image messages found for '{camera_id}' on {camera['image_topic']}"
            )
        selected = best[camera_id]
        destination = static_root / f"{camera_id}.png"
        if not cv2.imwrite(str(destination), selected.image):
            raise RuntimeError(f"Could not write selected image: {destination}")
        if camera_id in infos:
            (info_root / f"{camera_id}.json").write_text(
                json.dumps(infos[camera_id], indent=2) + "\n", encoding="utf-8"
            )
        rows.append(
            {
                "camera_id": camera_id,
                "image_topic": camera["image_topic"],
                "camera_info_topic": camera.get("camera_info_topic") or "",
                "messages_seen": counts[camera_id],
                "selected_timestamp_ns": selected.timestamp_ns,
                "detected_marker_ids": ";".join(map(str, selected.marker_ids)),
                "marker_count": selected.score[0],
                "marker_area_px2": selected.score[1],
                "sharpness": selected.score[2],
                "output_image": str(destination),
            }
        )
    if rows:
        fields = list(rows[0])
        with (metadata_root / "selected_static_frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (metadata_root / "mcap_topic_mapping.json").write_text(
        json.dumps(
            {
                "source": str(source.selected_path),
                "resolved_uri": str(source.uri),
                "storage_id": source.storage_id,
                "topics": topic_types,
                "cameras": cameras,
                "moving_camera": moving_camera,
                "moving_frames": moving_count,
                "moving_sampling_hz": moving_hz or None,
                "static_frame_selection_detector": effective_detector_config(
                    selection_detection_mode,
                    dictionary_name,
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[OK] extracted {len(rows)} static cameras and {moving_count} moving "
        f"frames from {source.selected_path}"
    )


if __name__ == "__main__":
    main()
