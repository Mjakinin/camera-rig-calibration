from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class McapTopic:
    name: str
    message_type: str
    message_count: int | None = None

    @property
    def is_image(self) -> bool:
        return self.message_type in {
            "sensor_msgs/msg/Image",
            "sensor_msgs/msg/CompressedImage",
        }

    @property
    def is_camera_info(self) -> bool:
        return self.message_type == "sensor_msgs/msg/CameraInfo"


@dataclass(frozen=True)
class RosbagSource:
    """The URI and storage plugin ROS 2 needs for a selected bag path."""

    selected_path: Path
    uri: Path
    storage_id: str
    metadata_path: Path | None = None


def _read_metadata(path: Path) -> dict[str, object] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    information = payload.get("rosbag2_bagfile_information", payload)
    return information if isinstance(information, dict) else None


def resolve_rosbag_source(path: Path) -> RosbagSource:
    """Resolve a bag selection and its explicit ROS 2 storage plugin.

    ROS 2 recordings commonly consist of ``metadata.yaml`` plus one or more
    ``.mcap``/``.db3`` chunks. Passing a chunk without a storage id makes
    ``ros2 bag`` attempt plugin auto-detection, which is unreliable on Humble.
    An explicitly selected chunk remains the URI: this also lets Humble read
    bags whose newer metadata schema it cannot deserialize itself.
    """

    selected = path.expanduser().resolve()
    if not selected.exists():
        raise FileNotFoundError(f"MCAP or ROS bag not found: {selected}")

    metadata_path = (
        selected / "metadata.yaml"
        if selected.is_dir()
        else selected.parent / "metadata.yaml"
    )
    metadata = _read_metadata(metadata_path) if metadata_path.is_file() else None
    storage_id = ""
    if metadata is not None:
        storage_id = str(metadata.get("storage_identifier") or "").strip()

    if not storage_id:
        suffix = selected.suffix.lower()
        storage_id = {
            ".mcap": "mcap",
            ".db3": "sqlite3",
        }.get(suffix, "")

    return RosbagSource(
        selected_path=selected,
        uri=selected,
        storage_id=storage_id,
        metadata_path=metadata_path if metadata is not None else None,
    )


def _topics_from_metadata(source: RosbagSource) -> list[McapTopic]:
    if source.metadata_path is None:
        return []
    metadata = _read_metadata(source.metadata_path)
    if metadata is None:
        return []
    raw_topics = metadata.get("topics_with_message_count", [])
    if not isinstance(raw_topics, list):
        return []

    topics: list[McapTopic] = []
    for raw_topic in raw_topics:
        if not isinstance(raw_topic, dict):
            continue
        topic_metadata = raw_topic.get("topic_metadata", {})
        if not isinstance(topic_metadata, dict):
            continue
        name = topic_metadata.get("name")
        message_type = topic_metadata.get("type")
        if not name or not message_type:
            continue
        raw_count = raw_topic.get("message_count")
        try:
            count = int(raw_count) if raw_count is not None else None
        except (TypeError, ValueError):
            count = None
        topics.append(McapTopic(str(name), str(message_type), count))
    return topics


def list_mcap_topics(path: Path) -> list[McapTopic]:
    """Read ROS bag topics without importing ROS Python modules into the CLI."""

    source = resolve_rosbag_source(path)
    metadata_topics = _topics_from_metadata(source)
    if metadata_topics:
        return metadata_topics

    command = ["ros2", "bag", "info"]
    if source.storage_id:
        command.extend(["--storage", source.storage_id])
    command.append(str(source.uri))
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not inspect ROS bag topics because the 'ros2' command is not "
            "available. Run rigcal inside the ros2humble container."
        ) from exc
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        plugin_hint = (
            f" Install ros-humble-rosbag2-storage-{source.storage_id} in the "
            "container."
            if source.storage_id
            else ""
        )
        raise RuntimeError(
            f"Could not inspect ROS bag topics at {source.uri} using storage "
            f"'{source.storage_id or 'auto'}': {detail}.{plugin_hint}"
        )

    topics: list[McapTopic] = []
    pattern = re.compile(
        r"Topic:\s*(?P<name>\S+)\s*\|\s*Type:\s*(?P<type>\S+)"
        r"(?:\s*\|\s*Count:\s*(?P<count>\d+))?"
    )
    for line in process.stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        count = match.group("count")
        topics.append(
            McapTopic(
                match.group("name"),
                match.group("type"),
                int(count) if count is not None else None,
            )
        )
    if not topics:
        raise RuntimeError(
            "ROS bag inspection returned no topics. Ensure ROS 2 is sourced and "
            "the required storage plugin is installed."
        )
    return topics
