from __future__ import annotations

from pathlib import Path

import yaml

from camera_rig_calibration.input.topics import (
    list_mcap_topics,
    resolve_rosbag_source,
)


def _write_metadata(directory: Path) -> Path:
    directory.mkdir(parents=True)
    chunk = directory / "recording_0.mcap"
    chunk.write_bytes(b"fixture")
    (directory / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "rosbag2_bagfile_information": {
                    "storage_identifier": "mcap",
                    "relative_file_paths": [chunk.name],
                    "topics_with_message_count": [
                        {
                            "topic_metadata": {
                                "name": "/front/image/compressed",
                                "type": "sensor_msgs/msg/CompressedImage",
                            },
                            "message_count": 42,
                        },
                        {
                            "topic_metadata": {
                                "name": "/front/camera_info",
                                "type": "sensor_msgs/msg/CameraInfo",
                            },
                            "message_count": 3,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return chunk


def test_rosbag_chunk_keeps_explicit_uri_and_resolves_storage_id_from_metadata(
    tmp_path: Path,
) -> None:
    chunk = _write_metadata(tmp_path / "recording")

    source = resolve_rosbag_source(chunk)

    assert source.selected_path == chunk
    assert source.uri == chunk
    assert source.storage_id == "mcap"
    assert source.metadata_path == chunk.parent / "metadata.yaml"


def test_topics_are_read_from_metadata_without_opening_storage_plugin(
    tmp_path: Path,
) -> None:
    chunk = _write_metadata(tmp_path / "recording")

    topics = list_mcap_topics(chunk)

    assert [(topic.name, topic.message_count) for topic in topics] == [
        ("/front/image/compressed", 42),
        ("/front/camera_info", 3),
    ]
    assert topics[0].is_image
    assert topics[1].is_camera_info
