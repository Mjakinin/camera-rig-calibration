from __future__ import annotations

from pathlib import Path

import pytest

from camera_rig_calibration.assets import MaterializedAsset
from camera_rig_calibration.input.simulation import (
    _intrinsics_capture_partition,
    _validate_known_world_assets,
    capture_frame_diversity,
)


def _write_bus_world(path: Path) -> None:
    path.write_text(
        """
<sdf version="1.8">
  <world name="fixture">
    <include><uri>model://beintelli_bus</uri></include>
  </world>
</sdf>
""".strip(),
        encoding="utf-8",
    )


def test_bus_world_uses_materialized_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = tmp_path / "world.sdf"
    _write_bus_world(world)
    mesh = tmp_path / "beintelli_erklarbus.obj"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(
        b"# fixture\nv 0 0 0\nf 1 1 1\n" + (b"# padding\n" * 120_000)
    )
    monkeypatch.setattr(
        "camera_rig_calibration.input.simulation.ensure_bus_mesh",
        lambda _repository: MaterializedAsset(
            path=mesh,
            created=True,
            sha256="fixture",
            size_bytes=mesh.stat().st_size,
        ),
    )

    _validate_known_world_assets(tmp_path, world)


def test_non_bus_custom_world_does_not_require_bus_asset(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    world.write_text(
        '<sdf version="1.8"><world name="fixture"/></sdf>',
        encoding="utf-8",
    )

    _validate_known_world_assets(tmp_path, world)


def test_stale_simulation_frames_are_rejected(tmp_path: Path) -> None:
    images = []
    for index in range(10):
        path = tmp_path / f"frame_{index:04d}.png"
        path.write_bytes(b"same rendered frame")
        images.append(path)

    with pytest.raises(RuntimeError, match=r"only 1/10 images are unique"):
        capture_frame_diversity(images)


def test_diverse_simulation_frames_are_accepted(tmp_path: Path) -> None:
    images = []
    for index in range(10):
        path = tmp_path / f"frame_{index:04d}.png"
        path.write_bytes(f"rendered frame {index}".encode())
        images.append(path)

    result = capture_frame_diversity(images)

    assert result["frames"] == 10
    assert result["unique_frames"] == 10


def test_explicit_static_and_moving_intrinsics_are_preserved() -> None:
    generated, preserved = _intrinsics_capture_partition(
        {
            "static_cameras": [
                {
                    "id": "static_generated",
                    "intrinsics_source": "gazebo_camera_info",
                },
                {"id": "static_loaded", "intrinsics_source": "provided"},
            ],
            "moving_camera": {
                "id": "moving_loaded",
                "intrinsics_source": "provided",
            },
        }
    )

    assert generated == {"static_generated"}
    assert preserved == {"static_loaded", "moving_loaded"}


def test_legacy_simulation_mapping_uses_gazebo_camera_info() -> None:
    generated, preserved = _intrinsics_capture_partition(
        {
            "static_cameras": [{"id": "static_camera"}],
            "moving_camera": {"id": "moving_camera"},
        }
    )

    assert generated == {"static_camera", "moving_camera"}
    assert preserved == set()
