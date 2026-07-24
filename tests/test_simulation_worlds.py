from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from camera_rig_calibration.simulation_worlds import (
    discover_world_manifests,
    install_world_manifest,
    load_world_manifest,
)


def _fixture_manifest(root: Path) -> Path:
    sdf = root / "world.sdf"
    sdf.write_text(
        """
<sdf version="1.8">
  <world name="fixture">
    <model name="static_left">
      <link name="link"><sensor name="static_sensor" type="camera">
        <camera><horizontal_fov>1.0</horizontal_fov>
          <image><width>640</width><height>480</height></image>
        </camera>
      </sensor></link>
    </model>
    <model name="moving">
      <link name="link"><sensor name="moving_sensor" type="camera">
        <camera><horizontal_fov>1.2</horizontal_fov>
          <image><width>1920</width><height>1080</height></image>
        </camera>
      </sensor></link>
    </model>
  </world>
</sdf>
""".strip()
    )
    route = root / "route.json"
    route.write_text(
        json.dumps({"frames": [{"pose": [0]}, {"pose": [1]}]})
    )
    manifest = root / "fixture.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "warehouse",
                "display_name": "Warehouse",
                "sdf": "world.sdf",
                "resource_paths": ["."],
                "static_cameras": [
                    {
                        "id": "left",
                        "model_name": "static_left",
                        "sensor_name": "static_sensor",
                        "image_topic": "/left/image",
                        "camera_info_topic": "/left/info",
                    }
                ],
                "moving_camera": {
                    "id": "wand",
                    "model_name": "moving",
                    "sensor_name": "moving_sensor",
                    "image_topic": "/wand/image",
                    "camera_info_topic": "/wand/info",
                },
                "routes": [
                    {
                        "id": "survey",
                        "path": "route.json",
                        "baseline": True,
                    }
                ],
                "capabilities": [
                    "route",
                    "density",
                    "resolution",
                    "fov",
                    "capture",
                ],
            }
        )
    )
    return manifest


def test_world_manifest_derives_moving_sensor_baseline(tmp_path: Path) -> None:
    manifest = load_world_manifest(_fixture_manifest(tmp_path))

    assert manifest.id == "warehouse"
    assert manifest.baseline["route_name"] == "survey"
    assert manifest.baseline["moving_width"] == 1920
    assert manifest.baseline["moving_height"] == 1080
    assert manifest.baseline["target_route_frames"] == 2
    assert manifest.static_cameras[0].id == "left"


def test_invalid_static_sensor_is_rejected(tmp_path: Path) -> None:
    path = _fixture_manifest(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["static_cameras"][0]["sensor_name"] = "missing"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="found 0"):
        load_world_manifest(path)


def test_imported_manifest_keeps_external_relative_assets_resolvable(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    source_root.mkdir()
    source = _fixture_manifest(source_root)
    repository = tmp_path / "repository"

    installed = install_world_manifest(repository, source)

    assert installed.manifest_path == (
        repository / "config/simulation_worlds/warehouse.yaml"
    ).resolve()
    assert installed.sdf == (source_root / "world.sdf").resolve()
    assert installed.baseline_route.path == (
        source_root / "route.json"
    ).resolve()


def test_builtin_bus_world_is_discovered_without_python_registration() -> None:
    repository = Path(__file__).resolve().parents[1]

    worlds = discover_world_manifests(repository)

    bus = next(item for item in worlds if item.id == "bus")
    assert bus.baseline["route_name"] == "route2"
    assert bus.baseline["moving_width"] == 1280
    assert "lighting" in bus.capabilities
    assert "baseline" in bus.lighting_profiles
