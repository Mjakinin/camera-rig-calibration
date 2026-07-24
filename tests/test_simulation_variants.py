from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from camera_rig_calibration.input.simulation_variants import (
    compose_route,
    compose_world,
)


def _write_world(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<sdf version="1.8">
  <world name="fixture">
    <scene><ambient>0.2 0.3 0.4 1</ambient></scene>
    <light name="ceiling"><diffuse>0.4 0.4 0.4 1</diffuse></light>
    <model name="static_camera">
      <link name="link"><sensor name="static_sensor" type="camera">
        <update_rate>15</update_rate><camera>
          <horizontal_fov>0.91</horizontal_fov>
          <image><width>848</width><height>480</height></image>
          <distortion><k1>0.01</k1><k2>-0.02</k2></distortion>
        </camera>
      </sensor></link>
    </model>
    <model name="moving_calib_camera">
      <link name="link"><sensor name="sensor" type="camera"><camera>
        <horizontal_fov>1.2060225131</horizontal_fov>
        <image><width>1280</width><height>720</height></image>
      </camera></sensor></link>
    </model>
  </world>
</sdf>
""",
        encoding="utf-8",
    )


def test_baseline_world_is_reused_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "source.sdf"
    _write_world(source)
    result = compose_world(
        source,
        tmp_path / "generated.sdf",
        model_name="moving_calib_camera",
        width=1280,
        height=720,
        hfov_deg=69.1,
        custom_lighting_scale=None,
    )
    assert result == source.resolve()
    assert not (tmp_path / "generated.sdf").exists()


def test_world_parameters_can_be_combined_in_one_generated_variant(tmp_path: Path) -> None:
    source = tmp_path / "source.sdf"
    destination = tmp_path / "generated.sdf"
    _write_world(source)
    source_root = ET.parse(source).getroot()
    source_static_camera = next(
        camera
        for model in source_root.iter("model")
        if model.get("name") == "static_camera"
        for camera in model.iter("camera")
    )
    expected_static_camera = ET.canonicalize(
        ET.tostring(source_static_camera, encoding="unicode"),
        strip_text=True,
    )
    result = compose_world(
        source,
        destination,
        model_name="moving_calib_camera",
        width=640,
        height=360,
        hfov_deg=100.0,
        custom_lighting_scale=0.5,
        static_camera_update_rate_hz=1.0,
    )

    root = ET.parse(result).getroot()
    camera = next(
        camera
        for model in root.iter("model")
        if model.get("name") == "moving_calib_camera"
        for camera in model.iter("camera")
    )
    assert camera.findtext("image/width") == "640"
    assert camera.findtext("image/height") == "360"
    assert math.isclose(float(camera.findtext("horizontal_fov")), math.radians(100.0))
    assert root.findtext("world/scene/ambient") == "0.100000 0.150000 0.200000 1.000000"
    static_sensor = next(
        sensor
        for model in root.iter("model")
        if model.get("name") == "static_camera"
        for sensor in model.iter("sensor")
    )
    assert static_sensor.findtext("update_rate") == "1"
    actual_static_camera = static_sensor.find("camera")
    assert actual_static_camera is not None
    assert (
        ET.canonicalize(
            ET.tostring(actual_static_camera, encoding="unicode"),
            strip_text=True,
        )
        == expected_static_camera
    )


def test_route_is_resampled_to_an_exact_user_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "route.json"
    destination = tmp_path / "route_5.json"
    source.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame": 0, "x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 3.0},
                    {"frame": 1, "x": 4, "y": 2, "z": 1, "roll": 0, "pitch": 0, "yaw": -3.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = compose_route(source, destination, target_frames=5)
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert len(payload["frames"]) == 5
    assert payload["frames"][0]["x"] == 0.0
    assert payload["frames"][-1]["x"] == 4.0
    assert payload["frames"][2]["x"] == 2.0
    assert abs(payload["frames"][2]["yaw"]) > 3.0
    assert payload["rigcal_resampling"]["target_frames"] == 5
