from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .simulation_routes import load_simulation_route


def _child(parent: ET.Element, name: str) -> ET.Element:
    node = parent.find(name)
    if node is None:
        node = ET.SubElement(parent, name)
    return node


def _moving_camera(
    root: ET.Element, model_name: str, sensor_name: str | None = None
) -> ET.Element:
    matches: list[ET.Element] = []
    for model in root.iter("model"):
        if model.get("name") != model_name:
            continue
        for sensor in model.iter("sensor"):
            if sensor_name is not None and sensor.get("name") != sensor_name:
                continue
            camera = sensor.find("camera")
            if camera is not None:
                matches.append(camera)
    if len(matches) != 1:
        raise RuntimeError(
            f"Moving camera model '{model_name}'"
            + (f" sensor '{sensor_name}'" if sensor_name else "")
            + f" resolved to {len(matches)} camera sensors; expected exactly one"
        )
    return matches[0]


def _static_camera_contract(
    root: ET.Element, moving_model_name: str
) -> list[tuple[str, str, bytes]]:
    """Return the immutable camera-model payload for every static sensor."""
    contract: list[tuple[str, str, bytes]] = []
    for model in root.iter("model"):
        model_name = str(model.get("name", ""))
        if model_name == moving_model_name:
            continue
        for sensor in model.iter("sensor"):
            camera = sensor.find("camera")
            if camera is None:
                continue
            contract.append(
                (
                    model_name,
                    str(sensor.get("name", "")),
                    ET.tostring(camera, encoding="utf-8"),
                )
            )
    return contract


def _scale_rgba(node: ET.Element, scale: float) -> None:
    if not node.text:
        return
    values = [float(value) for value in node.text.split()]
    if len(values) != 4:
        return
    values[:3] = [min(1.0, max(0.0, value * scale)) for value in values[:3]]
    node.text = " ".join(f"{value:.6f}" for value in values)


def _apply_custom_lighting(root: ET.Element, scale: float) -> None:
    world = next(root.iter("world"), None)
    if world is None:
        raise RuntimeError("Simulation SDF has no world")
    scene = world.find("scene")
    if scene is not None:
        ambient = scene.find("ambient")
        if ambient is not None:
            _scale_rgba(ambient, scale)
    for light in world.findall("light"):
        for name in ("diffuse", "specular"):
            node = light.find(name)
            if node is not None:
                _scale_rgba(node, scale)
    for model in world.findall("model"):
        if model.get("name") != "bus_ceiling_led_panels":
            continue
        for material in model.iter("material"):
            for name in ("ambient", "diffuse", "emissive"):
                node = material.find(name)
                if node is not None:
                    _scale_rgba(node, scale)


def compose_world(
    source: Path,
    destination: Path,
    *,
    model_name: str,
    sensor_name: str | None = None,
    width: int,
    height: int,
    hfov_deg: float,
    custom_lighting_scale: float | None,
    static_camera_update_rate_hz: float | None = None,
) -> Path:
    source_tree = ET.parse(source)
    source_root = source_tree.getroot()
    source_camera = _moving_camera(source_root, model_name, sensor_name)
    source_width = int(source_camera.findtext("image/width", "0"))
    source_height = int(source_camera.findtext("image/height", "0"))
    source_hfov = math.degrees(
        float(source_camera.findtext("horizontal_fov", "nan"))
    )
    needs_camera_change = (
        width != source_width
        or height != source_height
        or not math.isclose(hfov_deg, source_hfov, abs_tol=1e-9)
    )
    if (
        not needs_camera_change
        and custom_lighting_scale is None
        and static_camera_update_rate_hz is None
    ):
        return source.resolve()
    tree = source_tree
    root = source_root
    static_camera_contract = _static_camera_contract(root, model_name)
    camera = _moving_camera(root, model_name, sensor_name)
    _child(camera, "horizontal_fov").text = f"{math.radians(hfov_deg):.12f}"
    image = _child(camera, "image")
    _child(image, "width").text = str(width)
    _child(image, "height").text = str(height)
    if custom_lighting_scale is not None:
        _apply_custom_lighting(root, custom_lighting_scale)
    if static_camera_update_rate_hz is not None:
        for model in root.iter("model"):
            if model.get("name") == model_name:
                continue
            for sensor in model.iter("sensor"):
                if sensor.get("type") != "camera":
                    continue
                _child(sensor, "update_rate").text = (
                    f"{static_camera_update_rate_hz:g}"
                )
    if _static_camera_contract(root, model_name) != static_camera_contract:
        raise RuntimeError(
            "Simulation composition attempted to change a static camera model. "
            "Resolution, FOV and intrinsic camera parameters may only be changed "
            "for the configured moving camera."
        )
    ET.indent(tree, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination.resolve()


def _angle_interpolate(first: float, second: float, fraction: float) -> float:
    difference = (second - first + math.pi) % (2.0 * math.pi) - math.pi
    return first + fraction * difference


def compose_route(
    source: Path, destination: Path, *, target_frames: int | None
) -> Path:
    load_simulation_route(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise RuntimeError(f"Simulation route needs at least two frames: {source}")
    if target_frames is None or target_frames == len(frames):
        return source.resolve()
    if target_frames < 2:
        raise ValueError("target route frame count must be at least two")
    numeric_linear = ("x", "y", "z")
    numeric_angles = ("roll", "pitch", "yaw")
    generated = []
    for index in range(target_frames):
        position = (len(frames) - 1) * index / (target_frames - 1)
        lower = min(int(math.floor(position)), len(frames) - 2)
        fraction = position - lower
        first = frames[lower]
        second = frames[lower + 1]
        row = copy.deepcopy(first)
        row["frame"] = index
        for key in numeric_linear:
            row[key] = float(first[key]) + fraction * (
                float(second[key]) - float(first[key])
            )
        for key in numeric_angles:
            row[key] = _angle_interpolate(
                float(first[key]), float(second[key]), fraction
            )
        generated.append(row)
    result = dict(payload)
    result["frames"] = generated
    result["rigcal_resampling"] = {
        "source": str(source.resolve()),
        "source_frames": len(frames),
        "target_frames": target_frames,
        "rule": "linear position and shortest-angle interpolation over source index",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return destination.resolve()


def apply_motion_blur(
    image_paths: list[Path], *, kernel_size: int, angle_deg: float
) -> None:
    if kernel_size == 0:
        return
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Motion-blur simulation requires OpenCV and NumPy") from exc
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    radius = center
    angle = math.radians(angle_deg)
    dx = radius * math.cos(angle)
    dy = radius * math.sin(angle)
    start = (int(round(center - dx)), int(round(center - dy)))
    end = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, start, end, 1.0, 1)
    total = float(kernel.sum())
    if total <= 0:
        raise RuntimeError("Could not construct motion-blur kernel")
    kernel /= total
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read captured image for blur: {path}")
        blurred = cv2.filter2D(image, -1, kernel)
        if not cv2.imwrite(str(path), blurred):
            raise RuntimeError(f"Could not write blurred capture: {path}")
