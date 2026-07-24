from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..assets import ensure_bus_mesh
from .simulation_variants import apply_motion_blur, compose_route, compose_world


_GAZEBO_ASSET_ERRORS = (
    "Cannot load mesh with zero sub-meshes",
    "Failed to get Ogre item for",
    "Unable to find uri",
    "Unable to find file",
)


def _world_name(path: Path) -> str:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RuntimeError(f"Could not read simulation world {path}: {exc}") from exc
    world = root.find("world") if root.tag != "world" else root
    if world is None or not world.get("name"):
        raise RuntimeError(f"Simulation world has no named <world> element: {path}")
    return str(world.get("name"))


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required simulation command '{name}' is not available. "
            "Run rigcal inside the ros2humble container."
        )


def _validate_known_world_assets(repository: Path, world: Path) -> None:
    """Materialize and validate the known bus asset before starting Gazebo."""
    try:
        world_text = world.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Could not inspect simulation world {world}: {exc}") from exc
    if "model://beintelli_bus" not in world_text:
        return

    mesh = ensure_bus_mesh(repository).path
    with mesh.open("rb") as stream:
        sample = stream.read(1_000_000)
    if mesh.stat().st_size < 1_000_000 or b"\nv " not in sample or b"\nf " not in sample:
        raise RuntimeError(
            f"The BeIntelli bus OBJ does not contain usable mesh geometry: {mesh}"
        )


def capture_frame_diversity(
    images: list[Path],
    *,
    minimum_unique_fraction: float = 0.9,
) -> dict[str, float | int]:
    """Reject stale-frame captures before they enter a calibration dataset."""
    if not images:
        raise RuntimeError("Simulation route capture produced no moving frames")
    hashes: set[str] = set()
    for image in images:
        digest = hashlib.sha256()
        with image.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes.add(digest.hexdigest())
    total = len(images)
    unique = len(hashes)
    required = max(2, math.ceil(total * minimum_unique_fraction))
    if unique < required:
        raise RuntimeError(
            "Simulation capture contains stale repeated frames: "
            f"only {unique}/{total} images are unique, but at least {required} "
            "are required. The capture was not published and no AP method was "
            "started. Increase the frame timeout if Gazebo renders slowly."
        )
    return {
        "frames": total,
        "unique_frames": unique,
        "unique_fraction": unique / total,
        "minimum_unique_fraction": minimum_unique_fraction,
    }


def _raise_for_gazebo_asset_errors(log_path: Path) -> None:
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    matches = [message for message in _GAZEBO_ASSET_ERRORS if message in log_text]
    if matches:
        raise RuntimeError(
            "Gazebo reported an invalid or missing render asset "
            f"({'; '.join(matches)}). See {log_path}. No frames will be accepted."
        )


def _listed(command: list[str]) -> set[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _wait_for_service(world_name: str, timeout_seconds: float) -> None:
    expected = f"/world/{world_name}/set_pose"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if expected in _listed(["ign", "service", "-l"]):
            print(f"[OK] Gazebo world ready: {world_name}", flush=True)
            return
        time.sleep(1.0)
    raise RuntimeError(f"Gazebo service did not appear within {timeout_seconds:g}s: {expected}")


def _wait_for_topics(topics: set[str], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    missing = set(topics)
    while time.monotonic() < deadline:
        missing = topics - _listed(["ros2", "topic", "list"])
        if not missing:
            print(f"[OK] ROS bridge ready: {len(topics)} topics", flush=True)
            return
        time.sleep(1.0)
    raise RuntimeError(
        "ROS topics did not appear within "
        f"{timeout_seconds:g}s: {', '.join(sorted(missing))}"
    )


def _camera_info_payload(message: Any, camera_id: str) -> dict[str, Any]:
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


def _intrinsics_capture_partition(
    mapping: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """Split cameras into Gazebo-derived and explicitly provided intrinsics."""
    generated: set[str] = set()
    preserved: set[str] = set()
    cameras = [*mapping["static_cameras"], mapping["moving_camera"]]
    for camera in cameras:
        camera_id = str(camera["id"])
        source = str(camera.get("intrinsics_source", "gazebo_camera_info"))
        if source == "provided":
            preserved.add(camera_id)
        elif source == "gazebo_camera_info":
            generated.add(camera_id)
        else:
            raise ValueError(
                f"Unknown intrinsics source '{source}' for camera '{camera_id}'"
            )
    return generated, preserved


def _capture_static_and_intrinsics(
    mapping: dict[str, Any], dataset: Path, timeout_seconds: float
) -> dict[str, Any]:
    try:
        import cv2
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
        raise RuntimeError(
            "Simulation capture requires rclpy, sensor_msgs, cv_bridge and OpenCV "
            "inside the ros2humble container."
        ) from exc

    static_cameras = mapping["static_cameras"]
    moving_camera = mapping["moving_camera"]
    generated_intrinsics, preserved_intrinsics = _intrinsics_capture_partition(mapping)
    info_root = dataset / "raw_images" / "camera_info"
    missing_preserved = sorted(
        camera_id
        for camera_id in preserved_intrinsics
        if not (info_root / f"{camera_id}.json").is_file()
    )
    if missing_preserved:
        raise RuntimeError(
            "Explicitly provided intrinsics were not materialized before simulation "
            f"capture: {', '.join(missing_preserved)}"
        )

    class SnapshotNode(Node):
        def __init__(self) -> None:
            super().__init__("rigcal_simulation_snapshot")
            self.bridge = CvBridge()
            self.images: dict[str, Any] = {}
            self.infos: dict[str, Any] = {}
            for camera in static_cameras:
                camera_id = camera["id"]
                self.create_subscription(
                    Image,
                    camera["image_topic"],
                    lambda message, value=camera_id: self.images.setdefault(
                        value, message
                    ),
                    qos_profile_sensor_data,
                )
                if camera_id in generated_intrinsics:
                    self.create_subscription(
                        CameraInfo,
                        camera["camera_info_topic"],
                        lambda message, value=camera_id: self.infos.setdefault(
                            value, message
                        ),
                        qos_profile_sensor_data,
                    )
            if moving_camera["id"] in generated_intrinsics:
                self.create_subscription(
                    CameraInfo,
                    moving_camera["camera_info_topic"],
                    lambda message: self.infos.setdefault(
                        moving_camera["id"], message
                    ),
                    qos_profile_sensor_data,
                )

    rclpy.init(args=None)
    node = SnapshotNode()
    expected_images = {camera["id"] for camera in static_cameras}
    expected_infos = generated_intrinsics
    try:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if expected_images <= node.images.keys() and expected_infos <= node.infos.keys():
                break
        missing_images = sorted(expected_images - node.images.keys())
        missing_infos = sorted(expected_infos - node.infos.keys())
        if missing_images or missing_infos:
            raise RuntimeError(
                f"Simulation snapshot timeout; images={missing_images}, "
                f"CameraInfo={missing_infos}"
            )

        static_root = dataset / "raw_images" / "static"
        static_root.mkdir(parents=True, exist_ok=True)
        info_root.mkdir(parents=True, exist_ok=True)
        for camera in static_cameras:
            camera_id = camera["id"]
            image = node.bridge.imgmsg_to_cv2(
                node.images[camera_id], desired_encoding="bgr8"
            )
            destination = static_root / f"{camera_id}.png"
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"Could not write simulation image: {destination}")
        for camera_id, message in node.infos.items():
            (info_root / f"{camera_id}.json").write_text(
                json.dumps(_camera_info_payload(message, camera_id), indent=2) + "\n",
                encoding="utf-8",
            )
        all_camera_ids = expected_images | {moving_camera["id"]}
        missing_files = sorted(
            camera_id
            for camera_id in all_camera_ids
            if not (info_root / f"{camera_id}.json").is_file()
        )
        if missing_files:
            raise RuntimeError(
                "Simulation capture has no usable intrinsics for: "
                + ", ".join(missing_files)
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(
        f"[OK] captured {len(static_cameras)} static images and "
        f"generated {len(expected_infos)} CameraInfo records; "
        f"preserved {len(preserved_intrinsics)} explicitly provided intrinsic files",
        flush=True,
    )
    return {
        "static_images": len(static_cameras),
        "generated_from_gazebo_camera_info": sorted(generated_intrinsics),
        "preserved_explicit_intrinsics": sorted(preserved_intrinsics),
    }


def _signal_process_group(
    process: subprocess.Popen[str], signal_number: signal.Signals
) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(signal_number)
        except ProcessLookupError:
            pass


def _stop(process: subprocess.Popen[str] | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            process.wait(timeout=5)
    print(f"[OK] stopped {label}", flush=True)


def _bridge_arguments(mapping: dict[str, Any]) -> tuple[list[str], set[str]]:
    image_suffix = "@sensor_msgs/msg/Image@gz.msgs.Image"
    info_suffix = "@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo"
    arguments: list[str] = []
    topics: set[str] = set()
    cameras = [*mapping["static_cameras"], mapping["moving_camera"]]
    for camera in cameras:
        image_topic = str(camera["image_topic"])
        info_topic = str(camera["camera_info_topic"])
        arguments.extend([image_topic + image_suffix, info_topic + info_suffix])
        topics.update((image_topic, info_topic))
    return arguments, topics


def capture(repository: Path, dataset: Path, mapping_path: Path) -> None:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_world = Path(mapping["world"]).resolve()
    source_route = Path(mapping["route"]).resolve()
    if not source_world.is_file() or not source_route.is_file():
        raise FileNotFoundError(
            f"Simulation world/route not found: {source_world}, {source_route}"
        )
    _validate_known_world_assets(repository, source_world)
    generated = dataset / "metadata" / "simulation" / "generated"
    world = compose_world(
        source_world,
        generated / "composed_world.sdf",
        model_name=str(mapping["moving_model_name"]),
        sensor_name=(
            str(mapping["moving_sensor_name"])
            if mapping.get("moving_sensor_name")
            else None
        ),
        width=int(mapping.get("moving_width", 1280)),
        height=int(mapping.get("moving_height", 720)),
        hfov_deg=float(mapping.get("moving_hfov_deg", 69.1)),
        custom_lighting_scale=(
            float(mapping.get("lighting_scale", 1.0))
            if mapping.get("lighting") == "custom"
            else None
        ),
        # Registered-world validation and composition preserve every static
        # camera sensor exactly. Capture records one static snapshot, but does
        # not silently rewrite its SDF update rate or intrinsics.
        static_camera_update_rate_hz=None,
    )
    route = compose_route(
        source_route,
        generated / "composed_route.json",
        target_frames=mapping.get("target_route_frames"),
    )
    _require_command("ign")
    _require_command("ros2")

    world_name = _world_name(world)
    exact_service = f"/world/{world_name}/set_pose"
    if exact_service in _listed(["ign", "service", "-l"]):
        raise RuntimeError(
            f"A Gazebo world named '{world_name}' is already active. Stop it before "
            "starting a reproducible rigcal capture."
        )

    metadata = dataset / "metadata" / "simulation"
    metadata.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    resource_paths = list(
        dict.fromkeys(
            [
                *(
                    str(Path(path).resolve())
                    for path in mapping.get("resource_paths", [])
                ),
                str(repository / "src/calib_lab/bus_real_data/models"),
                str(world.parent),
            ]
        )
    )
    for name in ("IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_RESOURCE_PATH"):
        current = environment.get(name)
        environment[name] = os.pathsep.join(resource_paths + ([current] if current else []))
    model_path = str(repository / "src/calib_lab/bus_real_data/models")
    current_model_path = environment.get("GAZEBO_MODEL_PATH")
    environment["GAZEBO_MODEL_PATH"] = os.pathsep.join(
        [model_path] + ([current_model_path] if current_model_path else [])
    )

    gazebo: subprocess.Popen[str] | None = None
    bridge: subprocess.Popen[str] | None = None
    route_process: subprocess.Popen[str] | None = None
    bridge_arguments, topics = _bridge_arguments(mapping)
    print(f"[INFO] simulation preset: {mapping.get('preset', 'custom')}", flush=True)
    if mapping.get("capture_id"):
        print(f"[INFO] capture ID: {mapping['capture_id']}", flush=True)
    print(f"[INFO] world: {world}", flush=True)
    print(f"[INFO] route: {route}", flush=True)
    gazebo_log_path = metadata / "gazebo.log"
    gazebo_command = ["ign", "gazebo", "-r", "-s", str(world)]
    print("[INFO] Gazebo mode: headless server", flush=True)
    with gazebo_log_path.open("a", encoding="utf-8") as gazebo_log, (
        metadata / "bridge.log"
    ).open("a", encoding="utf-8") as bridge_log:
        try:
            gazebo = subprocess.Popen(
                gazebo_command,
                cwd=repository,
                env=environment,
                stdout=gazebo_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            startup_timeout = float(mapping["startup_timeout_seconds"])
            _wait_for_service(world_name, startup_timeout)
            time.sleep(0.5)
            _raise_for_gazebo_asset_errors(gazebo_log_path)
            bridge = subprocess.Popen(
                [
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    *bridge_arguments,
                ],
                cwd=repository,
                env=environment,
                stdout=bridge_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            _wait_for_topics(topics, startup_timeout)
            intrinsic_capture = _capture_static_and_intrinsics(
                mapping, dataset, startup_timeout
            )

            moving_capture = metadata / "moving_route_capture"
            command = [
                sys.executable,
                str(
                    repository
                    / "run/bus_real_data/_shared/tools/capture/04_capture_moving_camera_route.py"
                ),
                "--route",
                str(route),
                "--world",
                world_name,
                "--name",
                str(mapping["moving_model_name"]),
                "--topic",
                str(mapping["moving_camera"]["image_topic"]),
                "--out",
                str(moving_capture),
                "--settle",
                str(mapping["settle_seconds"]),
                "--post-pose-skip",
                str(mapping["post_pose_skip"]),
                "--timeout",
                str(mapping["frame_timeout_seconds"]),
                "--clean",
            ]
            print("[INFO] capturing moving-camera route", flush=True)
            route_process = subprocess.Popen(
                command,
                cwd=repository,
                env=environment,
                text=True,
                start_new_session=True,
            )
            route_code = route_process.wait()
            if route_code != 0:
                raise RuntimeError(
                    f"Moving-camera route capture exited with code {route_code}"
                )
            moving_root = dataset / "raw_images" / "moving"
            moving_root.mkdir(parents=True, exist_ok=True)
            for previous in moving_root.glob("frame_*.png"):
                previous.unlink()
            images = sorted((moving_capture / "images").glob("frame_*.png"))
            if not images:
                raise RuntimeError("Simulation route capture produced no moving frames")
            route_payload = json.loads(route.read_text(encoding="utf-8"))
            expected_frames = len(route_payload.get("frames", []))
            if len(images) != expected_frames:
                raise RuntimeError(
                    "Simulation route capture is incomplete: "
                    f"captured {len(images)}/{expected_frames} fresh frames. "
                    "No input was published and no AP method was started. "
                    "Increase the frame timeout if Gazebo renders slowly."
                )
            diversity = capture_frame_diversity(images)
            for image in images:
                shutil.copy2(image, moving_root / image.name)
            captured_images = sorted(moving_root.glob("frame_*.png"))
            apply_motion_blur(
                captured_images,
                kernel_size=int(mapping.get("motion_blur_kernel", 0)),
                angle_deg=float(mapping.get("motion_blur_angle_deg", 0.0)),
            )
            shutil.copy2(
                moving_capture / "route_commanded.csv", metadata / "route_commanded.csv"
            )
            (metadata / "capture_metadata.json").write_text(
                json.dumps(
                    {
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "capture_id": mapping.get("capture_id"),
                        "preset": mapping.get("preset", "custom"),
                        "world": str(world),
                        "world_name": world_name,
                        "route": str(route),
                        "moving_frames": len(images),
                        "moving_frame_diversity": diversity,
                        "gazebo": {
                            "mode": "headless_server",
                            "command": gazebo_command,
                            "static_camera_update_rate_hz": 1.0,
                        },
                        "camera_parameter_scope": {
                            "camera_overrides": "moving_camera_only",
                            "moving_camera_id": mapping["moving_camera"]["id"],
                            "moving_only_parameters": [
                                "route",
                                "target_route_frames",
                                "route_sampling_strategy",
                                "moving_width",
                                "moving_height",
                                "moving_hfov_deg",
                                "motion_blur_kernel",
                                "motion_blur_angle_deg",
                            ],
                            "static_camera_ids": [
                                camera["id"] for camera in mapping["static_cameras"]
                            ],
                            "static_images_per_camera": 1,
                            "static_camera_model_overrides_applied": False,
                            "lighting_scope": (
                                "world appearance; affects rendered pixels but never "
                                "camera intrinsics"
                            ),
                        },
                        "intrinsics_capture": intrinsic_capture,
                        "simulation_parameters": {
                            key: mapping.get(key)
                            for key in (
                                "route_name",
                                "moving_width",
                                "moving_height",
                                "moving_hfov_deg",
                                "lighting",
                                "lighting_scale",
                                "motion_blur_kernel",
                                "motion_blur_angle_deg",
                                "target_route_frames",
                                "route_sampling_strategy",
                            )
                        },
                        "settings": {
                            key: mapping[key]
                            for key in (
                                "moving_model_name",
                                "settle_seconds",
                                "post_pose_skip",
                                "frame_timeout_seconds",
                                "startup_timeout_seconds",
                            )
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[OK] simulation capture complete: {len(images)} moving frames", flush=True)
        finally:
            _stop(route_process, "moving-camera capture")
            _stop(bridge, "ROS-Gazebo bridge")
            _stop(gazebo, "Gazebo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a new, isolated canonical dataset from Gazebo."
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mapping", required=True)
    args = parser.parse_args()

    def stop_on_signal(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, stop_on_signal)
    capture(
        Path(args.repository).resolve(),
        Path(args.dataset).resolve(),
        Path(args.mapping).resolve(),
    )


if __name__ == "__main__":
    main()
