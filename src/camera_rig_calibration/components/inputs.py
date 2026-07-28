"""Input-source adapters used during rigcal preflight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config.models import RigConfig
from ..contracts import CommandSpec, RequirementResult, RunContext


@dataclass(frozen=True)
class PreparedInputAdapter:
    """Validate an already normalized rigcal dataset."""

    id: str = "prepared_dataset"
    display_name: str = "Prepared dataset"

    def matches(self, config: RigConfig) -> bool:
        return config.dataset.prepared_root is not None

    def requirements(self, config: RigConfig) -> RequirementResult:
        root = config.dataset.prepared_root
        if root is None:
            return RequirementResult.unavailable("dataset.prepared_root is not set")
        if not root.exists():
            return RequirementResult.unavailable(
                f"prepared dataset does not exist: {root}"
            )
        dataset_root = root.parent if root.name == "raw_images" else root
        raw = dataset_root / "raw_images"
        reasons = []
        for camera in config.static_cameras:
            if not any((raw / "static").glob(f"{camera.id}.*")):
                reasons.append(f"prepared static image is missing for '{camera.id}'")
            if not (raw / "camera_info" / f"{camera.id}.json").is_file():
                reasons.append(f"prepared intrinsics are missing for '{camera.id}'")
        if not any((raw / "moving").glob("frame_*.*")):
            reasons.append("prepared moving frames are missing")
        if not (
            raw / "camera_info" / f"{config.moving_camera.id}.json"
        ).is_file():
            reasons.append(
                f"prepared intrinsics are missing for '{config.moving_camera.id}'"
            )
        return (
            RequirementResult.unavailable(*reasons)
            if reasons
            else RequirementResult.ok()
        )

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()


@dataclass(frozen=True)
class FilesystemInputAdapter:
    """Validate videos, extracted frames and direct images."""

    id: str = "filesystem"
    display_name: str = "Videos, frames and direct images"

    def matches(self, config: RigConfig) -> bool:
        return (
            config.dataset.prepared_root is None
            and config.mcap.path is None
            and not config.simulation.enabled
        )

    def requirements(self, config: RigConfig) -> RequirementResult:
        reasons = []
        moving = config.moving_camera
        for label, path, expected_kind in (
            ("moving video", moving.video, "file"),
            ("moving frames", moving.frames, "directory"),
            ("moving intrinsics", moving.intrinsics, "file"),
            (
                "intrinsic calibration video",
                moving.intrinsic_calibration_video,
                "file",
            ),
            (
                "intrinsic calibration images",
                moving.intrinsic_calibration_images,
                "directory",
            ),
        ):
            if path is None:
                continue
            exists = path.is_file() if expected_kind == "file" else path.is_dir()
            if not exists:
                reasons.append(f"{label} does not exist: {path}")
        for camera in config.static_cameras:
            for path in camera.images:
                if not path.is_file():
                    reasons.append(f"static image does not exist: {path}")
            if camera.video is not None and not camera.video.is_file():
                reasons.append(f"static video does not exist: {camera.video}")
            if camera.intrinsics is not None and not camera.intrinsics.is_file():
                reasons.append(f"static intrinsics do not exist: {camera.intrinsics}")
        return (
            RequirementResult.unavailable(*reasons)
            if reasons
            else RequirementResult.ok()
        )

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()


@dataclass(frozen=True)
class McapInputAdapter:
    """Validate MCAP/ROS bag topics and optional local moving-camera input."""

    id: str = "mcap"
    display_name: str = "MCAP or ROS bag"

    def matches(self, config: RigConfig) -> bool:
        return config.mcap.path is not None

    def requirements(self, config: RigConfig) -> RequirementResult:
        if config.mcap.path is None:
            return RequirementResult.unavailable("mcap.path is not set")
        missing = [
            camera.id for camera in config.static_cameras if not camera.image_topic
        ]
        if missing:
            return RequirementResult.unavailable(
                f"image topics are missing for cameras: {', '.join(missing)}"
            )
        if not config.mcap.path.is_file():
            return RequirementResult.unavailable(
                f"MCAP does not exist: {config.mcap.path}"
            )
        moving = config.moving_camera
        if moving.video is None and moving.frames is None and not moving.image_topic:
            return RequirementResult.unavailable(
                "moving camera needs a local video/frame folder or an MCAP image topic"
            )
        return FilesystemInputAdapter().requirements(config)

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()


@dataclass(frozen=True)
class SimulationInputAdapter:
    """Validate the built-in Gazebo bus capture configuration."""

    id: str = "simulation"
    display_name: str = "Gazebo simulation capture"

    def matches(self, config: RigConfig) -> bool:
        return config.simulation.enabled

    def requirements(self, config: RigConfig) -> RequirementResult:
        simulation = config.simulation
        reasons = []
        if simulation.world is None or not simulation.world.is_file():
            reasons.append(f"simulation world does not exist: {simulation.world}")
        if simulation.route is None or not simulation.route.is_file():
            reasons.append(f"simulation route does not exist: {simulation.route}")
        for camera in config.static_cameras:
            if not camera.image_topic:
                reasons.append(f"image topic is missing for '{camera.id}'")
            if not camera.camera_info_topic:
                reasons.append(f"CameraInfo topic is missing for '{camera.id}'")
        if not config.moving_camera.image_topic:
            reasons.append("moving-camera image topic is missing")
        if not config.moving_camera.camera_info_topic:
            reasons.append("moving-camera CameraInfo topic is missing")
        return (
            RequirementResult.unavailable(*reasons)
            if reasons
            else RequirementResult.ok()
        )

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()
