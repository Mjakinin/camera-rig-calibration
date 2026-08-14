"""Typed internal contract for the one reviewed bus simulation world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..inventory import BASELINE_SIMULATION_PARAMETERS
from .simulation_routes import (
    SimulationRouteAsset,
    discover_local_simulation_routes,
    simulation_route_asset,
)


WORLD_PROFILE_CONTRACT = "rigcal_simulation_world_profile_v1"


def reviewed_bus_world_paths(repository_root: Path) -> tuple[Path, ...]:
    """Return the exact source SDFs maintained by this repository."""

    world_root = (
        repository_root.resolve() / "src/calib_lab/bus_real_data/worlds"
    )
    return (
        (world_root / "bus_real_data_moving_camera.sdf").resolve(),
        *(
            (
                world_root
                / "lighting"
                / f"bus_real_data_moving_camera_light_ceiling_{name}.sdf"
            ).resolve()
            for name in ("dark_extreme", "low", "normal", "bright")
        ),
    )


def validate_reviewed_bus_world(
    repository_root: Path, world: Path
) -> Path:
    """Reject arbitrary SDF paths before Gazebo can load plugins."""

    resolved = world.expanduser().resolve()
    if resolved not in set(reviewed_bus_world_paths(repository_root)):
        raise RuntimeError(
            "Arbitrary simulation SDF paths are disabled. Use the reviewed "
            "bus world and its typed simulation parameters."
        )
    return resolved


@dataclass(frozen=True)
class SimulationCameraProfile:
    id: str
    image_topic: str
    camera_info_topic: str
    model_name: str
    sensor_name: str


@dataclass(frozen=True)
class SimulationWorldProfile:
    schema_version: int
    contract: str
    id: str
    display_name: str
    sdf: Path
    resource_paths: tuple[Path, ...]
    static_cameras: tuple[SimulationCameraProfile, ...]
    moving_camera: SimulationCameraProfile
    routes: tuple[SimulationRouteAsset, ...]
    baseline_route: SimulationRouteAsset
    capabilities: tuple[str, ...]
    lighting_profiles: dict[str, Path | None]
    baseline: dict[str, object]
    trusted_builtin: bool = True


def bus_world_profile(repository_root: Path) -> SimulationWorldProfile:
    root = repository_root.resolve()
    route_root = root / "src/calib_lab/bus_real_data/config"
    world_root = root / "src/calib_lab/bus_real_data/worlds"
    builtin_routes = (
        simulation_route_asset(
            route_root / "moving_camera_route2_interpolated_final.json",
            route_id="route2",
            source="builtin",
        ),
        simulation_route_asset(
            route_root / "moving_camera_route1_interpolated_final.json",
            route_id="route1",
            source="builtin",
        ),
    )
    routes = (*builtin_routes, *discover_local_simulation_routes(root))
    static_cameras = tuple(
        SimulationCameraProfile(
            id=camera_id,
            model_name=camera_id,
            sensor_name=f"{camera_id}_sensor",
            image_topic=f"/bus_real_data/{camera_id}/image",
            camera_info_topic=f"/bus_real_data/{camera_id}/camera_info",
        )
        for camera_id in (
            "cam_edge_0",
            "cam_edge_1",
            "cam_edge_3",
            "cam_edge_5",
        )
    )
    return SimulationWorldProfile(
        schema_version=1,
        contract=WORLD_PROFILE_CONTRACT,
        id="bus",
        display_name="Bus interior calibration world",
        sdf=(world_root / "bus_real_data_moving_camera.sdf").resolve(),
        resource_paths=(
            (root / "src/calib_lab/bus_real_data/models").resolve(),
        ),
        static_cameras=static_cameras,
        moving_camera=SimulationCameraProfile(
            id="moving_calib_camera",
            model_name="moving_calib_camera",
            sensor_name="moving_calib_camera_sensor",
            image_topic="/bus_real_data/moving_calib_camera/image",
            camera_info_topic=(
                "/bus_real_data/moving_calib_camera/camera_info"
            ),
        ),
        routes=routes,
        baseline_route=builtin_routes[0],
        capabilities=(
            "route",
            "density",
            "resolution",
            "fov",
            "lighting",
            "motion_blur",
            "capture",
        ),
        lighting_profiles={
            "baseline": None,
            **{
                name: (
                    world_root
                    / "lighting"
                    / (
                        "bus_real_data_moving_camera_light_ceiling_"
                        f"{name}.sdf"
                    )
                ).resolve()
                for name in ("dark_extreme", "low", "normal", "bright")
            },
            "custom": (
                world_root
                / "lighting"
                / "bus_real_data_moving_camera_light_ceiling_normal.sdf"
            ).resolve(),
        },
        baseline=dict(BASELINE_SIMULATION_PARAMETERS),
    )


__all__ = [
    "WORLD_PROFILE_CONTRACT",
    "SimulationCameraProfile",
    "SimulationWorldProfile",
    "bus_world_profile",
    "reviewed_bus_world_paths",
    "validate_reviewed_bus_world",
]
