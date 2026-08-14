from __future__ import annotations

import json
from pathlib import Path

import pytest

from camera_rig_calibration.config.models import DatasetCategory
from camera_rig_calibration.experiments import experiment_fingerprint
from camera_rig_calibration.input.simulation_routes import (
    discover_local_simulation_routes,
    load_simulation_route,
    simulation_route_manifest,
)


def _route(path: Path, *, last_x: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame": 0,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 1.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                    },
                    {
                        "frame": 1,
                        "segment": "return",
                        "x": last_x,
                        "y": 0.0,
                        "z": 1.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.5,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_data_local_routes_are_discovered_with_stable_identity(
    tmp_path: Path,
) -> None:
    route = tmp_path / "data_local/simulation_routes/lab/sweep.json"
    _route(route)

    discovered = discover_local_simulation_routes(tmp_path)

    assert len(discovered) == 1
    assert discovered[0].id == "lab__sweep"
    assert discovered[0].source == "data_local"
    assert discovered[0].frame_count == 2
    assert simulation_route_manifest(route)["sha256"] == discovered[0].sha256


def test_invalid_route_is_rejected_before_capture(tmp_path: Path) -> None:
    route = tmp_path / "invalid.json"
    _route(route)
    payload = json.loads(route.read_text(encoding="utf-8"))
    payload["frames"][1]["frame"] = 0
    route.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_simulation_route(route)


def test_unknown_route_contract_is_rejected(tmp_path: Path) -> None:
    route = tmp_path / "unknown-contract.json"
    _route(route)
    payload = json.loads(route.read_text(encoding="utf-8"))
    payload["contract"] = "some_other_route_v1"
    route.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rigcal_simulation_route_v1"):
        load_simulation_route(route)


def test_discovery_rejects_colliding_stable_route_ids(tmp_path: Path) -> None:
    root = tmp_path / "data_local/simulation_routes"
    _route(root / "a b.json")
    _route(root / "a?b.json")

    with pytest.raises(ValueError, match="Duplicate local simulation route ID"):
        discover_local_simulation_routes(tmp_path)


def test_route_content_changes_experiment_identity(
    prepared_config, tmp_path: Path
) -> None:
    route = tmp_path / "route.json"
    world = tmp_path / "world.sdf"
    _route(route, last_x=1.0)
    world.write_text(
        '<sdf version="1.8"><world name="bus"/></sdf>', encoding="utf-8"
    )
    config = prepared_config.model_copy(deep=True)
    config.dataset.category = DatasetCategory.SIMULATION
    config.simulation.enabled = True
    config.simulation.route = route
    config.simulation.world = world
    first = experiment_fingerprint(config)

    _route(route, last_x=2.0)

    assert experiment_fingerprint(config) != first
