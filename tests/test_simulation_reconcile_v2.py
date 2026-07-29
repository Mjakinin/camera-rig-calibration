from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from camera_rig_calibration.evaluation.ap03_derived import (
    ensure_ap03_derived_results,
)
from camera_rig_calibration.evaluation.simulation_ground_truth import (
    GROUND_TRUTH_CONTRACT,
    resolve_simulation_ground_truth,
)


def _write_world(path: Path) -> None:
    path.write_text(
        """
<sdf version="1.9">
  <world name="test">
    <model name="cam0"><pose>0 0 1 0 0 0</pose></model>
    <model name="cam1"><pose>1 0 1 0 0 0</pose></model>
    <model name="marker_014"><pose>0 2 0 0 0 0</pose></model>
  </world>
</sdf>
""".strip(),
        encoding="utf-8",
    )


def test_ground_truth_rejects_empty_cache_and_dataset_camera_drift(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "experiment"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(
        json.dumps(
            {"static_cameras": [{"id": "cam0"}, {"id": "cam1"}]}
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.sdf"
    _write_world(source)
    first = resolve_simulation_ground_truth(
        dataset, world_path=source
    )
    assert first.payload["status"] == "available"
    assert first.payload["contract"] == GROUND_TRUTH_CONTRACT

    ground_truth = (
        dataset / "metadata" / "simulation" / "ground_truth.json"
    )
    broken = dict(first.payload)
    broken["static_cameras"] = {}
    ground_truth.write_text(json.dumps(broken), encoding="utf-8")
    source.write_text("<sdf version=\"1.9\"/>", encoding="utf-8")

    repaired = resolve_simulation_ground_truth(
        dataset, world_path=source, backfilled=True
    )
    assert repaired.regenerated is True
    assert repaired.payload["status"] == "available"
    assert set(repaired.payload["static_cameras"]) == {"cam0", "cam1"}
    assert repaired.payload["world_sha256"] == first.payload["world_sha256"]

    (dataset / "dataset.json").write_text(
        json.dumps(
            {
                "static_cameras": [
                    {"id": "cam0"},
                    {"id": "cam1"},
                    {"id": "cam2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    unavailable = resolve_simulation_ground_truth(dataset)
    assert unavailable.payload["status"] == "unavailable"
    assert unavailable.payload["missing_static_cameras"] == ["cam2"]
    assert unavailable.payload.get("static_cameras") in (None, {})


def _write_pose_csv(path: Path, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "entity_type",
        "entity_id",
        "source",
        "reference_frame",
        "transform_convention",
        "x_m",
        "y_m",
        "z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "rvec_x",
        "rvec_y",
        "rvec_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, camera in enumerate(("cam0", "cam1", "cam2", "cam3")):
            writer.writerow(
                {
                    "entity_type": "static_camera",
                    "entity_id": camera,
                    "source": "fixture",
                    "reference_frame": "colmap",
                    "transform_convention": "T_reference_camera",
                    "x_m": scale * index,
                    "y_m": 0,
                    "z_m": 0,
                    "roll_deg": 0,
                    "pitch_deg": 0,
                    "yaw_deg": 0,
                    "rvec_x": 0,
                    "rvec_y": 0,
                    "rvec_z": 0,
                }
            )


def test_ap03_single_and_multi_are_derived_from_one_container(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "route2"
    observations = experiment / "observations"
    observations.mkdir(parents=True)
    (observations / "SELECTION_CANDIDATES.json").write_text(
        json.dumps({"evaluation_anchor": {"selected": 14}}),
        encoding="utf-8",
    )
    container = experiment / "methods" / "ap03" / "baseline"
    container.mkdir(parents=True)
    (container / "RESULT.json").write_text(
        json.dumps(
            {
                "method": "ap03",
                "label": "baseline",
                "artifact_status": "available",
                "primary_result": "multi",
            }
        ),
        encoding="utf-8",
    )
    corners_path = (
        container
        / "diagnostics"
        / "method"
        / "scale_multi"
        / "AP03_MARKER_SIZE_SCALE_ONLY_TRIANGULATED_CORNERS.csv"
    )
    corners_path.parent.mkdir(parents=True)
    with corners_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "marker_id",
                "corner_idx",
                "status",
                "x_colmap",
                "y_colmap",
                "z_colmap",
            ],
        )
        writer.writeheader()
        for index, point in enumerate(
            (
                (-0.1, 0.1, 0.0),
                (0.1, 0.1, 0.0),
                (0.1, -0.1, 0.0),
                (-0.1, -0.1, 0.0),
            )
        ):
            writer.writerow(
                {
                    "marker_id": 14,
                    "corner_idx": index,
                    "status": "OK",
                    "x_colmap": point[0],
                    "y_colmap": point[1],
                    "z_colmap": point[2],
                }
            )
    for mode, scale in (("single", 0.75), ("multi", 0.80)):
        scale_root = (
            container / "diagnostics" / "method" / f"scale_{mode}"
        )
        scale_root.mkdir(parents=True, exist_ok=True)
        (scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json").write_text(
            json.dumps(
                {
                    "best_model": "0",
                    "scale_m_per_colmap_unit": scale,
                    "used_rel_std_scale": 0.02,
                    "registered_static_cameras": 4,
                    "num_sparse_points3d": 10,
                }
            ),
            encoding="utf-8",
        )
        _write_pose_csv(
            scale_root
            / "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv",
            scale,
        )

    outcomes = ensure_ap03_derived_results(experiment)

    assert outcomes["ap03_single/baseline"]["scale_m_per_colmap_unit"] == 0.75
    assert outcomes["ap03_multi/baseline"]["scale_m_per_colmap_unit"] == 0.80
    single = json.loads(
        (
            experiment / "methods" / "ap03_single" / "baseline" / "RESULT.json"
        ).read_text(encoding="utf-8")
    )
    multi = json.loads(
        (
            experiment / "methods" / "ap03_multi" / "baseline" / "RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert single["primary_result"] == "single"
    assert multi["primary_result"] == "multi"
    assert single["metrics"]["ap03_scale"]["scale_m_per_colmap_unit"] == 0.75
    assert multi["metrics"]["ap03_scale"]["scale_m_per_colmap_unit"] == 0.80
    assert (
        single["shared_colmap_container"]
        == multi["shared_colmap_container"]
        == "methods/ap03/baseline"
    )
    shared = json.loads(
        (
            container
            / "diagnostics"
            / "derived"
            / "shared_anchor_geometry"
            / "marker_14_corners_colmap.json"
        ).read_text(encoding="utf-8")
    )
    assert len(shared["corners"]) == 4
    assert np.isclose(
        float(
            next(
                csv.DictReader(
                    (
                        experiment
                        / "methods"
                        / "ap03_single"
                        / "baseline"
                        / "camera_extrinsics.csv"
                    ).open(newline="", encoding="utf-8")
                )
            )["x_m"]
        ),
        0.0,
    )
    base = json.loads((container / "RESULT.json").read_text(encoding="utf-8"))
    assert base["comparison_visibility"] == (
        "hidden_when_scale_variants_available"
    )
