from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from camera_rig_calibration.evaluation.reporting import (
    PoseRecord,
    _ap02_marker_map,
    _camera_map_rows,
    _factor_report,
    _real_results_text,
    _real_variant_disagreement,
    _simulation_pairwise,
    ensure_simulation_ground_truth,
    pairwise_rows,
    refresh_method_reports,
)
from camera_rig_calibration.evaluation.marker_consistency import (
    report as write_marker_report,
)
from camera_rig_calibration.methods.common.geometry import make_T, rpy_to_R


def _pose(
    entity_id: str,
    transform: np.ndarray,
) -> PoseRecord:
    return PoseRecord(
        entity_id=entity_id,
        transform=transform,
        source="synthetic",
        reference_frame="synthetic reference",
        transform_convention="T_reference_camera",
    )


def _write_poses(path: Path, poses: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "reference_frame",
        "transform_convention",
        "entity_type",
        "entity_id",
        "source",
        "x_m",
        "y_m",
        "z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entity_id, transform in poses.items():
            writer.writerow(
                {
                    "reference_frame": "synthetic reference",
                    "transform_convention": "T_reference_camera",
                    "entity_type": "static_camera",
                    "entity_id": entity_id,
                    "source": "synthetic",
                    "x_m": transform[0, 3],
                    "y_m": transform[1, 3],
                    "z_m": transform[2, 3],
                    "roll_deg": 0.0,
                    "pitch_deg": 0.0,
                    "yaw_deg": 0.0,
                }
            )


def test_pairwise_gt_is_gauge_invariant_and_has_six_pairs() -> None:
    ground_truth = {
        "cam0": make_T(np.eye(3), [0.0, 0.0, 0.0]),
        "cam1": make_T(np.eye(3), [1.0, 0.0, 0.0]),
        "cam2": make_T(np.eye(3), [0.0, 2.0, 0.0]),
        "cam3": make_T(np.eye(3), [0.0, 0.0, 3.0]),
    }
    gauge = make_T(
        rpy_to_R(0.2, -0.1, 0.3),
        [4.0, -2.0, 7.0],
    )
    estimated = {
        name: _pose(name, gauge @ transform)
        for name, transform in ground_truth.items()
    }

    rows = _simulation_pairwise(
        "ap_test",
        "baseline",
        estimated,
        ground_truth,
    )

    assert len(rows) == 6
    assert max(row["translation_error_cm"] for row in rows) < 1e-10
    assert max(row["rotation_error_deg"] for row in rows) < 1e-8
    assert max(row["baseline_error_cm"] for row in rows) < 1e-10
    assert max(row["direction_error_deg"] for row in rows) < 1e-6

    aligned = _camera_map_rows(
        "ap_test",
        "baseline",
        estimated,
        ground_truth,
    )
    assert len(aligned) == 4
    assert max(row["translation_error_cm"] for row in aligned) < 1e-10
    assert max(row["rotation_error_deg"] for row in aligned) < 1e-8


def test_real_variant_comparison_reports_agreement_not_accuracy() -> None:
    poses = {
        f"cam{index}": _pose(
            f"cam{index}",
            make_T(np.eye(3), [float(index), 0.0, 0.0]),
        )
        for index in range(4)
    }
    first = pairwise_rows(poses, method="ap01", label="baseline")
    second = pairwise_rows(poses, method="ap02", label="baseline")

    summaries, detailed = _real_variant_disagreement(first + second)

    assert len(summaries) == 1
    assert summaries[0]["pair_count"] == 6
    assert summaries[0]["mean_translation_delta_cm"] == 0.0
    assert summaries[0]["mean_rotation_delta_deg"] == 0.0
    assert len(detailed) == 6


def test_real_report_starts_with_marker_metric_and_has_no_how_to_text(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    evaluation = experiment / "evaluations" / "anchor"
    evaluation.mkdir(parents=True)
    marker_report = evaluation / "REAL_DATA_MARKER_CONSISTENCY.txt"
    write_marker_report(
        marker_report,
        experiment,
        3,
        0.17,
        [
            {
                "method": "AP02__combined_nfev_60",
                "status": "OK",
                "available_static_camera_count": 4,
                "registered_moving_frames": 20,
                "evaluated_non_anchor_markers": 2,
                "median_absolute_size_error_cm": 0.2,
                "p90_absolute_size_error_cm": 0.4,
                "moving_to_static_reprojection_rmse_px": 1.1,
                "moving_to_static_reprojection_observations": 96,
            }
        ],
        [],
    )

    marker_text = marker_report.read_text(encoding="utf-8")
    combined, _ = _real_results_text(experiment, [])

    assert "Cross P90" not in marker_text
    assert "HOW TO READ" not in marker_text
    assert "COMMON CALCULATION" not in marker_text
    assert "INTERPRETATION" not in marker_text
    assert combined.index("REAL-DATA SINGLE-ANCHOR") < combined.index(
        "METHOD / VARIANT OVERVIEW"
    )


def test_ground_truth_snapshot_is_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(
        json.dumps(
            {
                "static_cameras": [
                    {"id": "cam0"},
                    {"id": "cam1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    world = tmp_path / "captured.sdf"
    world.write_text(
        """
<sdf version="1.9">
  <world name="test">
    <model name="cam0"><pose>0 0 1 0 0 0</pose></model>
    <model name="cam1"><pose>1 0 1 0 0 0</pose></model>
    <model name="marker_007"><pose>0 2 0 0 0 0</pose></model>
  </world>
</sdf>
""".strip(),
        encoding="utf-8",
    )

    first = ensure_simulation_ground_truth(
        dataset,
        world_path=world,
    )
    snapshot = (
        dataset / "metadata" / "simulation" / "world_snapshot.sdf"
    )
    before = snapshot.read_bytes()
    second = ensure_simulation_ground_truth(
        dataset,
        world_path=world,
    )

    assert first == second
    assert first["status"] == "available"
    assert first["snapshot_origin"] == "captured_before_calibration"
    assert set(first["static_cameras"]) == {"cam0", "cam1"}
    assert set(first["markers"]) == {"7"}
    assert snapshot.read_bytes() == before == world.read_bytes()


def test_ap02_limit_is_quality_warning_not_artifact_failure(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    method = experiment / "methods" / "ap02" / "variant2"
    poses = {
        f"cam{index}": make_T(np.eye(3), [float(index), 0.0, 0.0])
        for index in range(4)
    }
    _write_poses(method / "camera_extrinsics.csv", poses)
    (method / "RESULT.json").write_text(
        json.dumps(
            {
                "method": "ap02",
                "label": "variant2",
                "runtime_seconds": 12.5,
                "primary_result": "combined",
            }
        ),
        encoding="utf-8",
    )
    provenance = method / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "resolved_config.yaml").write_text(
        """
methods:
  ap02:
    reference_marker_id: 3
    static_only_ba_max_function_evaluations: 100
    combined_ba_max_function_evaluations: 60
    ba_robust_loss: soft_l1
    ba_robust_loss_scale_px: 3.0
observation_quality:
  maximum_pnp_reprojection_error_px: 25.0
""".strip(),
        encoding="utf-8",
    )
    report = method / "diagnostics" / "method" / "final_results"
    report.mkdir(parents=True)
    (report / "AP02_REPORT.json").write_text(
        json.dumps(
            {
                "combined_optimizer": {
                    "success": False,
                    "nfev": 60,
                    "maximum_function_evaluations": 60,
                    "message": (
                        "The maximum number of function evaluations "
                        "is exceeded."
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    [payload] = refresh_method_reports(experiment)

    assert payload["artifact_status"] == "available"
    assert payload["quality_status"] == "optimizer_limit_reached"
    assert payload["static_camera_count"] == 4
    assert payload["pairwise_camera_count"] == 6
    assert "60 evaluations" in payload["warnings"][0]
    assert (method / "pairwise_camera_extrinsics.csv").is_file()
    assert "FINAL STATIC-CAMERA POSES" in (
        method / "RESULT.txt"
    ).read_text(encoding="utf-8")


def test_ap02_marker_map_has_direct_and_best_fit_results(
    tmp_path: Path,
) -> None:
    method = tmp_path / "methods" / "ap02" / "baseline"
    method.mkdir(parents=True)
    (method / "RESULT.json").write_text(
        json.dumps(
            {
                "method": "ap02",
                "label": "baseline",
                "reference_marker_id": 7,
            }
        ),
        encoding="utf-8",
    )
    gt_cameras = {
        f"cam{index}": make_T(np.eye(3), [float(index), 0.0, 1.0])
        for index in range(4)
    }
    gt_markers = {
        marker: make_T(np.eye(3), [float(marker - 7), 2.0, 0.0])
        for marker in (7, 8, 9, 10)
    }
    world_from_reference = gt_markers[7]
    reference_from_world = np.linalg.inv(world_from_reference)
    _write_poses(
        method / "camera_extrinsics.csv",
        {
            name: reference_from_world @ transform
            for name, transform in gt_cameras.items()
        },
    )
    _write_poses(
        method
        / "diagnostics"
        / "method"
        / "graph_ba"
        / "with_moving"
        / "optimized_marker_poses_ref_marker.csv",
        {
            str(marker): reference_from_world @ transform
            for marker, transform in gt_markers.items()
        },
    )

    result, text = _ap02_marker_map(
        method,
        gt_cameras,
        gt_markers,
    )

    direct = result["direct_reference_frame"]
    fitted = result["best_fit_diagnostic"]
    assert result["status"] == "available"
    assert direct["summary"]["count"] == 4
    assert fitted["summary"]["count"] == 4
    assert max(
        row["translation_error_cm"] for row in direct["rows"]
    ) < 1e-10
    assert max(
        row["translation_error_cm"] for row in fitted["rows"]
    ) < 1e-10
    reference_row = next(
        row for row in direct["rows"] if row["marker_id"] == 7
    )
    assert reference_row["translation_error_cm"] == 0.0
    assert "DIRECT REFERENCE-MARKER FRAME" in text
    assert "BEST-FIT SE(3) DIAGNOSTIC" in text


def test_factor_report_includes_baseline_and_excludes_mixed(
    tmp_path: Path,
) -> None:
    simulation = tmp_path / "results" / "simulation"
    pair = {
        "method": "ap01",
        "label": "baseline",
        "pair": "cam0-cam1",
        "translation_error_cm": 1.0,
        "rotation_error_deg": 2.0,
        "gt_baseline_m": 3.0,
        "estimated_baseline_m": 3.01,
        "baseline_error_cm": 1.0,
        "direction_error_deg": 0.5,
    }

    def write_result(
        path: Path,
        *,
        factor: str,
        value: str,
    ) -> None:
        path.mkdir(parents=True)
        (path / "RESULTS.json").write_text(
            json.dumps(
                {
                    "category": "simulation",
                    "storage": {"factor": factor, "value": value},
                    "primary_camera_pairwise": {"rows": [pair]},
                }
            ),
            encoding="utf-8",
        )

    write_result(
        simulation / "baseline" / "route2",
        factor="baseline",
        value="route2",
    )
    write_result(
        simulation / "fov" / "fov_40deg",
        factor="fov",
        value="40deg",
    )
    write_result(
        simulation / "fov" / "mixed_experiment",
        factor="mixed",
        value="mixed",
    )

    _factor_report(simulation / "fov", "fov")

    text = (simulation / "fov" / "RESULTS.txt").read_text(
        encoding="utf-8"
    )
    payload = json.loads(
        (simulation / "fov" / "RESULTS.json").read_text(
            encoding="utf-8"
        )
    )
    assert "baseline (route2)" in text
    assert "40deg" in text
    assert "mixed_experiment" not in text
    assert {
        row["factor_value"] for row in payload["rows"]
    } == {"baseline (route2)", "40deg"}
