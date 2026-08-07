from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from camera_rig_calibration.config import load_config, save_config
from camera_rig_calibration.config.models import DatasetCategory, SceneType
from camera_rig_calibration.dataset_identity import (
    build_dataset_identity,
    identities_match,
)
from camera_rig_calibration.experiments import experiment_fingerprint
from camera_rig_calibration.methods.ap01 import core as ap01_core
from camera_rig_calibration.methods.ap01.solve_extrinsics import (
    run as solve_ap01,
)
from camera_rig_calibration.methods.ap02.initialize import (
    build_graph,
    main_compat_widest_path_tree,
    marker_node,
    maximum_bottleneck_tree,
)
from camera_rig_calibration import rerun as rerun_module
from camera_rig_calibration.rerun import prepare_single_method_rerun
from camera_rig_calibration.runtime import PipelineOrchestrator, observation_id


def _identity_dataset(root: Path, *, image: bytes = b"image") -> Path:
    (root / "raw_images/static").mkdir(parents=True)
    (root / "raw_images/moving").mkdir()
    (root / "raw_images/camera_info").mkdir()
    (root / "metadata/simulation").mkdir(parents=True)
    (root / "raw_images/static/cam.png").write_bytes(image)
    (root / "raw_images/moving/frame.png").write_bytes(b"moving")
    (root / "raw_images/camera_info/cam.json").write_bytes(b"camera-info")
    (root / "metadata/simulation/world_snapshot.sdf").write_bytes(b"world")
    (root / "metadata/simulation/route_commanded.csv").write_bytes(b"route")
    (root / "metadata/dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "fixture",
                "scene_type": "simulation",
                "static_cameras": [{"id": "cam"}],
                "moving_camera": {"id": "moving"},
                "marker_dictionary": "DICT_4X4_50",
                "marker_length_m": 0.17,
                "simulation_parameters": {"route": "route2"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_frozen_observations(
    root: Path,
    *,
    dictionary: str = "DICT_4X4_50",
    length_m: float = 0.17,
    detection_mode: str = "baseline",
    input_id: str = "input_fixture",
    frozen_id: str = "detection_legacy_opencv",
) -> None:
    observations = root / "observations"
    observations.mkdir(exist_ok=True)
    (observations / "detection_config.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "input_id": input_id,
                "observation_id": frozen_id,
                "markers": {
                    "dictionary": dictionary,
                    "length_m": length_m,
                    "accepted_ids": "all_detected",
                    "detection_mode": detection_mode,
                },
                "effective_detector": {
                    "contract": "rigcal_aruco_detector_v2",
                    "mode": detection_mode,
                    "dictionary": dictionary,
                    "opencv_version": "4.5.4",
                },
                "detector_contract": "rigcal_aruco_detector_v2",
            }
        ),
        encoding="utf-8",
    )
    header = (
        "detection_success,detection_mode,detection_source,"
        "detector_contract,opencv_version,pnp_reprojection_rmse_px,"
        "corner0_u,corner3_v\n"
    )
    for name in (
        "shared_static_aruco_observations.csv",
        "shared_moving_aruco_observations.csv",
        "shared_all_aruco_observations.csv",
    ):
        (observations / name).write_text(header, encoding="utf-8")


def test_dataset_identity_changes_only_with_acquisition_content(
    tmp_path: Path,
) -> None:
    first = _identity_dataset(tmp_path / "first")
    second = _identity_dataset(tmp_path / "second")
    assert identities_match(
        build_dataset_identity(first), build_dataset_identity(second)
    )

    (second / "raw_images/static/cam.png").write_bytes(b"different")
    assert not identities_match(
        build_dataset_identity(first), build_dataset_identity(second)
    )
    (second / "raw_images/static/cam.png").write_bytes(b"image")
    (second / "raw_images/camera_info/cam.json").write_bytes(b"different")
    assert not identities_match(
        build_dataset_identity(first), build_dataset_identity(second)
    )
    (second / "raw_images/camera_info/cam.json").write_bytes(b"camera-info")
    (second / "metadata/simulation/world_snapshot.sdf").write_bytes(
        b"different"
    )
    assert not identities_match(
        build_dataset_identity(first), build_dataset_identity(second)
    )


def test_method_settings_do_not_change_experiment_dataset_contract(
    prepared_config,
) -> None:
    baseline = prepared_config.model_copy(deep=True)
    fingerprints = {experiment_fingerprint(baseline)}

    enabled = baseline.model_copy(deep=True)
    enabled.methods.enabled = ["ap01"]
    fingerprints.add(experiment_fingerprint(enabled))

    ap02_limits = baseline.model_copy(deep=True)
    ap02_limits.methods.ap02.static_only_ba_max_function_evaluations = 37
    ap02_limits.methods.ap02.combined_ba_max_function_evaluations = 41
    fingerprints.add(experiment_fingerprint(ap02_limits))

    ap02_anchor = baseline.model_copy(deep=True)
    ap02_anchor.methods.ap02 = ap02_anchor.methods.ap02.model_copy(
        update={
            "reference_marker_selection_mode": "explicit",
            "reference_marker_id": 7,
        }
    )
    fingerprints.add(experiment_fingerprint(ap02_anchor))

    colmap = baseline.model_copy(deep=True)
    colmap.colmap.compute_mode = "gpu"
    colmap.colmap.maximum_features = 2048
    fingerprints.add(experiment_fingerprint(colmap))

    assert len(fingerprints) == 1


def _observation(
    observer: str, marker: int, score: float
) -> dict[str, str]:
    return {
        "observer_id": observer,
        "observer_type": (
            "static" if observer.startswith("cam") else "moving"
        ),
        "marker_id": str(marker),
        "pnp_success": "true",
        "selection_score": str(score),
        "pnp_reprojection_rmse_px": "1",
        "marker_area_ratio": "0.01",
        "area_px2": "1000",
        "distance_m": "1",
        "tvec_z_m": "1",
        "rvec_x": "0",
        "rvec_y": "0",
        "rvec_z": "0",
        "tvec_x_m": "0",
        "tvec_y_m": "0",
        "frame_id": observer,
        "image_path": observer + ".png",
    }


def test_main_compat_ap02_uses_strongest_frontier_edge() -> None:
    rows = [
        _observation("cam_a", 14, 0.90),
        _observation("cam_a", 1, 0.20),
        _observation("cam_b", 14, 0.60),
        _observation("cam_b", 1, 0.95),
    ]
    graph = build_graph(rows)
    start = marker_node(14)
    parent, _metrics = main_compat_widest_path_tree(graph, start)
    v2_parent, _v2_metrics = maximum_bottleneck_tree(graph, start)
    assert parent[("observer", "cam_a")][0] == start
    assert ("observer", "cam_b") in parent
    assert ("observer", "cam_b") in v2_parent


def _transform(x: float) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[0, 3] = x
    return value


def test_ap01_relay_aggregates_independent_marker_chains() -> None:
    candidates = []
    for root_marker, target_marker, center in ((1, 2, 1.0), (3, 4, 1.1)):
        for offset in (-0.02, 0.0, 0.02, 0.01):
            candidates.append(
                {
                    "root_marker": root_marker,
                    "target_marker": target_marker,
                    "quality": 1.0,
                    "T": _transform(center + offset),
                }
            )
    pose, stats, chains = ap01_core.aggregate_relay_marker_chains(
        candidates
    )
    assert np.isfinite(pose).all()
    assert stats["raw_candidate_count"] == 8
    assert stats["independent_marker_chain_count"] == 2
    assert len(chains) == 2


def test_ap01_keeps_finite_rejected_estimate_for_evaluation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "02_AP01"
    candidates = output / "03_candidates"
    candidates.mkdir(parents=True)
    rows = []
    for marker, x in ((1, 0.0), (2, 1.0), (3, 2.0)):
        rows.append(
            {
                "mode": "direct",
                "root_camera": "root",
                "target_camera": "target",
                "root_marker": marker,
                "target_marker": marker,
                "root_frame": "",
                "target_frame": "",
                "quality": 1.0,
                "transform": _transform(x).tolist(),
            }
        )
    (candidates / "transform_candidates.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    solve_ap01(
        dataset=tmp_path,
        observations_root=tmp_path,
        output_root=output,
        camera_ids=("root", "target"),
        root_camera="root",
        moving_camera_id="moving",
        method_contract="recommended_wizard_v1",
    )
    solution = json.loads(
        (
            output / "03_static_extrinsics/solution_summary.json"
        ).read_text(encoding="utf-8")
    )
    status = solution["camera_statuses"]["target"]
    assert status["estimate_status"] == "available"
    assert status["deployment_eligible"] is False
    assert status["evaluation_status"] == "available"
    with (
        output
        / "03_static_extrinsics/AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
    ).open(newline="", encoding="utf-8") as handle:
        assert {row["entity_id"] for row in csv.DictReader(handle)} == {
            "root",
            "target",
        }
    with (
        output / "03_static_extrinsics/AP01_STATIC_CAMERA_POSES_ACCEPTED.csv"
    ).open(newline="", encoding="utf-8") as handle:
        assert {row["entity_id"] for row in csv.DictReader(handle)} == {
            "root"
        }


def test_prepare_single_method_rerun_reuses_dataset_without_capture(
    prepared_config,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    experiment = (
        repository
        / "results/simulation/baseline/route2_cpu_ref14_50x50"
    )
    _identity_dataset(experiment)
    (experiment / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "id": experiment.name,
                "category": "simulation",
                "input_fingerprint": "input_fixture",
            }
        ),
        encoding="utf-8",
    )
    _write_frozen_observations(
        experiment,
        dictionary=prepared_config.markers.dictionary,
        length_m=prepared_config.markers.length_m,
        detection_mode=prepared_config.markers.detection_mode,
    )
    config = prepared_config.model_copy(
        update={
            "project": prepared_config.project.model_copy(
                update={
                    "experiment_id": experiment.name,
                    "run_label": "baseline",
                }
            ),
            "methods": prepared_config.methods.model_copy(
                update={"enabled": ["ap02"]}, deep=True
            ),
            "dataset": prepared_config.dataset.model_copy(
                update={
                    "scene_type": SceneType.SIMULATION,
                    "category": DatasetCategory.SIMULATION,
                },
                deep=True,
            ),
        },
        deep=True,
    )
    attempt = (
        experiment
        / "attempts/ap02/baseline/20260101_000000/diagnostics"
    )
    save_config(config, attempt / "resolved_config.yaml")
    prepared = prepare_single_method_rerun(
        repository_root=repository,
        experiment=experiment,
        method="ap02",
        variant="baseline",
        reuse_prepared_input=True,
        reuse_matching_intermediates=False,
    )
    manifest = json.loads(
        (
            prepared.transaction_root
            / "jobs/queue_preflight/reused_prepared_dataset/run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["capture_repeated"] is False
    assert manifest["detection_repeated"] is False
    assert identities_match(
        prepared.dataset_identity,
        build_dataset_identity(prepared.transaction_root / "dataset"),
    )
    assert (
        prepared.observation_contract["observation_id"]
        == "detection_legacy_opencv"
    )


def test_ap01_single_method_rerun_accepts_explicit_contract(
    prepared_config,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    experiment = tmp_path / "experiment"
    repository.mkdir()
    experiment.mkdir()
    _identity_dataset(experiment)
    (experiment / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "id": experiment.name,
                "category": "simulation",
                "input_fingerprint": "input_fixture",
            }
        ),
        encoding="utf-8",
    )
    _write_frozen_observations(
        experiment,
        dictionary=prepared_config.markers.dictionary,
        length_m=prepared_config.markers.length_m,
        detection_mode=prepared_config.markers.detection_mode,
    )
    config = prepared_config.model_copy(
        update={
            "project": prepared_config.project.model_copy(
                update={
                    "experiment_id": experiment.name,
                    "run_label": "baseline",
                }
            ),
            "methods": prepared_config.methods.model_copy(
                update={"enabled": ["ap01"]}, deep=True
            ),
        },
        deep=True,
    )
    attempt = (
        experiment
        / "attempts/ap01/baseline/20260101_000000/diagnostics"
    )
    save_config(config, attempt / "resolved_config.yaml")

    prepared = prepare_single_method_rerun(
        repository_root=repository,
        experiment=experiment,
        method="ap01",
        variant="baseline",
        reuse_prepared_input=True,
        reuse_matching_intermediates=False,
        ap01_method_contract="main_route2_parity_v1",
    )

    assert prepared.config.methods.ap01.method_contract == (
        "main_route2_parity_v1"
    )
    saved = load_config(prepared.queue.entries[0].config)
    assert saved.methods.ap01.method_contract == "main_route2_parity_v1"


def test_ap01_contract_override_rejects_other_methods(
    prepared_config,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="valid only"):
        rerun_module._resolved_rerun_config(
            tmp_path,
            tmp_path,
            "ap02",
            "baseline",
            ap01_method_contract="main_route2_parity_v1",
        )


def test_method_rerun_reuses_frozen_detector_provenance(
    prepared_config,
    tmp_path: Path,
) -> None:
    transaction = tmp_path / "transaction"
    shared = transaction / "dataset"
    shared.mkdir(parents=True)
    _write_frozen_observations(
        shared,
        dictionary=prepared_config.markers.dictionary,
        length_m=prepared_config.markers.length_m,
        detection_mode=prepared_config.markers.detection_mode,
    )
    detection_path = shared / "observations/detection_config.json"
    original = detection_path.read_bytes()
    assert observation_id(prepared_config) != "detection_legacy_opencv"

    orchestrator = PipelineOrchestrator(
        tmp_path,
        transaction_root=transaction,
        rerun_metadata={
            "reuse_frozen_observations": True,
            "frozen_observation_contract": {
                "observation_id": "detection_legacy_opencv"
            },
        },
    )
    run = transaction / "jobs/method"
    run.mkdir(parents=True)
    orchestrator.run_directory = run
    orchestrator.manifest = {}
    bound = orchestrator._bind_observations_view(
        prepared_config, "input_fixture"
    )

    assert bound == shared / "observations"
    assert detection_path.read_bytes() == original
    assert orchestrator.manifest["observation_id"] == (
        "detection_legacy_opencv"
    )
    assert orchestrator.manifest["frozen_observations_reused"] is True
    assert PipelineOrchestrator._observation_contract_ready(
        bound, "detection_legacy_opencv"
    )


def test_method_rerun_rejects_changed_marker_contract(
    prepared_config,
    tmp_path: Path,
) -> None:
    transaction = tmp_path / "transaction"
    shared = transaction / "dataset"
    shared.mkdir(parents=True)
    _write_frozen_observations(
        shared,
        dictionary=prepared_config.markers.dictionary,
        length_m=prepared_config.markers.length_m,
        detection_mode=prepared_config.markers.detection_mode,
    )
    changed = prepared_config.model_copy(
        update={
            "markers": prepared_config.markers.model_copy(
                update={"detection_mode": "high_sensitivity"}
            )
        },
        deep=True,
    )
    orchestrator = PipelineOrchestrator(
        tmp_path,
        transaction_root=transaction,
        rerun_metadata={
            "reuse_frozen_observations": True,
            "frozen_observation_contract": {
                "observation_id": "detection_legacy_opencv"
            },
        },
    )
    orchestrator.run_directory = transaction / "jobs/method"
    orchestrator.run_directory.mkdir(parents=True)
    orchestrator.manifest = {}
    with pytest.raises(RuntimeError, match="different marker"):
        orchestrator._bind_observations_view(changed, "input_fixture")
