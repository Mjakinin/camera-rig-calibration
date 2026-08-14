from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from camera_rig_calibration.methods.ap02 import optimize_core
from camera_rig_calibration.methods.ap02.build_graph import run as build_graph
from camera_rig_calibration.methods.ap02.contracts import resolve_ap02_method_contract
from camera_rig_calibration.methods.ap02.frame_selection import (
    select_legacy_smart_moving_observations,
)
from camera_rig_calibration.methods.ap02.frozen_observations import (
    FROZEN_AP02_MANIFEST,
    resolve_ap02_observation_input,
    validate_frozen_ap02_observations,
)
from camera_rig_calibration.methods.ap02.initialize import (
    best_observations,
    build_graph as build_initialization_graph,
    initialize_from_tree,
    main_compat_widest_path_tree,
    main_observation_score,
    marker_node,
)
from camera_rig_calibration.methods.ap02.optimize import (
    validate_historical_pre_solver,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY / "results/simulation/baseline/route2_main_parity_v1"
FROZEN = (
    REPOSITORY
    / "parity/main_route2_v1/frozen/ap02_all_aruco_observations.csv"
)


def _contract():
    return resolve_ap02_method_contract()


def _validate(**overrides):
    contract = _contract()
    arguments = {
        "dataset": DATASET,
        "historical_reproduction": True,
        "method_contract_name": contract.name,
        "method_contract_sha256": contract.scientific_fingerprint(),
        "reference_marker_policy": contract.reference_marker_policy,
        "reference_marker_id": 14,
        "root_pose_policy": contract.root_pose_policy,
    }
    arguments.update(overrides)
    return validate_frozen_ap02_observations(**arguments)


def test_frozen_ap02_stream_requires_explicit_opt_in() -> None:
    with pytest.raises(RuntimeError, match="explicit historical reproduction"):
        _validate(historical_reproduction=False)


def test_normal_baseline_never_resolves_the_frozen_stream(tmp_path: Path) -> None:
    normal, frozen = resolve_ap02_observation_input(
        observations_root=tmp_path,
        dataset=None,
        historical_reproduction=False,
        method_contract_name="wrong-on-purpose",
        method_contract_sha256="wrong-on-purpose",
        reference_marker_policy="wrong-on-purpose",
        reference_marker_id=-1,
        root_pose_policy="wrong-on-purpose",
    )
    assert normal == tmp_path / "shared_all_aruco_observations.csv"
    assert frozen is None


def test_frozen_ap02_validation_is_strict_and_fail_closed(
    tmp_path: Path,
) -> None:
    if not (DATASET / "metadata/dataset_identity.json").is_file():
        with pytest.raises(
            RuntimeError, match="Prepared AP02 dataset identity is missing"
        ):
            _validate()
        return
    frozen = _validate()
    assert frozen.observations == FROZEN
    assert frozen.provenance["source_observation_count"] == 513
    manifest = json.loads(
        (REPOSITORY / FROZEN_AP02_MANIFEST).read_text(encoding="utf-8")
    )
    manifest["artifact"]["ordering_schema_sha256"] = "0" * 64
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ordering/schema"):
        _validate(manifest_path=bad_manifest, repository_root=REPOSITORY)


def test_historical_build_graph_records_validated_provenance(
    tmp_path: Path,
) -> None:
    contract = _contract()
    output = tmp_path / "03_AP02"
    if not (DATASET / "metadata/dataset_identity.json").is_file():
        with pytest.raises(
            RuntimeError, match="Prepared AP02 dataset identity is missing"
        ):
            build_graph(
                observations_root=DATASET / "observations",
                dataset_root=DATASET,
                output_root=output,
                camera_ids=(
                    "cam_edge_0",
                    "cam_edge_1",
                    "cam_edge_3",
                    "cam_edge_5",
                ),
                reference_marker_id=14,
                reference_marker_maximum_frames=None,
                top_per_marker=8,
                top_per_marker_pair=4,
                maximum_total_frames=None,
                graph_observation_policy=contract.graph_observation_policy,
                method_contract=contract.fingerprint_payload(),
                method_contract_sha256=contract.scientific_fingerprint(),
                historical_reproduction=True,
            )
        return
    result = build_graph(
        observations_root=DATASET / "observations",
        dataset_root=DATASET,
        output_root=output,
        camera_ids=("cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"),
        reference_marker_id=14,
        reference_marker_maximum_frames=None,
        top_per_marker=8,
        top_per_marker_pair=4,
        maximum_total_frames=None,
        graph_observation_policy=contract.graph_observation_policy,
        method_contract=contract.fingerprint_payload(),
        method_contract_sha256=contract.scientific_fingerprint(),
        historical_reproduction=True,
    )
    assert result.status == "COMPLETED"
    provenance = json.loads(
        (
            output
            / "02_aruco_observations"
            / "HISTORICAL_REPRODUCTION_PROVENANCE.json"
        ).read_text(encoding="utf-8")
    )
    assert provenance["validation_status"] == "passed"
    assert provenance["historical_reproduction"] is True
    assert provenance["ground_truth_used"] is False
    assert provenance["source_artifact_sha256"] == (
        "d358b584ef10ecbf3d2c971719718d414a3e80228a15abf10e909620cbbe1071"
    )


def test_frozen_stream_reproduces_and_authorizes_pre_solver_invariants(
    tmp_path: Path,
) -> None:
    with FROZEN.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = best_observations(
        rows, edge_weight_policy="legacy_observation_quality_v1"
    )
    adjacency = build_initialization_graph(selected, preserve_input_order=True)
    parent, metrics = main_compat_widest_path_tree(
        adjacency,
        marker_node(14),
        edge_weight_policy="legacy_observation_quality_v1",
    )
    marker_poses, observer_poses, _, _ = initialize_from_tree(
        parent,
        14,
        path_metrics=metrics,
        algorithm="legacy_maximum_bottleneck_v1",
        edge_weight_policy="legacy_observation_quality_v1",
    )
    available = optimize_core.filter_observations(
        rows, "with_moving", marker_poses, observer_poses
    )
    static_rows = [
        row for row in available if row["observer_type"] == "static"
    ]
    selection = select_legacy_smart_moving_observations(
        [row for row in available if row["observer_type"] == "moving"],
        reference_marker_id=14,
        reference_marker_maximum_frames=None,
        top_per_marker=8,
        top_per_marker_pair=4,
        maximum_total_frames=None,
        observation_score=main_observation_score,
    )
    observations = [*static_rows, *selection.selected_rows]
    used_observers = {row["observer_id"] for row in observations}
    used_markers = {int(float(row["marker_id"])) for row in observations}
    x0, names = optimize_core.pack_params(
        {
            key: value
            for key, value in marker_poses.items()
            if key in used_markers or key == 14
        },
        {
            key: value
            for key, value in observer_poses.items()
            if key in used_observers
        },
        14,
    )
    residuals = optimize_core.make_residual_function(
        observations, names, 14
    )(x0)
    ap02_root = tmp_path / "03_AP02"
    provenance = ap02_root / "02_aruco_observations"
    provenance.mkdir(parents=True)
    (provenance / "HISTORICAL_REPRODUCTION_PROVENANCE.json").write_text(
        json.dumps(
            {
                "historical_reproduction": True,
                "ground_truth_used": False,
                "validation_status": "passed",
                "reference_marker_id": 14,
            }
        ),
        encoding="utf-8",
    )
    initialization = tmp_path / "initialization" / "with_moving"
    initialization.mkdir(parents=True)
    moving = sorted(
        key for key in observer_poses if key.startswith("moving_frame_")
    )
    with (
        initialization / "initial_moving_frame_poses_ref_marker.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["entity_id"])
        writer.writeheader()
        writer.writerows({"entity_id": key} for key in moving)
    evidence = validate_historical_pre_solver(
        ap02_root=ap02_root,
        initialization_root=tmp_path / "initialization",
        reference_marker_id=14,
        maximum_function_evaluations=80,
        robust_loss="soft_l1",
        robust_loss_scale_px=3.0,
        initial_parameters=x0,
        initial_residuals=residuals,
    )
    assert evidence["solver_authorized"] is True
    assert evidence["actual"]["initialized_moving_frames"] == 170
    assert evidence["actual"]["ba_observations"] == 458
    assert evidence["actual"]["variable_poses"] == 160
    assert evidence["actual"]["parameter_count"] == 960
    assert evidence["actual"]["initial_mean_reprojection_px"] == pytest.approx(
        10.2476446393, abs=1e-10
    )
