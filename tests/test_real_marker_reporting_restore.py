from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from camera_rig_calibration.legacy_preferred_anchor_repair import (
    _primary_method_roots,
)
from camera_rig_calibration.real_marker_reporting_policy import (
    _authoritative_anchor,
    _run_authoritative_marker_consistency,
)


def test_legacy_result_discovery_uses_public_ap03_multi_layout(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    for method, label in (
        ("ap01", "root_cam_edge_3"),
        ("ap02", "ref_marker_0__ref_mode_auto"),
        ("ap03_multi", "multi_markers_auto__single_marker_0"),
    ):
        root = experiment / "methods" / method / label
        root.mkdir(parents=True)
        (root / "camera_extrinsics.csv").write_text("entity_id,x_m\n", encoding="utf-8")
        (root / "provenance").mkdir()
        (root / "provenance" / "resolved_config.yaml").write_text(
            "schema_version: 5\n", encoding="utf-8"
        )

    roots = _primary_method_roots(experiment)
    assert [method for method, _ in roots] == ["ap01", "ap02", "ap03_multi"]
    assert roots[2][1].name == "multi_markers_auto__single_marker_0"


def test_authoritative_anchor_supersedes_preserved_preflight_selection(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    dataset = experiment
    (experiment / "evaluations").mkdir(parents=True)
    (experiment / "observations").mkdir(parents=True)
    (experiment / "evaluations" / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps({"anchor_marker_id": 0}), encoding="utf-8"
    )
    (experiment / "observations" / "SELECTION_CANDIDATES.json").write_text(
        json.dumps({"evaluation_anchor": {"selected": 2}}), encoding="utf-8"
    )
    assert _authoritative_anchor(experiment, dataset) == 0


def test_real_marker_consistency_invokes_evaluator_with_authoritative_anchor(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = tmp_path / "experiment"
    dataset = experiment
    (experiment / "evaluations").mkdir(parents=True)
    (experiment / "observations").mkdir(parents=True)
    (experiment / "evaluations" / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps({"anchor_marker_id": 0}), encoding="utf-8"
    )
    (experiment / "observations" / "SELECTION_CANDIDATES.json").write_text(
        json.dumps({"evaluation_anchor": {"selected": 2}}), encoding="utf-8"
    )
    (experiment / "dataset.json").write_text(
        json.dumps(
            {
                "category": "real_vehicle",
                "static_cameras": [{"id": "cam_edge_0"}],
            }
        ),
        encoding="utf-8",
    )
    method = experiment / "methods" / "ap02" / "ref_marker_0__ref_mode_auto"
    (method / "diagnostics" / "method").mkdir(parents=True)
    (method / "provenance").mkdir(parents=True)
    (method / "RESULT.json").write_text(
        json.dumps(
            {
                "method": "ap02",
                "label": method.name,
                "config_summary": {"reference_marker_id": 0},
            }
        ),
        encoding="utf-8",
    )
    (method / "provenance" / "resolved_config.yaml").write_text(
        "markers:\n  length_m: 0.17\n", encoding="utf-8"
    )

    def fake_run(command, **kwargs):
        anchor_index = command.index("--anchor-marker-id") + 1
        assert command[anchor_index] == "0"
        output_index = command.index("--output-root") + 1
        output = Path(command[output_index])
        output.mkdir(parents=True, exist_ok=True)
        (output / "REAL_DATA_MARKER_CONSISTENCY.txt").write_text(
            "Metric anchor: marker 0 = 17.00 cm\nCross RMSE [px]\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    report = _run_authoritative_marker_consistency(experiment, dataset)
    assert report is not None
    assert report.is_file()
    status = json.loads(
        (
            experiment
            / "evaluations"
            / "method_anchors_reconciled"
            / "COMMON_ANCHOR_STATUS.json"
        ).read_text(encoding="utf-8")
    )
    assert status["anchor_marker_id"] == 0
    assert status["ground_truth_used"] is False
