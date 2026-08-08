from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from camera_rig_calibration.anchor_export.adapters import AnchorResolution
from camera_rig_calibration.config.models import MethodSettings, RigConfig
from camera_rig_calibration.legacy_preferred_anchor_repair import (
    repair_legacy_preferred_anchor,
)
from camera_rig_calibration.rviz_manifest_policy import _synchronize_manifest


def _real_vehicle_preference_config() -> RigConfig:
    methods = MethodSettings(enabled=["ap01", "ap02", "ap03"])
    methods = methods.model_copy(
        update={
            "ap02": methods.ap02.model_copy(
                update={
                    "reference_marker_selection_mode": "auto",
                    "reference_marker_id": 0,
                }
            ),
            "ap03": methods.ap03.model_copy(
                update={
                    "single": methods.ap03.single.model_copy(
                        update={"scale_marker_id": 0}
                    )
                },
                deep=True,
            ),
        },
        deep=True,
    )
    return RigConfig(methods=methods)


def _write_legacy_selection(experiment: Path, *, mode: str = "auto") -> None:
    observations = experiment / "observations"
    observations.mkdir(parents=True)
    (observations / "SELECTION_CANDIDATES.json").write_text(
        json.dumps(
            {
                "evaluation_anchor": {
                    "selected": 2,
                    "configured": 2,
                    "selection_mode": mode,
                }
            }
        ),
        encoding="utf-8",
    )
    (experiment / "dataset.json").write_text(
        json.dumps({"category": "real_vehicle"}), encoding="utf-8"
    )


def test_legacy_real_vehicle_anchor_zero_is_repaired_only_after_all_method_checks(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = tmp_path / "experiment"
    _write_legacy_selection(experiment)
    roots = [
        ("ap01", experiment / "methods" / "ap01" / "root_cam_edge_3"),
        ("ap02", experiment / "methods" / "ap02" / "ref_marker_0__ref_mode_auto"),
        ("ap03", experiment / "methods" / "ap03" / "multi__single_marker_0"),
    ]
    for _, root in roots:
        root.mkdir(parents=True)

    config = _real_vehicle_preference_config()
    monkeypatch.setattr(
        "camera_rig_calibration.legacy_preferred_anchor_repair._primary_method_roots",
        lambda _: roots,
    )

    from camera_rig_calibration.anchor_export import adapters, exporter

    monkeypatch.setattr(exporter, "_config_for_result", lambda _: config)
    monkeypatch.setattr(
        adapters,
        "load_camera_poses",
        lambda _: {"cam_edge_0": np.eye(4, dtype=np.float64)},
    )

    def resolve(method_root, config_value, method, anchor, native):
        assert anchor == 0
        return AnchorResolution(
            np.eye(4, dtype=np.float64),
            "OK",
            True,
            (),
            {"method": method, "ground_truth_used": False},
        )

    monkeypatch.setattr(adapters, "resolve_method_anchor", resolve)
    result = repair_legacy_preferred_anchor(experiment)
    assert result["status"] == "repaired_legacy_category_preference"
    assert result["published_anchor_marker_id"] == 0
    selected = json.loads(
        (experiment / "evaluations" / "SELECTED_COMMON_EVALUATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected["anchor_marker_id"] == 0
    assert selected["superseded_preflight_anchor_marker_id"] == 2
    assert selected["ground_truth_used"] is False


def test_legacy_repair_never_overrides_manual_anchor(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    _write_legacy_selection(experiment, mode="manual")
    result = repair_legacy_preferred_anchor(experiment)
    assert result["status"] == "not_applicable"
    assert not (
        experiment / "evaluations" / "SELECTED_COMMON_EVALUATION.json"
    ).exists()


def test_rviz_manifest_defaults_match_primary_method_visibility(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    output = experiment / "visualization"
    output.mkdir(parents=True)
    manifest = {
        "available": True,
        "variants": [
            {
                "method": "ap01",
                "label": "root_cam_edge_3",
                "default_visible": False,
                "anchor_edges_default_visible": False,
            },
            {
                "method": "ap02",
                "label": "ref_marker_0__ref_mode_auto",
                "default_visible": False,
                "anchor_edges_default_visible": False,
            },
            {
                "method": "ap03_multi",
                "label": "multi__single_marker_0",
                "default_visible": True,
                "anchor_edges_default_visible": False,
            },
            {
                "method": "ap03_single",
                "label": "multi__single_marker_0",
                "default_visible": False,
                "anchor_edges_default_visible": False,
            },
        ],
    }
    updated = _synchronize_manifest(experiment, manifest)
    by_method = {item["method"]: item for item in updated["variants"]}
    for method in ("ap01", "ap02", "ap03_multi"):
        assert by_method[method]["default_visible"] is True
        assert by_method[method]["anchor_edges_default_visible"] is True
    assert by_method["ap03_single"]["default_visible"] is False
    assert by_method["ap03_single"]["anchor_edges_default_visible"] is False
