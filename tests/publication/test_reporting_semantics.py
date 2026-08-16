from __future__ import annotations

from pathlib import Path

from camera_rig_calibration.evaluation.reporting_semantics import (
    configuration_summary,
)


def test_ap02_summary_reports_productive_initialization_strategy(
    tmp_path: Path,
) -> None:
    provenance = tmp_path / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "resolved_config.yaml").write_text(
        """
methods:
  ap02:
    method_contract: baseline_v1
    reference_marker_selection_mode: baseline
    reference_marker_id: 14
    initialization_strategy: maximum_frontier_v1
    static_only_ba_max_function_evaluations: 80
    combined_ba_max_function_evaluations: 80
    ba_robust_loss: soft_l1
    ba_robust_loss_scale_px: 3.0
markers:
  detection_mode: baseline
evaluation:
  anchor_marker_id: 14
""".strip()
        + "\n",
        encoding="utf-8",
    )

    summary = configuration_summary(tmp_path, "ap02")

    assert summary["initialization_strategy"] == "maximum_frontier_v1"
    assert "initialization_algorithm" not in summary
    assert "initialization_diagnostic" not in summary


def test_ap02_summary_preserves_explicit_nondefault_strategy(
    tmp_path: Path,
) -> None:
    provenance = tmp_path / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "resolved_config.yaml").write_text(
        """
methods:
  ap02:
    initialization_strategy: wizard_maximum_bottleneck_v2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    summary = configuration_summary(tmp_path, "ap02")

    assert summary["initialization_strategy"] == "wizard_maximum_bottleneck_v2"
