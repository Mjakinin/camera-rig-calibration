from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.policies.result_view_policy import install_result_view_policy


install_result_view_policy()

from camera_rig_calibration.ui import result_browser  # noqa: E402


def test_completed_result_view_skips_expensive_reconciliation(tmp_path: Path) -> None:
    experiment = tmp_path / "results" / "real_vehicle" / "native_rate" / "IMG_4319"
    experiment.mkdir(parents=True)
    (experiment / "RESULTS.txt").write_text("done\n", encoding="utf-8")
    (experiment / "COMPARISON.json").write_text(
        json.dumps({"status": "done"}), encoding="utf-8"
    )

    assert getattr(
        result_browser.reconcile_existing_experiment,
        "_rigcal_fast_result_view",
        False,
    )
    payload = result_browser.reconcile_existing_experiment(
        experiment,
        dataset_root=experiment,
        category="real_vehicle",
    )

    assert payload["status"] == "already_materialized"
    assert payload["reason"] == "completed published result opened read-only"
