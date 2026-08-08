#!/usr/bin/env python3
"""Preview AP01 automatic Direct-target selection on a published experiment.

This is a read-only scientific-policy check. It reuses the already published,
quality-filtered observation table and resolved AP01 config; it does not invoke
COLMAP, calibration methods, evaluation, publication, or ground truth.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from camera_rig_calibration.ap01_auto_direct import automatic_ap01_direct_target
from camera_rig_calibration.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        type=Path,
        help="Published experiment root, e.g. results/simulation/baseline/route2",
    )
    args = parser.parse_args()
    experiment = args.experiment.resolve()

    config_path = (
        experiment
        / "methods"
        / "ap01"
        / "baseline"
        / "provenance"
        / "resolved_config.yaml"
    )
    accepted = experiment / "observations" / "quality" / "accepted_observations.csv"
    if not config_path.is_file():
        raise FileNotFoundError(f"AP01 baseline resolved config is missing: {config_path}")
    if not accepted.is_file():
        raise FileNotFoundError(f"Published accepted observations are missing: {accepted}")

    config = load_config(config_path)
    root = str(config.methods.ap01.root_camera)
    if root == "auto":
        raise RuntimeError(
            "Published AP01 baseline still has an unresolved root; reconcile/preflight first"
        )
    ap01 = config.methods.ap01.model_copy(update={"direct_target_camera": "auto"})
    config = config.model_copy(
        update={
            "methods": config.methods.model_copy(update={"ap01": ap01}, deep=True)
        },
        deep=True,
    )

    with tempfile.TemporaryDirectory(prefix="rigcal_ap01_direct_check_") as temp:
        observations = Path(temp)
        shutil.copy2(accepted, observations / "shared_all_aruco_observations.csv")
        selected, candidates = automatic_ap01_direct_target(
            config,
            observations,
            root,
        )

    print(f"AP01 root: {root}")
    print(f"Automatic Direct target: {selected or 'none -> Relay-only'}")
    print("Candidates:")
    for item in candidates:
        rmse = item.get("median_pair_pnp_reprojection_rmse_px")
        rmse_text = f"{float(rmse):.3f}px" if rmse is not None else "n/a"
        print(
            "  "
            f"{item['id']}: eligible={item['compatible']}, "
            f"shared={item['independent_shared_markers']}, "
            f"quality={item['quality_filtered_markers']}, "
            f"inliers={item['independent_inlier_markers']}, "
            f"fallback={item['quality_filter_fallback_used']}, "
            f"median_pair_rmse={rmse_text}, "
            f"markers={item['inlier_marker_ids']}"
        )
    print("Ground truth used: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
