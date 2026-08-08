from __future__ import annotations

import argparse
import json
from pathlib import Path

from camera_rig_calibration.common_anchor_authority_policy import (
    install_common_anchor_authority_policy,
    repair_published_preferred_anchor,
)
from camera_rig_calibration.legacy_preferred_anchor_repair import (
    repair_legacy_preferred_anchor,
)
from camera_rig_calibration.marker_preference_policy import (
    install_marker_preference_policy,
)
from camera_rig_calibration.product_policy import install_product_policy
from camera_rig_calibration.queue_anchor_preference_policy import (
    install_queue_anchor_preference_policy,
)
from camera_rig_calibration.reanchor_existing_results_policy import (
    install_reanchor_existing_results_policy,
)
from camera_rig_calibration.real_marker_reporting_policy import (
    install_real_marker_reporting_policy,
)
from camera_rig_calibration.reporting_authority_policy import (
    install_reporting_authority_policy,
)
from camera_rig_calibration.result_output_policy import install_result_output_policy
from camera_rig_calibration.rviz_manifest_policy import install_rviz_manifest_policy
from camera_rig_calibration.submission_bindings import install_submission_bindings
from camera_rig_calibration.submission_policy import install_submission_policy
from camera_rig_calibration.ui_display_policy import install_ui_display_policy


install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
install_common_anchor_authority_policy()
install_queue_anchor_preference_policy()
install_reanchor_existing_results_policy()
install_result_output_policy()
install_real_marker_reporting_policy()
install_rviz_manifest_policy()
install_submission_bindings()
install_ui_display_policy()

from camera_rig_calibration.evaluation.reporting import (  # noqa: E402
    write_scientific_experiment_reports,
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _category(dataset_root: Path) -> str:
    dataset = _read_json(dataset_root / "dataset.json")
    value = dataset.get("category") or dataset.get("dataset", {}).get("category")
    return str(value or "real_vehicle")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh only derived common-anchor exports, RESULTS and RViz artifacts. "
            "Calibration methods and COLMAP are never rerun."
        )
    )
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Canonical dataset root; defaults to the experiment root.",
    )
    args = parser.parse_args()
    experiment_root = args.experiment_root.expanduser().resolve()
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else experiment_root
    )
    if not (experiment_root / "methods").is_dir():
        raise SystemExit(f"No published methods directory: {experiment_root / 'methods'}")
    if not (dataset_root / "observations").is_dir():
        raise SystemExit(f"No canonical observations directory: {dataset_root / 'observations'}")

    before = _read_json(
        experiment_root / "observations" / "SELECTION_CANDIDATES.json"
    ).get("evaluation_anchor", {}).get("selected")
    repair = repair_published_preferred_anchor(experiment_root)
    if repair.get("status") == "not_applicable":
        repair = repair_legacy_preferred_anchor(experiment_root)

    payload = write_scientific_experiment_reports(
        experiment_root,
        dataset_root=dataset_root,
        category=_category(dataset_root),
    )
    authoritative = _read_json(
        experiment_root / "evaluations" / "SELECTED_COMMON_EVALUATION.json"
    ).get("anchor_marker_id")
    if authoritative is None:
        authoritative = payload.get("evaluation_anchor", {}).get("selected")

    print("[OK] Derived output refresh complete")
    print(f"  experiment: {experiment_root}")
    print(f"  preflight anchor preserved: {before}")
    print(f"  authoritative common anchor: {authoritative}")
    print(f"  anchor repair status: {repair.get('status')}")
    print("  method rerun: False")
    print("  COLMAP rerun: False")
    print("  native method outputs modified: False")
    marker_status = _read_json(
        experiment_root
        / "evaluations"
        / "method_anchors_reconciled"
        / "COMMON_ANCHOR_STATUS.json"
    )
    print(f"  marker consistency: {marker_status.get('status', 'unavailable')}")
    print(f"  results: {experiment_root / 'RESULTS.txt'}")
    print(
        "  6DoF YAML: "
        f"{experiment_root / 'CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml'}"
    )
    print(
        "  RViz manifest: "
        f"{experiment_root / 'visualization' / 'visualization_manifest.json'}"
    )


if __name__ == "__main__":
    main()
