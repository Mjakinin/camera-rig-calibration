from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Importing the public product entrypoint installs the same publication policies
# used by rigcal itself. It does not execute the CLI.
import camera_rig_calibration.product_cli  # noqa: F401,E402
from camera_rig_calibration.anchor_export import ensure_experiment_anchor_exports  # noqa: E402
from camera_rig_calibration.evaluation.reporting import (  # noqa: E402
    write_scientific_experiment_reports,
)
from camera_rig_calibration.visualization.scene import (  # noqa: E402
    ensure_visualization_artifacts,
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _category(root: Path) -> str:
    payload = _read_json(root / "dataset.json")
    return str(
        payload.get("category")
        or payload.get("dataset", {}).get("category")
        or "real_vehicle"
    )


def _invalidate_derived_anchor_outputs(root: Path) -> int:
    removed = 0
    for variant in sorted((root / "methods").glob("*/*")):
        if not variant.is_dir():
            continue
        for relative in (
            "camera_extrinsics_anchor.json",
            "camera_extrinsics_anchor.yaml",
            "camera_extrinsics_anchor.csv",
            "diagnostics/anchor_alignment.json",
        ):
            path = variant / relative
            if path.is_file():
                path.unlink()
                removed += 1
    for name in (
        "CAMERA_EXTRINSICS_COMMON_ANCHOR.json",
        "CAMERA_EXTRINSICS_COMMON_ANCHOR.yaml",
        "CAMERA_EXTRINSICS_COMMON_ANCHOR.csv",
    ):
        path = root / name
        if path.is_file():
            path.unlink()
            removed += 1
    visualization = root / "visualization"
    if visualization.is_dir():
        shutil.rmtree(visualization)
        removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild only derived anchor exports, scientific reports and RViz "
            "artifacts for a completed experiment. Calibration methods and "
            "COLMAP are never rerun."
        )
    )
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.expanduser().resolve()
    if not (root / "methods").is_dir():
        raise SystemExit(f"No published methods directory: {root / 'methods'}")
    if not (root / "observations").is_dir():
        raise SystemExit(f"No published observations directory: {root / 'observations'}")

    removed = _invalidate_derived_anchor_outputs(root)
    anchor_status = ensure_experiment_anchor_exports(root)
    report = write_scientific_experiment_reports(
        root,
        dataset_root=root,
        category=_category(root),
    )
    visualization = ensure_visualization_artifacts(root)

    print("[OK] Derived outputs rebuilt")
    print(f"  experiment: {root}")
    print(f"  invalidated derived artifacts: {removed}")
    print("  method rerun: False")
    print("  COLMAP rerun: False")
    print("  native camera extrinsics modified: False")
    print(f"  anchor exports: {anchor_status}")
    print(
        "  common anchor: "
        f"{report.get('evaluation_anchor', {}).get('selected', 'unknown')}"
    )
    print(f"  RESULTS: {root / 'RESULTS.txt'}")
    print(
        "  RViz fixed frame: "
        f"{visualization.get('fixed_frame', 'unavailable')}"
    )


if __name__ == "__main__":
    main()
