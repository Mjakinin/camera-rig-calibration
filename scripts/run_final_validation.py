#!/usr/bin/env python3
"""Create and optionally run the final calibration validation matrix.

This script is intentionally a thin orchestration layer around the normal
schema-v5 rigcal queue/batch machinery. It never calls a calibration method
implementation directly and never consumes simulation ground truth while
building a method configuration.

Default study (recommended before feature freeze):

* main_route2_reference / AP01 historical semantics:
  baseline_v1, root cam_edge_3, direct target cam_edge_1.
* main_route2_reference / AP01 robust direct-first:
  recommended_wizard_v1, automatic root, Direct for every non-root camera with
  usable shared-marker support, Relay fallback otherwise. A sparse Direct path
  is allowed with two independent agreeing marker estimates; the regular
  translation/rotation dispersion gates remain active.
* route2 / the same two AP01 variants.

All validation variants pin the common evaluation/export anchor to marker 14 so
that the comparison frame matches the historical Route-2 study. This does not
change AP02's calibration reference marker; AP02 is independently pinned to
reference marker 14 by its method contract.

Use ``--profile full`` to add the current AP02 and AP03 baselines to each
selected experiment. AP02 is explicitly kept at reference marker 14 and 80/80
maximum function evaluations. AP03 stays on baseline_v1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import RigConfig
from camera_rig_calibration.queueing import save_batch, save_queue
from camera_rig_calibration.rerun import _resolved_rerun_config


EXPERIMENTS = {
    "route2": Path("results/simulation/baseline/route2"),
    "main_route2_reference": Path(
        "results/simulation/baseline/main_route2_reference"
    ),
}


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _project(config: RigConfig, *, label: str) -> RigConfig:
    """Apply validation-wide publication settings without changing calibration."""
    evaluation = config.evaluation.model_copy(
        update={
            "enabled": True,
            "anchor_marker_id": 14,
            "anchor_selection_mode": "explicit",
        },
        deep=True,
    )
    return RigConfig.model_validate(
        config.model_copy(
            update={
                "project": config.project.model_copy(
                    update={
                        "run_label": label,
                        "execution_mode": "complete",
                        "duplicate_policy": "force",
                    },
                    deep=True,
                ),
                "evaluation": evaluation,
            },
            deep=True,
        ).model_dump(mode="python")
    )


def historical_ap01(repository: Path, experiment: Path) -> RigConfig:
    _, config = _resolved_rerun_config(
        repository,
        experiment,
        "ap01",
        "baseline",
        ap01_method_contract="baseline_v1",
    )
    camera_ids = {camera.id for camera in config.static_cameras}
    required = {"cam_edge_3", "cam_edge_1"}
    if not required.issubset(camera_ids):
        raise RuntimeError(
            "The historical AP01 comparison requires cam_edge_3 as root and "
            "cam_edge_1 as its legacy Direct target. Available cameras: "
            + ", ".join(sorted(camera_ids))
        )
    ap01 = config.methods.ap01.model_copy(
        update={
            "method_contract": "baseline_v1",
            "historical_reproduction": False,
            "advanced_strategy": "legacy_main_v1",
            "root_camera": "cam_edge_3",
            "direct_target_camera": "cam_edge_1",
        },
        deep=True,
    )
    methods = config.methods.model_copy(
        update={"enabled": ["ap01"], "ap01": ap01},
        deep=True,
    )
    configured = RigConfig.model_validate(
        config.model_copy(
            update={
                "methods": methods,
                "selection": config.selection.model_copy(
                    update={"mode": "explicit"}, deep=True
                ),
            },
            deep=True,
        ).model_dump(mode="python")
    )
    return _project(configured, label="ap01_historical_root_cam3")


def robust_ap01(repository: Path, experiment: Path) -> RigConfig:
    _, config = _resolved_rerun_config(
        repository,
        experiment,
        "ap01",
        "baseline",
        ap01_method_contract="recommended_wizard_v1",
    )
    direct_gate = config.methods.ap01.direct_quality_gate.model_copy(
        update={
            # A pair such as cam_edge_1 <-> cam_edge_3 can legitimately have
            # only a small number of shared static markers. Two independent
            # agreeing marker transforms are sufficient for this validation;
            # the existing MAD/dispersion and path-consistency checks still
            # reject geometrically inconsistent Direct estimates.
            "minimum_independent_markers": 2,
            # With sparse overlap, a small high-quality consensus must not be
            # rejected solely because additional weak common markers became
            # outliers. Two agreeing inliers among six historical candidates
            # corresponds to 1/3 support, so 0.30 is the conservative floor.
            "minimum_inlier_ratio": 0.30,
        },
        deep=True,
    )
    ap01 = config.methods.ap01.model_copy(
        update={
            "method_contract": "recommended_wizard_v1",
            "historical_reproduction": False,
            "advanced_strategy": "wizard_robustness_v1",
            "root_camera": "auto",
            "direct_quality_gate": direct_gate,
        },
        deep=True,
    )
    methods = config.methods.model_copy(
        update={"enabled": ["ap01"], "ap01": ap01},
        deep=True,
    )
    configured = RigConfig.model_validate(
        config.model_copy(
            update={
                "methods": methods,
                # Auto is intentional here. Unlike baseline_v1, the robust
                # contract creates Direct candidates for every non-root camera,
                # so the selected root cannot accidentally consume the sole
                # Direct target. Shared-marker pairs such as cam1<->cam3 are
                # therefore eligible for Direct calibration.
                "selection": config.selection.model_copy(
                    update={"mode": "auto"}, deep=True
                ),
            },
            deep=True,
        ).model_dump(mode="python")
    )
    return _project(
        configured,
        label="ap01_robust_direct_first_sparse2_auto_root",
    )


def baseline_ap02(repository: Path, experiment: Path) -> RigConfig:
    _, config = _resolved_rerun_config(
        repository,
        experiment,
        "ap02",
        "baseline",
        ap02_historical_reproduction=False,
    )
    ap02 = config.methods.ap02.model_copy(
        update={
            "method_contract": "baseline_v1",
            "historical_reproduction": False,
            "reference_marker_selection_mode": "baseline",
            "reference_marker_id": 14,
            "frame_selection_strategy": "legacy_smart_v1",
            "initialization_strategy": "legacy_maximum_bottleneck_v1",
            "graph_edge_weight_strategy": "legacy_observation_quality_v1",
            "reprojection_model": "legacy_pinhole_v1",
            "reference_marker_maximum_frames": None,
            "top_per_marker": 8,
            "top_per_marker_pair": 4,
            "maximum_total_frames": None,
            "static_only_ba_max_function_evaluations": 80,
            "combined_ba_max_function_evaluations": 80,
            "ba_robust_loss": "soft_l1",
            "ba_robust_loss_scale_px": 3.0,
        },
        deep=True,
    )
    methods = config.methods.model_copy(
        update={"enabled": ["ap02"], "ap02": ap02},
        deep=True,
    )
    configured = RigConfig.model_validate(
        config.model_copy(update={"methods": methods}, deep=True).model_dump(
            mode="python"
        )
    )
    return _project(configured, label="ap02_baseline_ref14")


def baseline_ap03(repository: Path, experiment: Path) -> RigConfig:
    _, config = _resolved_rerun_config(
        repository,
        experiment,
        "ap03",
        "baseline",
        ap03_method_contract="baseline_v1",
    )
    methods = config.methods.model_copy(
        update={"enabled": ["ap03"]},
        deep=True,
    )
    configured = RigConfig.model_validate(
        config.model_copy(update={"methods": methods}, deep=True).model_dump(
            mode="python"
        )
    )
    return _project(configured, label="ap03_baseline_v1")


def build_configs(
    repository: Path,
    experiment: Path,
    *,
    profile: str,
) -> list[tuple[str, RigConfig]]:
    configs = [
        ("ap01_historical_root_cam3", historical_ap01(repository, experiment)),
        (
            "ap01_robust_direct_first_sparse2_auto_root",
            robust_ap01(repository, experiment),
        ),
    ]
    if profile == "full":
        configs.extend(
            [
                ("ap02_baseline_ref14", baseline_ap02(repository, experiment)),
                ("ap03_baseline_v1", baseline_ap03(repository, experiment)),
            ]
        )
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare/run the final AP01 study or full AP01/AP02/AP03 "
            "validation batch."
        )
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=tuple(EXPERIMENTS),
        help=(
            "Experiment to validate. Repeat for multiple experiments. Default: "
            "route2 and main_route2_reference."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("ap01-study", "full"),
        default="ap01-study",
        help=(
            "ap01-study = historical + robust AP01 only (recommended now); "
            "full = those two plus AP02 baseline and AP03 baseline."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run",
        action="store_true",
        help="Execute the generated batch immediately without UI prompts.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Run rigcal validation/dry-run for the generated batch.",
    )
    args = parser.parse_args()

    repository = repository_root()
    names = args.experiment or ["main_route2_reference", "route2"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = repository / "workspace" / "final_validation" / stamp
    queues: list[tuple[str, Path]] = []

    for name in names:
        experiment = (repository / EXPERIMENTS[name]).resolve()
        if not (experiment / "dataset.json").is_file():
            raise FileNotFoundError(
                f"Prepared experiment is missing dataset.json: {experiment}"
            )
        config_entries: list[tuple[str, Path]] = []
        experiment_dir = root / name
        for entry_id, config in build_configs(
            repository,
            experiment,
            profile=args.profile,
        ):
            path = experiment_dir / "configs" / f"{entry_id}.yaml"
            save_config(config, path)
            config_entries.append((entry_id, path))
        queue_path = experiment_dir / "queue.yaml"
        save_queue(
            f"final_validation_{name}_{stamp}",
            config_entries,
            queue_path,
        )
        queues.append((name, queue_path))

    batch = root / "batch.yaml"
    save_batch(f"final_validation_{stamp}", queues, batch)

    print("[OK] Final validation configuration prepared")
    print(f"[OK] profile: {args.profile}")
    print(f"[OK] experiments: {', '.join(names)}")
    print("[OK] common evaluation anchor: marker 14")
    print("[OK] robust AP01 sparse Direct gate: >=2 inlier markers, ratio >=0.30")
    print(f"[OK] batch: {batch}")
    print()
    print("Run without the Wizard:")
    print(f"  rigcal --config {batch} --yes")
    print("Validate only:")
    print(f"  rigcal --config {batch} --dry-run")

    if args.run or args.dry_run:
        command = [
            sys.executable,
            "-m",
            "camera_rig_calibration.cli",
            "--config",
            str(batch),
        ]
        command.append("--dry-run" if args.dry_run else "--yes")
        print()
        print("[RUN] " + " ".join(command))
        subprocess.run(command, cwd=repository, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
