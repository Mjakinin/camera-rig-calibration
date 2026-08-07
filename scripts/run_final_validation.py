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

Existing experiments are always rebound as PREPARED input. The script resolves
a local input root that actually contains the published/raw moving frames and
explicitly disables Gazebo capture before saving any validation config.

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
from camera_rig_calibration.config.models import InputSourceKind, RigConfig
from camera_rig_calibration.queueing import save_batch, save_queue
from camera_rig_calibration.rerun import _resolved_rerun_config


EXPERIMENTS = {
    "route2": Path("results/simulation/baseline/route2"),
    "main_route2_reference": Path(
        "results/simulation/baseline/main_route2_reference"
    ),
}

EXPECTED_MOVING_FRAMES = 189
EXPECTED_STATIC_IMAGES = 4


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _input_counts(root: Path) -> tuple[int, int, int]:
    moving = len(list((root / "raw_images" / "moving").glob("*.png")))
    static = len(list((root / "raw_images" / "static").glob("*.png")))
    camera_info = len(
        list((root / "raw_images" / "camera_info").glob("*.json"))
    )
    return moving, static, camera_info


def _candidate_input_roots(
    repository: Path,
    experiment: Path,
    config: RigConfig,
) -> list[Path]:
    candidates: list[Path] = [
        experiment.resolve(),
        (
            repository
            / "results"
            / "simulation"
            / "reference_inputs"
            / experiment.name
        ).resolve(),
    ]
    for configured in (
        config.dataset.prepared_root,
        config.dataset.input_root,
    ):
        if configured is not None:
            candidates.append(Path(configured).resolve())

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_existing_input_root(
    repository: Path,
    experiment: Path,
    config: RigConfig,
) -> Path:
    checked: list[str] = []
    for candidate in _candidate_input_roots(repository, experiment, config):
        moving, static, camera_info = _input_counts(candidate)
        checked.append(
            f"{candidate} [moving={moving}, static={static}, "
            f"camera_info={camera_info}]"
        )
        if (
            moving == EXPECTED_MOVING_FRAMES
            and static >= EXPECTED_STATIC_IMAGES
            and camera_info >= 5
        ):
            return candidate

    raise RuntimeError(
        "Final validation requires an existing complete prepared Route-2 input "
        f"({EXPECTED_MOVING_FRAMES} moving PNGs, at least "
        f"{EXPECTED_STATIC_IMAGES} static PNGs and 5 CameraInfo JSON files). "
        "No complete input root was found. Checked:\n  - "
        + "\n  - ".join(checked)
    )


def _bind_existing_input(
    config: RigConfig,
    prepared_root: Path,
) -> RigConfig:
    """Bind one validation job to existing pixels and forbid Gazebo recapture."""
    root = prepared_root.resolve()
    moving_intrinsics = (
        root
        / "raw_images"
        / "camera_info"
        / f"{config.moving_camera.id}.json"
    )
    dataset = config.dataset.model_copy(
        update={
            "source_kind": InputSourceKind.PREPARED,
            "prepared_root": root,
            "input_root": root,
        },
        deep=True,
    )
    moving = config.moving_camera.model_copy(
        update={
            "video": None,
            "frames": None,
            "intrinsics": (
                moving_intrinsics
                if moving_intrinsics.is_file()
                else config.moving_camera.intrinsics
            ),
            "intrinsic_calibration_video": None,
            "intrinsic_calibration_images": None,
        },
        deep=True,
    )
    simulation = config.simulation.model_copy(
        update={"enabled": False},
        deep=True,
    )
    return RigConfig.model_validate(
        config.model_copy(
            update={
                "dataset": dataset,
                "moving_camera": moving,
                "simulation": simulation,
            },
            deep=True,
        ).model_dump(mode="python")
    )


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


def _base_config(
    repository: Path,
    experiment: Path,
    prepared_root: Path,
    method: str,
    *,
    ap01_method_contract: str | None = None,
    ap02_historical_reproduction: bool = False,
    ap03_method_contract: str | None = None,
) -> RigConfig:
    _, config = _resolved_rerun_config(
        repository,
        experiment,
        method,
        "baseline",
        ap01_method_contract=ap01_method_contract,
        ap02_historical_reproduction=ap02_historical_reproduction,
        ap03_method_contract=ap03_method_contract,
    )
    return _bind_existing_input(config, prepared_root)


def historical_ap01(
    repository: Path,
    experiment: Path,
    prepared_root: Path,
) -> RigConfig:
    config = _base_config(
        repository,
        experiment,
        prepared_root,
        "ap01",
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


def robust_ap01(
    repository: Path,
    experiment: Path,
    prepared_root: Path,
) -> RigConfig:
    config = _base_config(
        repository,
        experiment,
        prepared_root,
        "ap01",
        ap01_method_contract="recommended_wizard_v1",
    )
    direct_gate = config.methods.ap01.direct_quality_gate.model_copy(
        update={
            "minimum_independent_markers": 2,
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


def baseline_ap02(
    repository: Path,
    experiment: Path,
    prepared_root: Path,
) -> RigConfig:
    config = _base_config(
        repository,
        experiment,
        prepared_root,
        "ap02",
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


def baseline_ap03(
    repository: Path,
    experiment: Path,
    prepared_root: Path,
) -> RigConfig:
    config = _base_config(
        repository,
        experiment,
        prepared_root,
        "ap03",
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
    prepared_root: Path,
    *,
    profile: str,
) -> list[tuple[str, RigConfig]]:
    configs = [
        (
            "ap01_historical_root_cam3",
            historical_ap01(repository, experiment, prepared_root),
        ),
        (
            "ap01_robust_direct_first_sparse2_auto_root",
            robust_ap01(repository, experiment, prepared_root),
        ),
    ]
    if profile == "full":
        configs.extend(
            [
                (
                    "ap02_baseline_ref14",
                    baseline_ap02(repository, experiment, prepared_root),
                ),
                (
                    "ap03_baseline_v1",
                    baseline_ap03(repository, experiment, prepared_root),
                ),
            ]
        )
    return configs


def _probe_config(repository: Path, experiment: Path) -> RigConfig:
    _, config = _resolved_rerun_config(
        repository,
        experiment,
        "ap01",
        "baseline",
        ap01_method_contract="baseline_v1",
    )
    return config


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
        probe = _probe_config(repository, experiment)
        prepared_root = _resolve_existing_input_root(
            repository,
            experiment,
            probe,
        )
        moving, static, camera_info = _input_counts(prepared_root)
        print(
            f"[OK] {name} prepared input: {prepared_root} "
            f"({moving} moving, {static} static, {camera_info} CameraInfo)"
        )

        config_entries: list[tuple[str, Path]] = []
        experiment_dir = root / name
        for entry_id, config in build_configs(
            repository,
            experiment,
            prepared_root,
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
    print("[OK] input mode: existing prepared pixels; Gazebo capture disabled")
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
