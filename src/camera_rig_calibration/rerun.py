"""Safe single-method reruns from an existing layout-v2 experiment."""

from __future__ import annotations

import json
import os
import shutil
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import config_fingerprint, load_config, save_config
from .config.models import AP03Settings, InputSourceKind, RigConfig
from .dataset_identity import build_dataset_identity, identities_match
from .experiments import colmap_artifact_fingerprint
from .publication import reconcile_existing_experiment
from .queueing import QueueConfig, QueueEntry, QueueRunner
from .storage_layout import queue_temporary_root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _copy_dataset(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("raw_images", "metadata"):
        item = source / name
        if item.is_dir():
            shutil.copytree(
                item,
                destination / name,
                copy_function=_link_or_copy,
            )
    observations = source / "observations"
    if observations.is_dir():
        # Preflight writes queue-specific quality evidence.  Regular copies
        # prevent those writes from modifying the canonical immutable dataset.
        shutil.copytree(observations, destination / "observations")
    for name in ("dataset.json", "README.txt"):
        item = source / name
        if item.is_file():
            shutil.copy2(item, destination / name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _published_observation_contract(
    experiment: Path,
    config: RigConfig,
) -> dict[str, Any]:
    observations = experiment / "observations"
    detection_path = observations / "detection_config.json"
    required_csvs = tuple(
        observations / name
        for name in (
            "shared_static_aruco_observations.csv",
            "shared_moving_aruco_observations.csv",
            "shared_all_aruco_observations.csv",
        )
    )
    missing = [
        str(path.relative_to(experiment))
        for path in (detection_path, *required_csvs)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Prepared rerun requires the published observations; "
            "missing: " + ", ".join(missing)
        )
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    observation_id = str(detection.get("observation_id", "")).strip()
    if not observation_id:
        raise RuntimeError(
            "Prepared observations do not declare an observation_id"
        )
    dataset = json.loads(
        (experiment / "dataset.json").read_text(encoding="utf-8")
    )
    input_id = str(dataset.get("input_fingerprint", "")).strip()
    if detection.get("input_id") != input_id:
        raise RuntimeError(
            "Prepared observation input_id does not match dataset.json"
        )
    stored_markers = detection.get("markers")
    if not isinstance(stored_markers, dict):
        raise RuntimeError(
            "Prepared observations do not contain their marker contract"
        )
    requested = config.markers
    compatible = (
        stored_markers.get("dictionary") == requested.dictionary
        and stored_markers.get("detection_mode") == requested.detection_mode
    )
    try:
        compatible = compatible and abs(
            float(stored_markers.get("length_m"))
            - float(requested.length_m)
        ) <= 1e-12
    except (TypeError, ValueError):
        compatible = False
    if not compatible:
        raise RuntimeError(
            "Prepared observations use a different marker dictionary, "
            "marker length, or detection mode. A single-method rerun must "
            "not reinterpret or overwrite that scientific evidence."
        )
    return {
        "observation_id": observation_id,
        "input_id": input_id,
        "detection_config_sha256": _file_sha256(detection_path),
        "observation_csv_sha256": {
            path.name: _file_sha256(path) for path in required_csvs
        },
        "effective_detector": detection.get("effective_detector"),
        "detector_contract": detection.get("detector_contract"),
        "source": "published_immutable_observations",
    }


def _method_config_path(
    experiment: Path, method: str, variant: str
) -> Path:
    published = (
        experiment
        / "methods"
        / method
        / variant
        / "provenance"
        / "resolved_config.yaml"
    )
    if published.is_file():
        return published
    candidates = sorted(
        (
            path
            for path in (
                experiment / "attempts" / method / variant
            ).glob("*/diagnostics/resolved_config.yaml")
            if path.is_file()
        ),
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (
                path
                for path in (
                    experiment / "attempts" / method / variant
                ).glob("*/resolved_config.yaml")
                if path.is_file()
            ),
            reverse=True,
        )
    if not candidates:
        # A prepared experiment may legitimately have no prior attempt for the
        # requested method. Reuse another published resolved config from this
        # same immutable experiment, then replace the enabled method/settings
        # below. This never imports scientific outputs from the other method.
        candidates = sorted(
            experiment.glob(
                "methods/*/*/provenance/resolved_config.yaml"
            ),
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(
            f"No resolved configuration exists for {method}/{variant} "
            f"below {experiment}"
        )
    return candidates[0]


def _latest_failed_attempt(
    experiment: Path, method: str, variant: str
) -> str | None:
    failures = sorted(
        (experiment / "attempts" / method / variant).glob("*/FAILURE.json"),
        reverse=True,
    )
    if not failures:
        return None
    return failures[0].parent.relative_to(experiment).as_posix()


def _resolved_rerun_config(
    repository_root: Path,
    experiment: Path,
    method: str,
    variant: str,
    ap01_method_contract: str | None = None,
    ap03_method_contract: str | None = None,
) -> tuple[RigConfig, RigConfig]:
    if ap01_method_contract is not None and method != "ap01":
        raise ValueError(
            "--ap01-method-contract is valid only with --method ap01"
        )
    if ap03_method_contract is not None and method != "ap03":
        raise ValueError(
            "--ap03-method-contract is valid only with --method ap03"
        )
    source = load_config(_method_config_path(experiment, method, variant))
    methods = source.methods.model_copy(update={"enabled": [method]}, deep=True)
    if method == "ap01" and variant == "baseline":
        contract_name = ap01_method_contract or "baseline_v1"
        methods = methods.model_copy(
            update={
                "ap01": methods.ap01.model_copy(
                    update={
                        "method_contract": contract_name,
                    },
                    deep=True,
                )
            },
            deep=True,
        )
    elif ap01_method_contract is not None:
        methods = methods.model_copy(
            update={
                "ap01": methods.ap01.model_copy(
                    update={"method_contract": ap01_method_contract},
                    deep=True,
                )
            },
            deep=True,
        )
    if method == "ap02" and variant == "baseline":
        methods = methods.model_copy(
            update={
                "ap02": methods.ap02.model_copy(
                    update={
                        "method_contract": "baseline_v1",
                        "reference_marker_selection_mode": "baseline",
                        "reference_marker_id": 14,
                        "frame_selection_strategy": "smart_v1",
                        "initialization_strategy": (
                            "maximum_frontier_v1"
                        ),
                        "graph_edge_weight_strategy": (
                            "geometric_observation_quality_v1"
                        ),
                        "reprojection_model": "pinhole_v1",
                        "reference_marker_maximum_frames": None,
                        "top_per_marker": 8,
                        "top_per_marker_pair": 4,
                        "maximum_total_frames": None,
                        "static_only_ba_max_function_evaluations": 80,
                        "combined_ba_max_function_evaluations": 80,
                        "ba_robust_loss": "soft_l1",
                        "ba_robust_loss_scale_px": 3.0,
                    }
                )
            },
            deep=True,
        )
    if method == "ap03" and variant == "baseline":
        contract_name = ap03_method_contract or "baseline_v1"
        if contract_name != "baseline_v1":
            raise ValueError("AP03 supports only baseline_v1")
        methods = methods.model_copy(
            update={
                "ap03": AP03Settings(method_contract="baseline_v1")
            },
            deep=True,
        )
    moving_intrinsics = (
        experiment
        / "raw_images"
        / "camera_info"
        / f"{source.moving_camera.id}.json"
    )
    updated = source.model_copy(
        update={
            "project": source.project.model_copy(
                update={
                    "workspace_root": repository_root / "workspace",
                    "dataset_cache_root": (
                        repository_root / "workspace" / "preparation_cache"
                    ),
                    "output_root": repository_root / "results",
                    "experiment_id": experiment.name,
                    "run_label": variant,
                    "execution_mode": "complete",
                    "duplicate_policy": "force",
                }
            ),
            "dataset": source.dataset.model_copy(
                update={
                    "source_kind": InputSourceKind.PREPARED,
                    "prepared_root": experiment,
                },
                deep=True,
            ),
            "moving_camera": source.moving_camera.model_copy(
                update={
                    "intrinsics": (
                        moving_intrinsics
                        if moving_intrinsics.is_file()
                        else source.moving_camera.intrinsics
                    ),
                    "video": None,
                    "frames": None,
                },
                deep=True,
            ),
            "methods": methods,
        },
        deep=True,
    )
    return source, RigConfig.model_validate(updated.model_dump(mode="python"))


def _validate_ap01_reuse(
    experiment: Path,
    original: RigConfig,
    rerun: RigConfig,
) -> Path:
    public = experiment / "methods" / "ap01" / rerun.project.run_label
    result = json.loads((public / "RESULT.json").read_text(encoding="utf-8"))
    descriptor = json.loads(
        (experiment / "dataset.json").read_text(encoding="utf-8")
    )
    input_id = str(descriptor.get("input_fingerprint", ""))
    if not input_id or result.get("input_fingerprint") != input_id:
        raise RuntimeError(
            "AP01 intermediate reuse refused: public method and dataset "
            "input fingerprints differ."
        )
    old = colmap_artifact_fingerprint(original, "ap01", input_id)
    new = colmap_artifact_fingerprint(rerun, "ap01", input_id)
    if old != new:
        raise RuntimeError(
            "AP01 intermediate reuse refused: the CPU-COLMAP configuration "
            "fingerprint changed."
        )
    method_root = public / "diagnostics" / "method"
    for relative in (
        "moving_colmap/sparse_txt_best/images.txt",
        "metric_scale/metric_scale.txt",
        "metric_scale/SCALE_DIAGNOSTICS.json",
    ):
        if not (method_root / relative).is_file():
            raise RuntimeError(
                "AP01 intermediate reuse refused: missing " + relative
            )
    return method_root


@dataclass(frozen=True)
class PreparedRerun:
    queue: QueueConfig
    config: RigConfig
    transaction_root: Path
    reuse_intermediates_from: Path | None
    dataset_identity: dict[str, Any]
    observation_contract: dict[str, Any]


def prepare_single_method_rerun(
    *,
    repository_root: Path,
    experiment: Path,
    method: str,
    variant: str,
    reuse_prepared_input: bool,
    reuse_matching_intermediates: bool,
    ap01_method_contract: str | None = None,
    ap03_method_contract: str | None = None,
) -> PreparedRerun:
    if method not in {"ap01", "ap02", "ap03"}:
        raise ValueError("method must be ap01, ap02 or ap03")
    if not reuse_prepared_input:
        raise ValueError(
            "Single-method repair runs require --reuse-prepared-input; "
            "capture is intentionally unavailable in this command."
        )
    repository = repository_root.resolve()
    experiment_root = experiment.resolve()
    if not (experiment_root / "dataset.json").is_file():
        raise FileNotFoundError(
            f"Layout-v2 dataset descriptor is missing: {experiment_root}"
        )
    original, config = _resolved_rerun_config(
        repository,
        experiment_root,
        method,
        variant,
        ap01_method_contract=ap01_method_contract,
        ap03_method_contract=ap03_method_contract,
    )
    observation_contract = _published_observation_contract(
        experiment_root, config
    )
    identity = build_dataset_identity(experiment_root)
    if not identity.get("content_files"):
        raise RuntimeError("Existing experiment has no immutable dataset content")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    queue_id = f"{experiment_root.name}_{method}_{variant}_rerun_{stamp}"
    entry_id = f"{method}_{variant}"
    config_path = (
        repository / "workspace" / "reruns" / queue_id / "config.yaml"
    )
    save_config(config, config_path)
    queue = QueueConfig(
        id=queue_id,
        entries=[QueueEntry(id=entry_id, config=config_path)],
    )
    transaction = queue_temporary_root(config, queue.id)
    if transaction.exists():
        raise RuntimeError(f"Rerun transaction already exists: {transaction}")
    _copy_dataset(experiment_root, transaction / "dataset")
    copied_identity = build_dataset_identity(transaction / "dataset")
    if not identities_match(identity, copied_identity):
        raise RuntimeError(
            "Prepared rerun copy does not match the canonical dataset identity"
        )
    preparation = (
        transaction
        / "jobs"
        / "queue_preflight"
        / "reused_prepared_dataset"
    )
    _write_json(
        preparation / "00_INPUT" / "dataset_pointer.json",
        {
            "dataset_root": str((transaction / "dataset").resolve()),
            "prepared_source_root": str(experiment_root),
            "prepared_input": True,
            "input_id": json.loads(
                (experiment_root / "dataset.json").read_text(
                    encoding="utf-8"
                )
            ).get("input_fingerprint"),
        },
    )
    _write_json(
        preparation / "run_manifest.json",
        {
            "schema_version": 5,
            "status": "completed",
            "execution_mode": "prepare_only_reused",
            "observations_root": str(
                (transaction / "dataset" / "observations").resolve()
            ),
            "observation_contract": observation_contract,
            "dataset_identity": identity,
            "capture_repeated": False,
            "detection_repeated": False,
        },
    )
    _write_json(
        transaction / "queue_state.json",
        {
            "schema_version": 5,
            "queue_id": queue.id,
            "updated_at": _now(),
            "entries": {},
            "source_fingerprints": {
                entry_id: config_fingerprint(config)
            },
            "resolved_configs": {},
            "preflight_preparation": str(preparation.resolve()),
            "observation_coverage_override": False,
            "observation_review": {
                "status": "prepared_dataset_reused",
                "capture_repeated": False,
                "detection_repeated": False,
            },
        },
    )
    _write_json(
        transaction / "queue_dataset_identity.json",
        {
            **identity,
            "queue_id": queue.id,
            "prepared_root": str(experiment_root),
            "scope": "queue_shared_immutable_dataset",
        },
    )
    reuse_source = None
    if reuse_matching_intermediates:
        if method != "ap01":
            raise ValueError(
                "--reuse-matching-intermediates is currently supported only "
                "for AP01 CPU-COLMAP and metric scale."
            )
        reuse_source = _validate_ap01_reuse(
            experiment_root, original, config
        )
    return PreparedRerun(
        queue=queue,
        config=config,
        transaction_root=transaction,
        reuse_intermediates_from=reuse_source,
        dataset_identity=identity,
        observation_contract=observation_contract,
    )


def run_single_method_rerun(
    *,
    repository_root: Path,
    experiment: Path,
    method: str,
    variant: str,
    reuse_prepared_input: bool,
    reuse_matching_intermediates: bool,
    reconcile_after: bool,
    ap01_method_contract: str | None = None,
    ap03_method_contract: str | None = None,
    console: Console | None = None,
) -> dict[str, dict[str, Any]]:
    prepared = prepare_single_method_rerun(
        repository_root=repository_root,
        experiment=experiment,
        method=method,
        variant=variant,
        reuse_prepared_input=reuse_prepared_input,
        reuse_matching_intermediates=reuse_matching_intermediates,
        ap01_method_contract=ap01_method_contract,
        ap03_method_contract=ap03_method_contract,
    )
    entry_id = prepared.queue.entries[0].id
    supersedes = _latest_failed_attempt(
        experiment.resolve(), method, variant
    )
    runner = QueueRunner(
        repository_root,
        console,
        reuse_method_intermediates=(
            {entry_id: prepared.reuse_intermediates_from}
            if prepared.reuse_intermediates_from is not None
            else None
        ),
        rerun_metadata={
            entry_id: {
                "supersedes_attempt": supersedes,
                "dataset_identity": prepared.dataset_identity,
                "reuse_published_observations": True,
                "published_observation_contract": (
                    prepared.observation_contract
                ),
                "capture_repeated": False,
                "detection_repeated": False,
                "rerun_requested_at": _now(),
            }
        },
        explicit_method_rerun=True,
    )
    results = runner.run(prepared.queue)
    published_successfully = bool(results) and all(
        result.get("status") == "completed"
        and result.get("published") is True
        for result in results.values()
    )
    if reconcile_after and published_successfully:
        reconcile_experiment(experiment.resolve())
    return results


def reconcile_experiment(experiment: Path) -> dict[str, Any]:
    root = experiment.resolve()
    descriptor = json.loads(
        (root / "dataset.json").read_text(encoding="utf-8")
    )
    return reconcile_existing_experiment(
        root,
        dataset_root=root,
        category=str(descriptor.get("category", "")),
    )
