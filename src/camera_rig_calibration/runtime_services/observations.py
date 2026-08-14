"""Runtime implementation grouped by one cohesive responsibility."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import csv
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.table import Table

from ..components import register_builtin_components
from ..config import config_fingerprint, load_config, save_config
from ..config.models import RigConfig, effective_observation_quality
from ..contracts import CommandSpec, RunContext
from ..dataset.manifest import AutoSelection, load_dataset_manifest, save_dataset_manifest
from ..dataset.validation import validate_dataset
from ..input.preparation import build_preparation_plan, finalize_dataset
from ..input.topics import resolve_rosbag_source
from ..intrinsics_profiles import resolve_intrinsic_profile
from ..methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from ..experiments import (
    colmap_artifact_fingerprint,
    evaluation_fingerprint,
    experiment_paths,
    input_fingerprint,
    method_config_diff,
    method_fingerprint,
    method_result_label,
    write_experiment_manifest,
)
from ..observations import (
    ResolvedSelections,
    freeze_selections,
    resolve_selections,
)
from ..observation_quality import ObservationQualityError, filter_observations
from ..progress import ProgressClock, progress_text, terminal_lines
from ..pipeline import StageContract, validate_stage_dag
from ..registry import calibration_methods, evaluators, input_adapters
from ..results import write_comparison


from .common import (
    _now,
    _write_json,
    _read_json,
    _materialize_tree,
    observation_id,
)
from .bindings import current_runtime_bindings


class ObservationMixin:
    def _publish_input_view(
        self,
        config: RigConfig,
        dataset_root: Path,
        dataset_manifest=None,
    ) -> str:
        """Publish an immutable, content-addressed input once per experiment."""
        assert self.run_directory is not None
        run = self.run_directory
        paths = self._working_paths(config)
        input_id = input_fingerprint(dataset_manifest, dataset_root)
        published = paths.datasets
        pointer_path = published / "metadata" / "source.json"
        descriptor_path = published / "dataset.json"
        source = dataset_root.resolve()
        reuse_existing = False
        if descriptor_path.is_file():
            descriptor = json.loads(
                descriptor_path.read_text(encoding="utf-8")
            )
            existing_input = str(
                descriptor.get("input_fingerprint", "")
            )
            if existing_input != input_id:
                raise RuntimeError(
                    f"Experiment '{paths.experiment_id}' already contains a "
                    "different immutable dataset. Choose a new experiment ID."
                )
            if not pointer_path.is_file() or not (
                published / "raw_images"
            ).is_dir():
                raise RuntimeError(
                    f"Existing dataset is incomplete: {published}"
                )
            reuse_existing = True
        elif published.is_dir() and any(published.iterdir()):
            raise RuntimeError(
                f"Dataset directory has files but no dataset descriptor: {published}. "
                "Refusing to mix an unknown input."
            )
        if not reuse_existing:
            published.mkdir(parents=True, exist_ok=True)
            totals = {"hardlinked": 0, "copied": 0, "existing": 0}
            for directory in ("raw_images", "metadata"):
                counts = _materialize_tree(
                    source / directory, published / directory
                )
                for key, value in counts.items():
                    totals[key] += value
            _write_json(
                pointer_path,
                {
                    "input_id": input_id,
                    "layout_version": 2,
                    "canonical_source_roots": [str(source)],
                    "status": "ready",
                    "published_at": _now(),
                    "storage": (
                        "hardlinks where supported, byte copies otherwise; "
                        "deleting the source does not remove files from this "
                        "dataset"
                    ),
                    "file_counts": totals,
                    "content_addressing": "normalized input SHA-256",
                },
            )
            if dataset_manifest is not None:
                save_dataset_manifest(
                    dataset_manifest,
                    published / "metadata" / "dataset_manifest.json",
                )
            (published / "README.txt").write_text(
                "Canonical immutable rigcal dataset (layout version 2).\n"
                "raw_images/ contains static, moving and camera_info inputs.\n"
                "observations/ contains shared ArUco CSVs, quality decisions "
                "and the single debug_images collection. metadata/ contains provenance and "
                "validation details.\n"
                "Calibration methods never modify this directory.\n",
                encoding="utf-8",
            )
        run_view = run / "00_INPUT" / "raw_images"
        if not run_view.exists() and not run_view.is_symlink():
            try:
                run_view.symlink_to(
                    (published / "raw_images").resolve(), target_is_directory=True
                )
            except OSError:
                (run / "00_INPUT" / "RAW_IMAGES_LOCATION.txt").write_text(
                    str(published / "raw_images") + "\n", encoding="utf-8"
                )
        write_experiment_manifest(config, paths, input_id)
        self.manifest["input_id"] = input_id
        self.manifest["experiment_root"] = str(paths.root)
        self._save_state()
        return input_id

    def _observation_id(self, config: RigConfig) -> str:
        return observation_id(config)

    def _published_observation_id(
        self,
        config: RigConfig,
        *,
        shared: Path,
        input_id: str,
    ) -> str | None:
        if not self.rerun_metadata.get("reuse_published_observations"):
            return None
        declared = self.rerun_metadata.get("published_observation_contract")
        if not isinstance(declared, dict):
            raise RuntimeError(
                "Prepared rerun is missing its published observation contract"
            )
        stored = _read_json(shared / "detection_config.json")
        stored_id = str(stored.get("observation_id", "")).strip()
        declared_id = str(declared.get("observation_id", "")).strip()
        if not stored_id or stored_id != declared_id:
            raise RuntimeError(
                "Prepared observations no longer match the observation "
                "contract selected for this rerun."
            )
        if stored.get("input_id") != input_id:
            raise RuntimeError(
                "Prepared observations belong to a different immutable input"
            )
        stored_markers = stored.get("markers")
        if not isinstance(stored_markers, dict):
            raise RuntimeError(
                "Prepared observations do not declare their marker contract"
            )
        markers_match = (
            stored_markers.get("dictionary")
            == config.markers.dictionary
            and stored_markers.get("detection_mode")
            == config.markers.detection_mode
        )
        try:
            markers_match = markers_match and abs(
                float(stored_markers.get("length_m"))
                - float(config.markers.length_m)
            ) <= 1e-12
        except (TypeError, ValueError):
            markers_match = False
        if not markers_match:
            raise RuntimeError(
                "Prepared observations use a different marker dictionary, "
                "marker length, or detection mode."
            )
        if not self._observation_contract_ready(shared, stored_id):
            raise RuntimeError(
                "Prepared observations are incomplete and cannot be reused"
            )
        return stored_id

    def detect_observations_only(
        self,
        config: RigConfig,
        *,
        dataset_root: Path,
        run_directory: Path,
    ) -> Path:
        """Re-run only ArUco detection on an already normalized dataset."""
        run = run_directory.resolve()
        run.mkdir(parents=True, exist_ok=True)
        (run / "logs").mkdir(exist_ok=True)
        commands = run / "commands.txt"
        if not commands.is_file():
            commands.write_text("", encoding="utf-8")
        self.run_directory = run
        self._run_command(
            self._detector_command(config, dataset_root.resolve())
        )
        return run / "01_OBSERVATIONS"

    def _bind_observations_view(
        self, config: RigConfig, input_id: str
    ) -> Path:
        assert self.run_directory is not None
        paths = self._working_paths(config)
        shared = paths.datasets / "observations"
        shared.mkdir(parents=True, exist_ok=True)
        observation_id = self._observation_id(config)
        existing_config = _read_json(shared / "detection_config.json")
        existing_observation_id = existing_config.get("observation_id")
        existing_csv = shared / "shared_all_aruco_observations.csv"
        published_observation_id = self._published_observation_id(
            config,
            shared=shared,
            input_id=input_id,
        )
        if published_observation_id is not None:
            observation_id = published_observation_id
        if (
            existing_csv.is_file()
            and existing_observation_id
            and existing_observation_id != observation_id
        ):
            raise RuntimeError(
                "This experiment already contains observations generated with "
                "a different ArUco detector contract. Use a distinct experiment "
                f"ID (recommended suffix: __aruco_{config.markers.detection_mode}) "
                "instead of overwriting scientific evidence."
            )
        if published_observation_id is None:
            _write_json(
                shared / "detection_config.json",
                {
                    "schema_version": 5,
                    "layout_version": 2,
                    "input_id": input_id,
                    "observation_id": observation_id,
                    "markers": config.markers.model_dump(mode="json"),
                    "effective_detector": effective_detector_config(
                        config.markers.detection_mode,
                        config.markers.dictionary,
                    ),
                    "detector_contract": DETECTOR_CONTRACT,
                    "observation_input_contract": (
                        "raw_detection_with_dimensions_and_area_ratio_v2"
                    ),
                },
            )
        view = self.run_directory / "01_OBSERVATIONS"
        if view.is_symlink():
            if view.resolve() != shared.resolve():
                raise RuntimeError(
                    f"Run is already bound to different observations: {view.resolve()}"
                )
        else:
            if view.is_dir() and any(view.iterdir()):
                _materialize_tree(view, shared)
                shutil.rmtree(view)
            if view.is_dir():
                view.rmdir()
            view.symlink_to(shared.resolve(), target_is_directory=True)
        self.manifest["observation_id"] = observation_id
        self.manifest["observations_root"] = str(shared)
        self.manifest["published_observations_reused"] = (
            published_observation_id is not None
        )
        if published_observation_id is not None:
            self.manifest["observation_runtime_detector_id"] = (
                self._observation_id(config)
            )
        self._save_state()
        return shared

    def _finalize_dataset_observations(
        self,
        config: RigConfig,
        *,
        quality_observations_root: Path,
    ) -> None:
        """Freeze selection and quality evidence into the immutable dataset."""
        assert self.run_directory is not None
        dataset_root = self._working_paths(config).datasets
        observations = dataset_root / "observations"
        observations.mkdir(parents=True, exist_ok=True)
        completion = observations / "PUBLICATION_COMPLETE.json"
        existing_completion = _read_json(completion)
        if existing_completion.get("status") == "complete":
            self.manifest["dataset_observation_evidence_reused"] = True
            self._save_state()
            return
        required_selection = (
            "SELECTION_CANDIDATES.json",
            "SELECTION_CANDIDATES.csv",
            "REFERENCE_SELECTIONS.json",
            "REFERENCE_MARKER_ID.txt",
        )
        missing = [
            name
            for name in required_selection
            if not (quality_observations_root / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                "Selection analysis completed without publishable evidence: "
                + ", ".join(missing)
            )
        for name in required_selection:
            shutil.copy2(
                quality_observations_root / name,
                observations / name,
            )

        quality = observations / "quality"
        quality.mkdir(parents=True, exist_ok=True)
        for name in (
            "accepted_observations.csv",
            "rejected_observations.csv",
            "observation_filter_summary.json",
            "preflight_summary.json",
            "marker_inventory.csv",
            "marker_inventory.json",
        ):
            source = self.run_directory / "preflight" / name
            if source.is_file():
                shutil.copy2(source, quality / name)
        manifest = self.run_directory / "00_INPUT" / "dataset_manifest.json"
        if manifest.is_file():
            destination = dataset_root / "metadata" / "dataset_manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, destination)
        # A queue shares one immutable prepared dataset across all method
        # jobs.  Its first/preflight finalization is authoritative; rewriting
        # the timestamp for every method would make byte-identical
        # publication look like a dataset conflict.
        _write_json(
            completion,
            {
                "schema_version": 5,
                "layout_version": 2,
                "status": "complete",
                "selection_files": list(required_selection),
                "quality_directory": "quality",
                "debug_images": (
                    "debug_images"
                    if (observations / "debug_images").is_dir()
                    else None
                ),
                "finalized_at": _now(),
            },
        )

    @staticmethod
    def _observation_contract_ready(
        root: Path,
        expected_observation_id: str | None = None,
    ) -> bool:
        paths = [
            root / name
            for name in (
                "shared_static_aruco_observations.csv",
                "shared_moving_aruco_observations.csv",
                "shared_all_aruco_observations.csv",
            )
        ]
        if not all(path.is_file() for path in paths):
            return False
        if expected_observation_id is not None:
            config = _read_json(root / "detection_config.json")
            if config.get("observation_id") != expected_observation_id:
                return False
        try:
            with paths[-1].open(newline="", encoding="utf-8") as handle:
                fields = set(next(csv.reader(handle)))
        except (OSError, StopIteration):
            return False
        return {
            "detection_success",
            "detection_mode",
            "detection_source",
            "detector_contract",
            "opencv_version",
            "pnp_reprojection_rmse_px",
            "corner0_u",
            "corner3_v",
        }.issubset(fields)



__all__ = ['ObservationMixin']
