from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .components import register_builtin_components
from .config import config_fingerprint, load_config, save_user_config
from .config.models import (
    ColmapSettings,
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    McapSettings,
    MethodSettings,
    MarkerSettings,
    MovingCameraSettings,
    IntrinsicScanSettings,
    InputSourceKind,
    ObservationQualitySettings,
    ProjectSettings,
    RigConfig,
    SamplingSettings,
    SceneType,
    SelectionSettings,
    SimulationSettings,
    StaticCameraSettings,
    effective_observation_quality,
)
from .dataset.discovery import (
    IMAGE_SUFFIXES,
    discover_image_directories,
    discover_inputs,
    inspect_prepared_dataset,
    media_path_role,
    safe_id,
)
from .doctor import run_checks
from .experiments import automatic_method_label
from .input.topics import McapTopic, list_mcap_topics
from .input.video_geometry import probe_video_geometry
from .intrinsics_profiles import (
    IntrinsicProfile,
    discover_intrinsic_profiles,
    intrinsic_dimensions,
)
from .inventory import (
    BASELINE_SIMULATION_PARAMETERS,
    PreparedDatasetSummary,
    RawInputSummary,
    SimulationExperimentSummary,
    discover_prepared_datasets,
    discover_raw_input_folders,
    discover_simulation_experiments,
    find_matching_simulation,
    format_simulation_parameters,
)
from .registry import (
    calibration_methods,
    experiment_providers,
    input_adapters,
)
from .runtime import PipelineOrchestrator
from .observation_quality import filter_observations
from .observations import ResolvedSelections, resolve_selections
from .queueing import SelectionReviewJob, save_batch
# Compatibility hooks wrapped by the product policy stack. The concrete result
# browser lives under ui/, but these names remain stable until the wrappers are
# converted to explicit composition.
from .publication import reconcile_existing_experiment
from .visualization import launch_isolated_rviz



from .ui.wizard_models import (
    WizardOutcome,
    QueuedRun,
    MethodQueueJob,
    SelectionDatasetContext,
    _refresh_method_job_label,
    _MANUAL_SELECTION_LABELS,
    _pending_selection_keys,
    _method_job_label,
    _method_job_identity,
    SimulationQueueJob,
    _BusCamera,
    _BusRoute,
    _BusDefinition,
    _bus_definition,
    WizardBack,
)
from .ui.wizard_prompts import (
    _clear_terminal,
    _show_input_error,
    _prompt_index,
    _simulation_experiment_id,
    _choice,
    _select_detected_path,
    _preferred_path,
    _looks_like_checkerboard_video,
    _checkerboard_sources,
    _select_checkerboard_source,
    _moving_media_dimensions,
    _show_video_geometry,
    _prompt_intrinsic_scan_settings,
    _bool_value,
    _format_setting_value,
    _PUBLIC_POLICY_NAMES,
    _public_policy_name,
    _optional_positive_int,
    _prompt_enum_choice,
)
from .ui.wizard_input_metadata import (
    _stored_prepared_marker_settings,
    _stored_prepared_sampling,
)
from .ui.wizard_media import (
    _moving_source,
    _camera_file_key,
    _detected_static_pairs,
    _static_group_key,
    _detected_static_camera_groups,
    _direct_static_cameras,
    _relative_display,
    _show_input_inventory,
    _show_prepared_choices,
    _prepared_input,
)
from .ui.wizard_prepared import (
    _prepared_moving_intrinsics,
)
from .ui.wizard_real_input import (
    _data_local_input_root,
    _ros_image_stream_prefix,
    _related_camera_info_topics,
    _camera_id_from_ros_topic,
    _mcap_camera_sources,
    _real_data_input,
)
from .ui.wizard_simulation_parameters import (
    _lighting_profiles,
    _show_lighting_profiles,
    _edit_simulation_parameters,
)
from .ui.wizard_simulation import (
    _simulation_input,
    _simulation_job_from_parameters,
    _simulation_signature,
    _show_simulation_input_queue,
    _parse_experiment_numbers,
    _simulation_input_queue,
)
from .ui.wizard_method_jobs import (
    _new_method_job,
    _method_job_summary,
    _show_method_queue,
    _ids_text,
    _parse_ids,
    GUIDED_SELECTION_KEYS,
    _methods_with_automatic_selections,
    _selection_value,
    _methods_with_selection,
    _selection_mode_for_methods,
    _job_methods,
    _job_selection,
    _refresh_job_selection_mode,
    _sync_context_methods,
    _preview_prepared_selections,
    _validate_prepared_job_selections,
    _selection_candidates,
    _candidate_compatible,
    _show_guided_candidates,
    _prompt_guided_candidate,
    _configure_guided_selection,
    _prompt_component_options,
)
from .ui.wizard_method_queue import (
    METHOD_JOB_GROUPS,
    _setting_rows,
    _edit_method_job,
    _clone_method_job,
    _method_queue,
)
from .ui.wizard_new_flow import (
    _base_project,
    _create_intrinsic_profile_only,
    _new_dataset_id,
    _aruco_experiment_id,
    _rekey_method_contexts,
    _build_simulation_batch_outcome,
    new_calibration_wizard,
    _save_wizard_queue,
)
from .ui.wizard_saved_flow import (
    _load_saved_setup_config,
    _config_candidates,
    saved_setup_count,
    choose_config,
    repeat_setup_wizard,
    advanced_wizard,
    show_summary,
)
from .ui.wizard_review import (
    _review_common_anchor,
    review_selection_candidates,
    review_queue_selection_candidates,
    show_queue_summary,
)

from .ui.result_browser import _is_internal_evidence_result, show_results
from .ui.storage_cleanup import human_size as _human_size


def cleanup_storage_wizard(
    repository_root: Path, console: Console
) -> None:
    from .ui.storage_cleanup import run_storage_cleanup

    run_storage_cleanup(
        repository_root,
        console,
        run_is_active=_run_process_is_active,
        show_error=_show_input_error,
    )


def manage_intrinsics_profiles(
    repository_root: Path, console: Console
) -> None:
    from .ui.intrinsics import run_intrinsics_manager

    run_intrinsics_manager(
        repository_root,
        console,
        choose=_choice,
        create_profile=_create_intrinsic_profile_only,
        relative_display=_relative_display,
        show_error=_show_input_error,
    )


def show_doctor(repository_root: Path, console: Console) -> None:
    checks = run_checks(repository_root, needs_ros=True)
    table = Table(title="Installation check")
    table.add_column("Status")
    table.add_column("Component")
    table.add_column("Detail")
    for check in checks:
        table.add_row("OK" if check.ok else "MISSING", check.name, check.detail)
    console.print(table)
    required_failures = [check for check in checks if check.required and not check.ok]
    if required_failures:
        console.print("Required components are missing. See the installation guide in README.md.")
    else:
        console.print("Installation is ready for the configured baseline methods.")


from .ui.run_management import (
    _active_run_stage,
    _delete_incomplete_run,
    _interrupt_incomplete_run,
    _manifest_process_is_active,
    _remove_failed_queue_jobs,
    _run_process_is_active,
    _transaction_payload,
    _validated_incomplete_run,
    find_incomplete_transaction,
    incomplete_resume_source,
    incomplete_runs,
    manage_incomplete_runs,
)

__all__ = [
    'WizardOutcome',
    'QueuedRun',
    'MethodQueueJob',
    'SelectionDatasetContext',
    '_refresh_method_job_label',
    '_MANUAL_SELECTION_LABELS',
    '_pending_selection_keys',
    '_method_job_label',
    '_method_job_identity',
    'SimulationQueueJob',
    '_BusCamera',
    '_BusRoute',
    '_BusDefinition',
    '_bus_definition',
    'WizardBack',
    '_clear_terminal',
    '_show_input_error',
    '_prompt_index',
    '_simulation_experiment_id',
    '_choice',
    '_select_detected_path',
    '_preferred_path',
    '_looks_like_checkerboard_video',
    '_checkerboard_sources',
    '_select_checkerboard_source',
    '_moving_media_dimensions',
    '_show_video_geometry',
    '_prompt_intrinsic_scan_settings',
    '_bool_value',
    '_format_setting_value',
    '_PUBLIC_POLICY_NAMES',
    '_public_policy_name',
    '_optional_positive_int',
    '_prompt_enum_choice',
    '_stored_prepared_marker_settings',
    '_stored_prepared_sampling',
    '_moving_source',
    '_camera_file_key',
    '_detected_static_pairs',
    '_static_group_key',
    '_detected_static_camera_groups',
    '_direct_static_cameras',
    '_relative_display',
    '_show_input_inventory',
    '_show_prepared_choices',
    '_prepared_input',
    '_prepared_moving_intrinsics',
    '_data_local_input_root',
    '_ros_image_stream_prefix',
    '_related_camera_info_topics',
    '_camera_id_from_ros_topic',
    '_mcap_camera_sources',
    '_real_data_input',
    '_lighting_profiles',
    '_show_lighting_profiles',
    '_edit_simulation_parameters',
    '_simulation_input',
    '_simulation_job_from_parameters',
    '_simulation_signature',
    '_show_simulation_input_queue',
    '_parse_experiment_numbers',
    '_simulation_input_queue',
    '_new_method_job',
    '_method_job_summary',
    '_show_method_queue',
    '_ids_text',
    '_parse_ids',
    'GUIDED_SELECTION_KEYS',
    '_methods_with_automatic_selections',
    '_selection_value',
    '_methods_with_selection',
    '_selection_mode_for_methods',
    '_job_methods',
    '_job_selection',
    '_refresh_job_selection_mode',
    '_sync_context_methods',
    '_preview_prepared_selections',
    '_validate_prepared_job_selections',
    '_selection_candidates',
    '_candidate_compatible',
    '_show_guided_candidates',
    '_prompt_guided_candidate',
    '_configure_guided_selection',
    '_prompt_component_options',
    'METHOD_JOB_GROUPS',
    '_setting_rows',
    '_edit_method_job',
    '_clone_method_job',
    '_method_queue',
    '_base_project',
    '_create_intrinsic_profile_only',
    '_new_dataset_id',
    '_aruco_experiment_id',
    '_rekey_method_contexts',
    '_build_simulation_batch_outcome',
    'new_calibration_wizard',
    '_save_wizard_queue',
    '_load_saved_setup_config',
    '_config_candidates',
    'saved_setup_count',
    'choose_config',
    'repeat_setup_wizard',
    'advanced_wizard',
    'show_summary',
    '_review_common_anchor',
    'review_selection_candidates',
    'review_queue_selection_candidates',
    'show_queue_summary',
    '_is_internal_evidence_result',
    'show_results',
    '_human_size',
    'cleanup_storage_wizard',
    'manage_intrinsics_profiles',
    'show_doctor',
    '_active_run_stage',
    '_delete_incomplete_run',
    '_interrupt_incomplete_run',
    '_manifest_process_is_active',
    '_remove_failed_queue_jobs',
    '_run_process_is_active',
    '_transaction_payload',
    '_validated_incomplete_run',
    'find_incomplete_transaction',
    'incomplete_resume_source',
    'incomplete_runs',
    'manage_incomplete_runs',
    'reconcile_existing_experiment',
    'launch_isolated_rviz',
]
