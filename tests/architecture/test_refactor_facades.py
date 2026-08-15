from __future__ import annotations

from camera_rig_calibration import (
    assets,
    dataset_identity,
    filesystem,
    observations,
    preflight,
    progress,
    queueing,
    results,
    runtime,
    storage,
    wizard,
)
from camera_rig_calibration.dataset import identity as dataset_identity_impl
from camera_rig_calibration.evaluation import reporting
from camera_rig_calibration.evaluation.reporting_bindings import (
    current_reporting_bindings,
)
from camera_rig_calibration.input import preparation
from camera_rig_calibration.input.preparation_planning import build_preparation_plan
from camera_rig_calibration.methods.ap01 import core as ap01_core
from camera_rig_calibration.methods.ap01.core_bindings import AP01CoreBindings
from camera_rig_calibration.methods.ap01.core_scale import robust_scale
from camera_rig_calibration.methods.ap02 import initialize as ap02_initialize
from camera_rig_calibration.methods.ap02.initialize_graph import build_graph
from camera_rig_calibration.observation_core import ResolvedSelections
from camera_rig_calibration.preflight_services.bindings import PreflightDependencies
from camera_rig_calibration.preflight_services.core import PreflightJob
from camera_rig_calibration.publication_services import results as results_impl
from camera_rig_calibration.queue_services.bindings import current_queue_bindings
from camera_rig_calibration.runtime_services import progress as progress_impl
from camera_rig_calibration.runtime_services.bindings import current_runtime_bindings
from camera_rig_calibration.storage_services import assets as assets_impl
from camera_rig_calibration.storage_services import cleanup as storage_impl
from camera_rig_calibration.storage_services import filesystem as filesystem_impl
from camera_rig_calibration.ui.wizard_bindings import current_wizard_bindings


def test_compatibility_facades_reexport_split_implementations() -> None:
    assert observations.ResolvedSelections is ResolvedSelections
    assert preflight.PreflightJob is PreflightJob
    assert preparation.build_preparation_plan is build_preparation_plan
    assert ap01_core.robust_scale is robust_scale
    assert ap02_initialize.build_graph is build_graph
    assert dataset_identity.build_dataset_identity is dataset_identity_impl.build_dataset_identity
    assert progress.ProgressClock is progress_impl.ProgressClock
    assert results.index_results is results_impl.index_results
    assert storage.CleanupPlan is storage_impl.CleanupPlan
    assert filesystem.rename_with_retry is filesystem_impl.rename_with_retry
    assert assets.materialize_gzip_asset is assets_impl.materialize_gzip_asset


def test_late_bindings_follow_facade_monkey_patches(monkeypatch) -> None:
    hook = lambda *args, **kwargs: None
    orchestrator = type("PatchedOrchestrator", (), {})

    monkeypatch.setattr(wizard, "_choice", hook)
    monkeypatch.setattr(queueing, "PipelineOrchestrator", orchestrator)
    monkeypatch.setattr(runtime, "resolve_selections", hook)
    monkeypatch.setattr(reporting, "_read_json", hook)
    monkeypatch.setattr(preflight, "resolve_selections", hook)
    monkeypatch.setattr(ap01_core, "run_colmap", hook)

    assert current_wizard_bindings().choice is hook
    assert current_queue_bindings().pipeline_orchestrator is orchestrator
    assert current_runtime_bindings().resolve_selections is hook
    assert current_reporting_bindings().read_json is hook
    assert PreflightDependencies.current().resolve_selections is hook
    assert AP01CoreBindings.current().run_colmap is hook
