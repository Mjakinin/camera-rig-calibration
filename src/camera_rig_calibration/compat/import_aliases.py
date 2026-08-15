"""Central compatibility aliases for historical rigcal import paths."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Mapping


class _LazyModuleAlias(ModuleType):
    """Module proxy that imports its canonical target on first attribute use."""

    def __init__(self, fullname: str, target: str) -> None:
        super().__init__(fullname)
        ModuleType.__setattr__(self, "_alias_target", target)
        ModuleType.__setattr__(self, "__package__", fullname.rpartition(".")[0])
        ModuleType.__setattr__(
            self,
            "__doc__",
            f"Compatibility alias for {target}.",
        )

    def _load(self) -> ModuleType:
        fullname = ModuleType.__getattribute__(self, "__name__")
        target_name = ModuleType.__getattribute__(self, "_alias_target")
        target = importlib.import_module(target_name)
        sys.modules[fullname] = target
        parent_name, _, attribute = fullname.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attribute, target)
        return target

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_") or (name.startswith("__") and name.endswith("__")):
            ModuleType.__setattr__(self, name, value)
            return
        setattr(self._load(), name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._load())))


def _bind_parent(fullname: str, module: ModuleType) -> None:
    parent_name, _, attribute = fullname.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attribute, module)


def _install_module_alias(fullname: str, target: str) -> None:
    if fullname in sys.modules:
        return
    proxy = _LazyModuleAlias(fullname, target)
    sys.modules[fullname] = proxy
    _bind_parent(fullname, proxy)


def _install_package_alias(fullname: str, target: str) -> None:
    if fullname in sys.modules:
        return
    module = importlib.import_module(target)
    sys.modules[fullname] = module
    _bind_parent(fullname, module)


def _full(relative: str) -> str:
    return f"camera_rig_calibration.{relative}"


LEGACY_MODULE_ALIASES: Mapping[str, str] = {
    "ap01_auto_direct": "methods.ap01.automatic_target",
    "ap02_graph": "methods.ap02.graph_diagnostics",
    "assets": "storage_services.assets",
    "bootstrap": "application.bootstrap",
    "cli": "application.cli",
    "contracts": "core.contracts",
    "dataset_identity": "dataset.identity",
    "doctor": "application.doctor",
    "experiments": "experiment_services.api",
    "filesystem": "storage_services.filesystem",
    "intrinsics_profiles": "input.profiles",
    "inventory": "dataset.inventory",
    "observation_core": "observation_services.core",
    "observation_detection": "observation_services.detection",
    "observation_quality": "observation_services.quality",
    "observations": "observation_services.api",
    "preflight": "preflight_services.api",
    "product_cli": "application.product_cli",
    "progress": "runtime_services.progress",
    "publication": "publication_services.api",
    "queueing": "queue_services.api",
    "registry": "core.registry",
    "rerun": "runtime_services.rerun",
    "results": "publication_services.results",
    "runtime": "runtime_services.api",
    "storage": "storage_services.cleanup",
    "storage_layout": "storage_services.layout",
    "wizard": "ui.wizard",
    "wizard_presentation": "ui.presentation",
}


# A few implementation files are moved byte-for-byte.  Their historical
# package-relative imports are mapped here to the canonical modules so the
# structural move does not duplicate scientific code or change behavior.
SCOPED_PACKAGE_ALIASES: Mapping[str, str] = {
    "application.components": "components",
    "core.config": "config",
    "core.method_sdk": "method_sdk",
    "dataset.dataset": "dataset",
    "input.dataset": "dataset",
    "input.input": "input",
    "storage_services.config": "config",
    "storage_services.dataset": "dataset",
    "experiment_services.config": "config",
    "experiment_services.experiment_services": "experiment_services",
    "observation_services.observation_services": "observation_services",
    "preflight_services.methods": "methods",
    "preflight_services.components": "components",
    "preflight_services.config": "config",
    "preflight_services.preflight_services": "preflight_services",
    "publication_services.config": "config",
    "publication_services.anchor_export": "anchor_export",
    "publication_services.dataset": "dataset",
    "publication_services.evaluation": "evaluation",
    "publication_services.publication_services": "publication_services",
    "queue_services.config": "config",
    "queue_services.dataset": "dataset",
    "queue_services.methods": "methods",
    "queue_services.queue_services": "queue_services",
    "runtime_services.components": "components",
    "runtime_services.config": "config",
    "runtime_services.dataset": "dataset",
    "runtime_services.input": "input",
    "runtime_services.methods": "methods",
    "runtime_services.pipeline": "pipeline",
    "runtime_services.runtime_services": "runtime_services",
    "ui.components": "components",
    "ui.config": "config",
    "ui.dataset": "dataset",
    "ui.input": "input",
    "ui.visualization": "visualization",
    "ui.ui": "ui",
}


SCOPED_MODULE_ALIASES: Mapping[str, str] = {
    # application/cli.py
    "application.assets": "storage_services.assets",
    "application.runtime": "runtime_services.api",
    "application.queueing": "queue_services.api",
    "application.wizard": "ui.wizard",
    # core
    "core.config.models": "config.models",
    "core.method_sdk.contracts": "method_sdk.contracts",
    # dataset/input/storage implementation moves
    "dataset.dataset.discovery": "dataset.discovery",
    "input.dataset.discovery": "dataset.discovery",
    "input.input.video_geometry": "input.video_geometry",
    "storage_services.config.models": "config.models",
    "storage_services.dataset.discovery": "dataset.discovery",
    # experiment public API
    "experiment_services.config.models": "config.models",
    "experiment_services.experiment_services.identity": "experiment_services.identity",
    "experiment_services.experiment_services.manifests": "experiment_services.manifests",
    "experiment_services.experiment_services.method_identity": "experiment_services.method_identity",
    "experiment_services.observations": "observation_services.api",
    # observation public API
    "observation_services.observation_services.candidates": "observation_services.candidates",
    "observation_services.observation_services.core": "observation_services.core",
    "observation_services.observation_services.freeze": "observation_services.freeze",
    "observation_services.observation_services.resolution": "observation_services.resolution",
    # preflight public API
    "preflight_services.methods.ap02.graph_diagnostics": "methods.ap02.graph_diagnostics",
    "preflight_services.config.models": "config.models",
    "preflight_services.contracts": "core.contracts",
    "preflight_services.methods.ap02.frame_selection": "methods.ap02.frame_selection",
    "preflight_services.observation_quality": "observation_services.quality",
    "preflight_services.observations": "observation_services.api",
    "preflight_services.preflight_services.coordinator": "preflight_services.coordinator",
    "preflight_services.preflight_services.core": "preflight_services.core",
    "preflight_services.registry": "core.registry",
    # publication public API
    "publication_services.dataset.discovery": "dataset.discovery",
    "publication_services.dataset_identity": "dataset.identity",
    "publication_services.evaluation.reporting": "evaluation.reporting",
    "publication_services.experiments": "experiment_services.api",
    "publication_services.filesystem": "storage_services.filesystem",
    "publication_services.storage_layout": "storage_services.layout",
    "publication_services.publication_services.core": "publication_services.core",
    "publication_services.publication_services.dataset": "publication_services.dataset",
    "publication_services.publication_services.method": "publication_services.method",
    "publication_services.publication_services.inventory": "publication_services.inventory",
    "publication_services.publication_services.transactions": "publication_services.transactions",
    # queue public API
    "queue_services.config.models": "config.models",
    "queue_services.dataset.discovery": "dataset.discovery",
    "queue_services.dataset_identity": "dataset.identity",
    "queue_services.experiments": "experiment_services.api",
    "queue_services.filesystem": "storage_services.filesystem",
    "queue_services.methods.common.aruco_utils": "methods.common.aruco_utils",
    "queue_services.observations": "observation_services.api",
    "queue_services.runtime": "runtime_services.api",
    "queue_services.preflight": "preflight_services.api",
    "queue_services.observation_quality": "observation_services.quality",
    "queue_services.publication": "publication_services.api",
    "queue_services.storage_layout": "storage_services.layout",
    "queue_services.queue_services.common": "queue_services.common",
    "queue_services.queue_services.models": "queue_services.models",
    "queue_services.queue_services.base": "queue_services.base",
    "queue_services.queue_services.runner": "queue_services.runner",
    "queue_services.queue_services.preflight_flow": "queue_services.preflight_flow",
    "queue_services.queue_services.dataset_publication": "queue_services.dataset_publication",
    "queue_services.queue_services.evaluation": "queue_services.evaluation",
    # runtime public API + rerun implementation
    "runtime_services.config.models": "config.models",
    "runtime_services.contracts": "core.contracts",
    "runtime_services.dataset.manifest": "dataset.manifest",
    "runtime_services.dataset.validation": "dataset.validation",
    "runtime_services.input.preparation": "input.preparation",
    "runtime_services.input.topics": "input.topics",
    "runtime_services.intrinsics_profiles": "input.profiles",
    "runtime_services.methods.common.aruco_utils": "methods.common.aruco_utils",
    "runtime_services.experiments": "experiment_services.api",
    "runtime_services.observations": "observation_services.api",
    "runtime_services.observation_quality": "observation_services.quality",
    "runtime_services.progress": "runtime_services.progress",
    "runtime_services.registry": "core.registry",
    "runtime_services.results": "publication_services.results",
    "runtime_services.runtime_services.common": "runtime_services.common",
    "runtime_services.runtime_services.environment": "runtime_services.environment",
    "runtime_services.runtime_services.observations": "runtime_services.observations",
    "runtime_services.runtime_services.artifacts": "runtime_services.artifacts",
    "runtime_services.runtime_services.commands": "runtime_services.commands",
    "runtime_services.runtime_services.execution": "runtime_services.execution",
    "runtime_services.dataset_identity": "dataset.identity",
    "runtime_services.publication": "publication_services.api",
    "runtime_services.queueing": "queue_services.api",
    "runtime_services.storage_layout": "storage_services.layout",
    # wizard moved into ui/
    "ui.config.models": "config.models",
    "ui.dataset.discovery": "dataset.discovery",
    "ui.doctor": "application.doctor",
    "ui.experiments": "experiment_services.api",
    "ui.input.topics": "input.topics",
    "ui.input.video_geometry": "input.video_geometry",
    "ui.intrinsics_profiles": "input.profiles",
    "ui.inventory": "dataset.inventory",
    "ui.registry": "core.registry",
    "ui.runtime": "runtime_services.api",
    "ui.observation_quality": "observation_services.quality",
    "ui.observations": "observation_services.api",
    "ui.queueing": "queue_services.api",
    "ui.publication": "publication_services.api",
    "ui.ui.wizard_models": "ui.wizard_models",
    "ui.ui.wizard_prompts": "ui.wizard_prompts",
    "ui.ui.wizard_input_metadata": "ui.wizard_input_metadata",
    "ui.ui.wizard_media": "ui.wizard_media",
    "ui.ui.wizard_prepared": "ui.wizard_prepared",
    "ui.ui.wizard_real_input": "ui.wizard_real_input",
    "ui.ui.wizard_simulation_parameters": "ui.wizard_simulation_parameters",
    "ui.ui.wizard_simulation": "ui.wizard_simulation",
    "ui.ui.wizard_method_jobs": "ui.wizard_method_jobs",
    "ui.ui.wizard_method_queue": "ui.wizard_method_queue",
    "ui.ui.wizard_new_flow": "ui.wizard_new_flow",
    "ui.ui.wizard_saved_flow": "ui.wizard_saved_flow",
    "ui.ui.wizard_review": "ui.wizard_review",
    "ui.ui.result_browser": "ui.result_browser",
    "ui.ui.storage_cleanup": "ui.storage_cleanup",
    "ui.ui.intrinsics": "ui.intrinsics",
    "ui.ui.run_management": "ui.run_management",
}


def install_import_aliases() -> None:
    """Install legacy and relocation aliases without duplicating modules."""

    for alias, target in SCOPED_PACKAGE_ALIASES.items():
        _install_package_alias(_full(alias), _full(target))
    for alias, target in SCOPED_MODULE_ALIASES.items():
        _install_module_alias(_full(alias), _full(target))
    for alias, target in LEGACY_MODULE_ALIASES.items():
        _install_module_alias(_full(alias), _full(target))
