from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import typer

from ..dataset.discovery import safe_id


_INSTALLED = False
_CAPTURE_KEYS = (
    "settle_seconds",
    "post_pose_skip",
    "frame_timeout_seconds",
    "startup_timeout_seconds",
)


def _fresh_capture_experiment_id(
    base_id: str,
    capture_id: str | None,
    reserved_ids: Iterable[str],
) -> str:
    """Return a deterministic unique ID for a deliberately fresh capture."""

    reserved = {safe_id(value) for value in reserved_ids}
    base = safe_id(base_id)
    suffix = safe_id(capture_id or "fresh_capture")
    candidate = safe_id(f"{base}__{suffix}")
    index = 2
    while candidate in reserved:
        candidate = safe_id(f"{base}__{suffix}_{index}")
        index += 1
    return candidate


def _simulation_summary_parameters(
    parameters: dict[str, object], simulation: object
) -> dict[str, object]:
    """Complete the readable simulation vector with capture timings."""

    resolved = dict(parameters)
    for key in _CAPTURE_KEYS:
        resolved[key] = getattr(simulation, key)
    return resolved


def install_simulation_ui_consistency_policy() -> None:
    """Keep simulation navigation, IDs and summaries consistent in the real UI.

    This policy is intentionally presentation/lifecycle-only. It does not alter
    calibration algorithms, observation filters, scientific thresholds, or the
    immutable-publication guard.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from .. import wizard
    from ..ui import wizard_prompts, wizard_saved_flow, wizard_simulation

    original_prompt_index = wizard_prompts._prompt_index

    def prompt_index(
        label: str,
        *,
        default: int | None = None,
        minimum: int = 1,
        maximum: int | None = None,
    ) -> int | None:
        """Accept b/back even when the displayed default is numeric."""

        while True:
            kwargs: dict[str, object] = {
                "show_default": default is not None,
                "type": str,
            }
            if default is not None:
                kwargs["default"] = str(default)
            raw = str(typer.prompt(label, **kwargs)).strip()
            if raw.lower() in {"0", "b", "back"}:
                wizard_prompts._clear_terminal()
                return None
            try:
                value = int(raw)
            except ValueError:
                wizard_prompts._show_input_error(
                    "Enter a number, or b to go back."
                )
                continue
            if value < minimum or (
                maximum is not None and value > maximum
            ):
                wizard_prompts._show_input_error(
                    f"Choose a number from {minimum}"
                    + (
                        f" to {maximum}"
                        if maximum is not None
                        else " upward"
                    )
                    + "."
                )
                continue
            return value

    prompt_index._rigcal_simulation_ui_consistency = True  # type: ignore[attr-defined]
    wizard_prompts._prompt_index = prompt_index
    # wizard_simulation and the compatibility facade imported the old function
    # object at module import time, so update those concrete call sites too.
    if getattr(wizard_simulation, "_prompt_index", None) is original_prompt_index:
        wizard_simulation._prompt_index = prompt_index
    wizard._prompt_index = prompt_index

    original_job_from_parameters = wizard_simulation._simulation_job_from_parameters

    def simulation_job_from_parameters(
        repository_root: Path,
        parameters: dict[str, object],
        *,
        experiment_id: str,
        prepared_root: Path | None,
        source: str,
    ):
        job = original_job_from_parameters(
            repository_root,
            parameters,
            experiment_id=experiment_id,
            prepared_root=prepared_root,
            source=source,
        )
        if prepared_root is not None:
            return job

        experiments = (
            wizard_simulation.current_wizard_bindings()
            .discover_simulation_experiments(repository_root)
        )
        reserved_ids = [item.variant for item in experiments]
        current_signature = wizard_simulation._simulation_signature(
            job.parameters
        )
        repeats_existing_parameters = any(
            wizard_simulation._simulation_signature(item.parameters)
            == current_signature
            for item in experiments
        )
        collides_by_name = safe_id(job.experiment_id) in {
            safe_id(value) for value in reserved_ids
        }
        if not repeats_existing_parameters and not collides_by_name:
            return job

        unique_id = _fresh_capture_experiment_id(
            job.experiment_id,
            job.simulation.capture_id,
            reserved_ids,
        )
        return replace(job, experiment_id=unique_id)

    simulation_job_from_parameters._rigcal_simulation_ui_consistency = True  # type: ignore[attr-defined]
    wizard_simulation._simulation_job_from_parameters = simulation_job_from_parameters
    wizard._simulation_job_from_parameters = simulation_job_from_parameters

    original_show_summary = wizard_saved_flow.show_summary

    def show_summary(config, config_path: Path, console) -> None:
        original_formatter = wizard_saved_flow.format_simulation_parameters

        def formatter(parameters: dict[str, object]) -> str:
            if getattr(config.dataset.scene_type, "value", "") == "simulation":
                parameters = _simulation_summary_parameters(
                    parameters, config.simulation
                )
            return original_formatter(parameters)

        wizard_saved_flow.format_simulation_parameters = formatter
        try:
            original_show_summary(config, config_path, console)
        finally:
            wizard_saved_flow.format_simulation_parameters = original_formatter

    show_summary._rigcal_simulation_ui_consistency = True  # type: ignore[attr-defined]
    wizard_saved_flow.show_summary = show_summary
    wizard.show_summary = show_summary

    _INSTALLED = True


__all__ = [
    "_fresh_capture_experiment_id",
    "_simulation_summary_parameters",
    "install_simulation_ui_consistency_policy",
]
