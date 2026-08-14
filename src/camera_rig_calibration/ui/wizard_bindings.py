"""Compatibility bindings between modular UI flows and the wizard facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Hook = Callable[..., Any]


@dataclass(frozen=True)
class WizardBindings:
    """Product-policy hooks resolved after the bootstrap stack is installed."""

    choice: Hook
    prompt_index: Hook
    show_input_error: Hook
    clear_terminal: Hook
    new_method_job: Hook
    method_job_label: Hook
    refresh_method_job_label: Hook
    method_queue: Hook
    edit_method_job: Hook
    show_method_queue: Hook
    validate_prepared_job_selections: Hook
    build_simulation_batch_outcome: Hook
    job_methods: Hook
    job_selection: Hook
    setting_rows: Hook
    configure_guided_selection: Hook
    save_wizard_queue: Hook
    probe_video_geometry: Hook
    list_mcap_topics: Hook
    discover_simulation_experiments: Hook
    preview_prepared_selections: Hook
    review_selection_candidates: Hook


def current_wizard_bindings() -> WizardBindings:
    """Read hooks from the facade so installed policies remain authoritative."""
    from .. import wizard

    return WizardBindings(
        choice=wizard._choice,
        prompt_index=wizard._prompt_index,
        show_input_error=wizard._show_input_error,
        clear_terminal=wizard._clear_terminal,
        new_method_job=wizard._new_method_job,
        method_job_label=wizard._method_job_label,
        refresh_method_job_label=wizard._refresh_method_job_label,
        method_queue=wizard._method_queue,
        edit_method_job=wizard._edit_method_job,
        show_method_queue=wizard._show_method_queue,
        validate_prepared_job_selections=(
            wizard._validate_prepared_job_selections
        ),
        build_simulation_batch_outcome=(
            wizard._build_simulation_batch_outcome
        ),
        job_methods=wizard._job_methods,
        job_selection=wizard._job_selection,
        setting_rows=wizard._setting_rows,
        configure_guided_selection=wizard._configure_guided_selection,
        save_wizard_queue=wizard._save_wizard_queue,
        probe_video_geometry=wizard.probe_video_geometry,
        list_mcap_topics=wizard.list_mcap_topics,
        discover_simulation_experiments=(
            wizard.discover_simulation_experiments
        ),
        preview_prepared_selections=wizard._preview_prepared_selections,
        review_selection_candidates=wizard.review_selection_candidates,
    )


__all__ = ["WizardBindings", "current_wizard_bindings"]
