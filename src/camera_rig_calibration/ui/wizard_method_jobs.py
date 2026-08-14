"""Focused wizard responsibilities extracted from the compatibility facade."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from ..config.models import (
    ColmapSettings,
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    MethodSettings,
    MarkerSettings,
    InputSourceKind,
    ObservationQualitySettings,
    RigConfig,
    SelectionSettings,
    effective_observation_quality,
)
from ..dataset.discovery import safe_id
from ..registry import calibration_methods
from ..observation_quality import filter_observations
from ..observations import ResolvedSelections, resolve_selections

from .wizard_models import (
    MethodQueueJob,
    SelectionDatasetContext,
    WizardBack,
    _method_job_identity,
    _pending_selection_keys,
    _refresh_method_job_label,
)
from .wizard_prompts import (
    _prompt_enum_choice,
    _prompt_index,
    _public_policy_name,
    _show_input_error,
)
from .wizard_bindings import current_wizard_bindings


def _new_method_job(
    method_id: str,
    *,
    prompt_for_single_marker: bool,
    markers: MarkerSettings | None = None,
) -> MethodQueueJob:
    method = calibration_methods.get(method_id)
    methods = MethodSettings(enabled=[method_id])
    extensions: dict[str, dict] = {}
    if method_id not in {"ap01", "ap02", "ap03"}:
        try:
            extensions[method_id] = method.config_model().model_dump(mode="python")
        except ValidationError:
            extensions[method_id] = _prompt_component_options(
                method.display_name, method.config_model
            )
        methods = methods.model_copy(update={"extensions": extensions}, deep=True)
    return MethodQueueJob(
        method_id=method_id,
        label="baseline",
        methods=methods,
        markers=(markers or MarkerSettings()).model_copy(deep=True),
        observation_quality=ObservationQualitySettings(),
        colmap=ColmapSettings(),
        evaluation=EvaluationSettings(),
    )


def _method_job_summary(job: MethodQueueJob) -> str:
    pending = _pending_selection_keys(job)
    method_settings = getattr(job.methods, job.method_id, None)
    method_quality = getattr(
        method_settings, "observation_quality", None
    )
    override_count = (
        sum(
            value is not None
            for value in method_quality.model_dump(
                mode="python"
            ).values()
        )
        if method_quality is not None
        else 0
    )
    quality_text = (
        "quality=global"
        if override_count == 0
        else f"quality=global+{override_count} override(s)"
    )

    def selection_text(key: str, value: object) -> str:
        if key in pending:
            return "manual after preflight"
        contextual = [
            _selection_value(methods, key)
            for methods in job.context_methods.values()
        ]
        serialized = {
            json.dumps(item, sort_keys=True)
            for item in contextual
        }
        if len(serialized) > 1:
            return "per experiment"
        if contextual:
            value = contextual[0]
        if isinstance(value, list):
            return ",".join(map(str, value))
        return str(value)

    if job.method_id == "ap01":
        value = job.methods.ap01
        summary = (
            f"baseline={value.method_contract}, "
            f"strategy={_public_policy_name(value.advanced_strategy)}, "
            f"root={selection_text('root_camera', value.root_camera)}, "
            f"direct={value.direct_target_camera}, "
            f"ArUco={job.markers.detection_mode}, {quality_text}"
        )
        if value.advanced_strategy == "wizard_robustness_v1":
            summary += (
                f", matcher={job.colmap.matcher}, "
                f"compute={job.colmap.compute_mode}, "
                f"relay_top={value.top_moving_per_marker}, "
                f"scale_top={value.scale_top_per_marker}"
            )
        return summary
    if job.method_id == "ap02":
        value = job.methods.ap02
        return (
            f"baseline={value.method_contract}, "
            f"frames={_public_policy_name(value.frame_selection_strategy)}, "
            f"init={_public_policy_name(value.initialization_strategy)}, "
            f"nfev={value.max_nfev_static}/{value.max_nfev_moving}, "
            f"loss={value.ba_robust_loss}@{value.ba_robust_loss_scale_px:g}px, "
            f"ref_mode={value.reference_marker_selection_mode}, "
            f"ref={selection_text('ap02_reference', value.reference_marker_id)}, "
            f"frames=marker:{value.top_per_marker}/pair:"
            f"{value.top_per_marker_pair}/total:{value.maximum_total_frames}, "
            f"ArUco={job.markers.detection_mode}, "
            f"{quality_text}"
        )
    if job.method_id == "ap03":
        single = job.methods.ap03.single
        multi = job.methods.ap03.multi
        summary = (
            f"baseline={job.methods.ap03.method_contract}, "
            f"features={_public_policy_name(job.methods.ap03.feature_limit_policy)}, "
            f"scale_input={_public_policy_name(job.methods.ap03.scale_input_policy)}, "
            f"single={selection_text('single_marker', single.scale_marker_id)}, "
            f"multi={selection_text('multi_markers', multi.marker_ids)}, "
            f"matcher={job.colmap.matcher}, ArUco={job.markers.detection_mode}; "
            f"{quality_text}; one COLMAP, multi primary"
        )
        if job.methods.ap03.scale_input_policy == "wizard_filtered_observations_v1":
            summary += (
                ", scale_top="
                f"{job.methods.ap03.scale.maximum_observations_per_marker}"
            )
        return summary
    payload = job.methods.extensions.get(job.method_id, {})
    return (
        "options="
        + yaml.safe_dump(payload, default_flow_style=True).strip()
    )


def _show_method_queue(console: Console, jobs: list[MethodQueueJob]) -> None:
    for job in jobs:
        _refresh_method_job_label(job)
    table = Table(title="Calibration queue — one reproducible result run per row")
    table.add_column("#", justify="right")
    table.add_column("Run label")
    table.add_column("Method")
    table.add_column("Resolved baseline/config summary", overflow="fold")
    table.add_column("Execution")
    first_identical_row: dict[str, int] = {}
    for index, job in enumerate(jobs, 1):
        identity = _method_job_identity(job)
        duplicate_of = first_identical_row.get(identity)
        if duplicate_of is None:
            first_identical_row[identity] = index
        table.add_row(
            str(index),
            job.label,
            calibration_methods.get(job.method_id).display_name,
            _method_job_summary(job),
            (
                f"exact duplicate of row {duplicate_of}; skipped after first"
                if duplicate_of is not None
                else "independent"
            ),
        )
    console.print(table)
    console.print(
        "[dim]Names are generated from deviations to baseline. Exact duplicate "
        "configurations do not recompute or overwrite an existing result.[/dim]"
    )


def _ids_text(value: str | list[int]) -> str:
    return value if value == "auto" else ",".join(map(str, value))


def _parse_ids(value: str) -> str | list[int]:
    value = value.strip()
    if value.lower() == "auto":
        return "auto"
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("enter 'auto' or at least one marker ID")
    return result


GUIDED_SELECTION_KEYS = frozenset(
    {"root_camera", "ap02_reference", "single_marker", "multi_markers"}
)


def _methods_with_automatic_selections(methods: MethodSettings) -> MethodSettings:
    ap03 = methods.ap03.model_copy(
        update={
            "single": methods.ap03.single.model_copy(
                update={"scale_marker_id": "auto"}
            ),
            "multi": methods.ap03.multi.model_copy(
                update={"marker_ids": "auto"}
            ),
        },
        deep=True,
    )
    return methods.model_copy(
        update={
            "ap01": methods.ap01.model_copy(
                update={"root_camera": "auto"}
            ),
            "ap02": methods.ap02.model_copy(
                update=(
                    {}
                    if methods.ap02.reference_marker_selection_mode
                    == "baseline"
                    else {
                        "reference_marker_id": "auto",
                        "reference_marker_selection_mode": "auto",
                    }
                )
            ),
            "ap03": ap03,
        },
        deep=True,
    )


def _selection_value(methods: MethodSettings, key: str) -> object:
    if key == "root_camera":
        return methods.ap01.root_camera
    if key == "ap02_reference":
        return methods.ap02.reference_marker_id
    if key == "single_marker":
        return methods.ap03.single.scale_marker_id
    if key == "multi_markers":
        return methods.ap03.multi.marker_ids
    raise KeyError(key)


def _methods_with_selection(
    methods: MethodSettings, key: str, value: object
) -> MethodSettings:
    if key == "root_camera":
        return methods.model_copy(
            update={
                "ap01": methods.ap01.model_copy(
                    update={"root_camera": str(value)}
                )
            },
            deep=True,
        )
    if key == "ap02_reference":
        return methods.model_copy(
            update={
                "ap02": methods.ap02.model_copy(
                    update={"reference_marker_id": value}
                )
            },
            deep=True,
        )
    if key == "single_marker":
        ap03 = methods.ap03.model_copy(
            update={
                "single": methods.ap03.single.model_copy(
                    update={"scale_marker_id": value}
                )
            },
            deep=True,
        )
        return methods.model_copy(update={"ap03": ap03}, deep=True)
    if key == "multi_markers":
        ap03 = methods.ap03.model_copy(
            update={
                "multi": methods.ap03.multi.model_copy(
                    update={"marker_ids": value}
                )
            },
            deep=True,
        )
        return methods.model_copy(update={"ap03": ap03}, deep=True)
    raise KeyError(key)


def _selection_mode_for_methods(
    method_id: str, methods: MethodSettings
) -> SelectionSettings:
    explicit = (
        method_id == "ap01"
        and methods.ap01.root_camera != "auto"
        or method_id == "ap02"
        and (
            methods.ap02.reference_marker_selection_mode
            in {"baseline", "explicit"}
            or (
                methods.ap02.reference_marker_selection_mode == "manual"
                and methods.ap02.reference_marker_id != "auto"
            )
        )
        or method_id == "ap03"
        and methods.ap03.single.scale_marker_id != "auto"
        and methods.ap03.multi.marker_ids != "auto"
    )
    return SelectionSettings(mode="explicit" if explicit else "auto")


def _job_methods(
    job: MethodQueueJob, context_key: str | None = None
) -> MethodSettings:
    if context_key is not None and context_key in job.context_methods:
        return job.context_methods[context_key]
    return job.methods


def _job_selection(
    job: MethodQueueJob, context_key: str | None = None
) -> SelectionSettings:
    if context_key is not None and context_key in job.context_selections:
        return job.context_selections[context_key]
    return job.selection


def _refresh_job_selection_mode(
    job: MethodQueueJob, context_key: str | None = None
) -> None:
    if context_key is None:
        pending = job.deferred_selection_keys
        methods = job.methods
        job.selection = (
            SelectionSettings(mode="review_once")
            if pending
            else _selection_mode_for_methods(job.method_id, methods)
        )
        return
    pending = job.context_deferred_selection_keys.get(context_key, set())
    methods = _job_methods(job, context_key)
    job.context_selections[context_key] = (
        SelectionSettings(mode="review_once")
        if pending
        else _selection_mode_for_methods(job.method_id, methods)
    )


def _sync_context_methods(job: MethodQueueJob) -> None:
    """Apply common method edits without losing per-dataset selections."""

    for context_key, contextual in tuple(job.context_methods.items()):
        merged = job.methods.model_copy(deep=True)
        for key in GUIDED_SELECTION_KEYS:
            merged = _methods_with_selection(
                merged, key, _selection_value(contextual, key)
            )
        job.context_methods[context_key] = MethodSettings.model_validate(
            merged.model_dump(mode="python")
        )
        _refresh_job_selection_mode(job, context_key)


def _preview_prepared_selections(
    job: MethodQueueJob,
    context: SelectionDatasetContext,
    *,
    automatic: bool = True,
) -> ResolvedSelections | None:
    observations_csv = context.observations_csv
    if observations_csv is None:
        return None
    configured_methods = _job_methods(job, context.key)
    methods = (
        _methods_with_automatic_selections(configured_methods)
        if automatic
        else configured_methods
    )
    config = RigConfig(
        dataset=DatasetSettings(
            id=safe_id(context.key),
            category=DatasetCategory.REAL_VEHICLE,
            source_kind=InputSourceKind.PREPARED,
            prepared_root=context.dataset_root,
            input_root=context.dataset_root,
        ),
        static_cameras=list(context.static_cameras),
        methods=methods,
        markers=job.markers,
        observation_quality=job.observation_quality,
        evaluation=job.evaluation,
        selection=(
            SelectionSettings(mode="auto")
            if automatic
            else _job_selection(job, context.key)
        ),
    )
    with tempfile.TemporaryDirectory(prefix="rigcal_selection_preview_") as temp:
        filtered = filter_observations(
            observations_csv,
            Path(temp),
            job_id=f"preview_{safe_id(context.key)}_{job.method_id}",
            marker_settings=job.markers,
            quality=effective_observation_quality(
                config, job.method_id
            )[0],
        )
        if filtered.accepted_count == 0:
            raise ValueError(
                f"{context.display_name}: no observation survives the "
                "current ArUco and quality filters"
            )
        return resolve_selections(
            config, filtered.filtered_observations_root
        )


def _validate_prepared_job_selections(
    jobs: list[MethodQueueJob],
    contexts: tuple[SelectionDatasetContext, ...],
) -> None:
    for job in jobs:
        for context in contexts:
            if context.observations_csv is None:
                continue
            methods = _job_methods(job, context.key)
            configured = (
                methods.ap01.root_camera != "auto"
                if job.method_id == "ap01"
                else (
                    methods.ap02.reference_marker_selection_mode
                    in {"baseline", "explicit"}
                    or (
                        methods.ap02.reference_marker_selection_mode
                        == "manual"
                        and methods.ap02.reference_marker_id != "auto"
                    )
                )
                if job.method_id == "ap02"
                else (
                    methods.ap03.single.scale_marker_id != "auto"
                    or methods.ap03.multi.marker_ids != "auto"
                )
                if job.method_id == "ap03"
                else False
            )
            if not configured:
                continue
            try:
                resolved = _preview_prepared_selections(
                    job, context, automatic=False
                )
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"{context.display_name}/{job.label}: the selected "
                    f"root/marker is no longer compatible after the current "
                    f"filters: {exc}"
                ) from exc
            if resolved is None:
                raise ValueError(
                    f"{context.display_name}/{job.label}: no observation "
                    "survives the current filters"
                )


def _selection_candidates(
    resolved: ResolvedSelections, key: str
) -> list[dict[str, object]]:
    if key == "root_camera":
        return list(
            resolved.payload["ap01_root_camera"]["candidates"]
        )
    if key == "ap02_reference":
        return list(
            resolved.payload["ap02_reference_marker"]["candidates"]
        )
    return list(
        resolved.payload["ap03_single_scale_marker"]["candidates"]
    )


def _candidate_compatible(item: dict[str, object], key: str) -> bool:
    if key == "root_camera":
        return bool(item.get("compatible", True))
    if key == "ap02_reference":
        return bool(item.get("compatible", False))
    return bool(item.get("ap03_compatible", False))


def _show_guided_candidates(
    console: Console,
    *,
    context: SelectionDatasetContext,
    resolved: ResolvedSelections,
    key: str,
) -> list[dict[str, object]]:
    candidates = _selection_candidates(resolved, key)
    table = Table(
        title=f"{context.display_name} — compatible selection candidates"
    )
    table.add_column("#", justify="right")
    table.add_column("Camera" if key == "root_camera" else "Marker")
    table.add_column("Coverage")
    table.add_column("Moving")
    table.add_column("Accepted")
    table.add_column("Median PnP RMSE")
    table.add_column("Status")
    for index, item in enumerate(candidates, 1):
        if key == "root_camera":
            coverage = (
                f"{len(item.get('reachable_cameras', []))}/"
                f"{len(context.static_cameras)}"
            )
            moving = str(len(item.get("moving_bridges", [])))
            accepted = str(item.get("observations", 0))
        else:
            coverage = (
                f"{item.get('combined_graph_reachable_static_count', 0)}/"
                f"{len(context.static_cameras)}"
            )
            moving = str(item.get("moving_frames", 0))
            accepted = str(item.get("accepted_observations", 0))
        rmse = item.get("median_pnp_reprojection_rmse_px")
        status = (
            "recommended"
            if item.get("recommended")
            else "compatible"
            if _candidate_compatible(item, key)
            else "not compatible"
        )
        table.add_row(
            str(index),
            str(item["id"]),
            coverage,
            moving,
            accepted,
            f"{float(rmse):.2f} px" if rmse is not None else "unknown",
            status,
        )
    console.print(table)
    return candidates


def _prompt_guided_candidate(
    candidates: list[dict[str, object]],
    key: str,
) -> object:
    compatible = {
        index: item
        for index, item in enumerate(candidates, 1)
        if _candidate_compatible(item, key)
    }
    selectable = (
        {
            index: item
            for index, item in enumerate(candidates, 1)
        }
        if key == "ap02_reference"
        else compatible
    )
    if not selectable:
        raise ValueError("No compatible selection candidate is available")
    if key == "multi_markers":
        default = ",".join(map(str, compatible))
        while True:
            raw = typer.prompt(
                "Compatible table numbers, comma-separated, or all "
                "(b = back)",
                default=default,
            ).strip().lower()
            if raw in {"b", "back"}:
                raise WizardBack()
            if raw == "all":
                return [int(item["id"]) for item in compatible.values()]
            try:
                numbers = list(
                    dict.fromkeys(
                        int(value.strip())
                        for value in raw.split(",")
                        if value.strip()
                    )
                )
            except ValueError:
                numbers = []
            if numbers and all(number in compatible for number in numbers):
                return [
                    int(compatible[number]["id"]) for number in numbers
                ]
            _show_input_error(
                "Choose compatible table numbers or enter 'all'."
            )
    recommended = next(
        (
            index
            for index, item in selectable.items()
            if item.get("recommended")
        ),
        next(iter(selectable)),
    )
    while True:
        selected = _prompt_index(
            (
                "Detected marker table number (0/b = back)"
                if key == "ap02_reference"
                else "Compatible table number (0/b = back)"
            ),
            default=recommended,
            maximum=len(candidates),
        )
        if selected is None:
            raise WizardBack()
        if selected in selectable:
            candidate = selectable[selected]
            if (
                key == "ap02_reference"
                and not _candidate_compatible(candidate, key)
                and not typer.confirm(
                    "This detected marker is not compatible with a complete "
                    "AP02 graph under the current filters. Continue with the "
                    "documented partial/diagnostic selection?",
                    default=False,
                )
            ):
                continue
            return candidate["id"]
        _show_input_error(
            "That row is not compatible with this method configuration."
        )


def _configure_guided_selection(
    console: Console,
    job: MethodQueueJob,
    *,
    key: str,
    label: str,
    contexts: tuple[SelectionDatasetContext, ...],
    requested_mode: str | None = None,
) -> None:
    hooks = current_wizard_bindings()
    _preview_prepared_selections = hooks.preview_prepared_selections
    _refresh_method_job_label = hooks.refresh_method_job_label
    configured_values = [
        _selection_value(job.methods, key),
        *(
            _selection_value(methods, key)
            for methods in job.context_methods.values()
        ),
    ]
    current_mode = (
        "manual"
        if (
            job.selection.mode == "review_once"
            or any(
                selection.mode == "review_once"
                for selection in job.context_selections.values()
            )
            or any(
                value != "auto"
                for value in configured_values
            )
        )
        else "auto"
    )
    mode = requested_mode or _prompt_enum_choice(
        label,
        current_mode,
        (
            (
                "auto",
                "use the deterministic recommendation and continue without a pause",
            ),
            (
                "manual",
                "choose from compatible candidates now or after preflight",
            ),
        ),
    )
    if mode not in {"auto", "manual"}:
        raise ValueError(f"Unsupported guided selection mode: {mode}")
    if mode == "auto":
        job.methods = _methods_with_selection(job.methods, key, "auto")
        job.deferred_selection_keys.discard(key)
        _refresh_job_selection_mode(job)
        for context in contexts:
            methods = _methods_with_selection(
                _job_methods(job, context.key), key, "auto"
            )
            job.context_methods[context.key] = methods
            job.context_deferred_selection_keys.setdefault(
                context.key, set()
            ).discard(key)
            _refresh_job_selection_mode(job, context.key)
        return

    if not contexts:
        job.methods = _methods_with_selection(job.methods, key, "auto")
        job.deferred_selection_keys.add(key)
        _refresh_job_selection_mode(job)
        typer.echo(
            "Manual selection scheduled after ArUco detection and the "
            "job-specific observation-quality preflight."
        )
        return

    immediate_values: list[object] = []
    for context in contexts:
        resolved = _preview_prepared_selections(job, context)
        if resolved is None:
            methods = _methods_with_selection(
                _job_methods(job, context.key), key, "auto"
            )
            job.context_methods[context.key] = methods
            job.context_deferred_selection_keys.setdefault(
                context.key, set()
            ).add(key)
            _refresh_job_selection_mode(job, context.key)
            typer.echo(
                f"{context.display_name}: complete observations are unavailable; "
                "manual selection will occur after preflight."
            )
            continue
        candidates = _show_guided_candidates(
            console,
            context=context,
            resolved=resolved,
            key=key,
        )
        chosen = _prompt_guided_candidate(candidates, key)
        immediate_values.append(chosen)
        methods = _methods_with_selection(
            _job_methods(job, context.key), key, chosen
        )
        job.context_methods[context.key] = methods
        job.context_deferred_selection_keys.setdefault(
            context.key, set()
        ).discard(key)
        _refresh_job_selection_mode(job, context.key)

    if len(contexts) == 1 and immediate_values:
        job.methods = _methods_with_selection(
            job.methods, key, immediate_values[0]
        )
        job.deferred_selection_keys.discard(key)
        _refresh_job_selection_mode(job)

def _prompt_component_options(display_name: str, model_class: type) -> dict:
    try:
        defaults = model_class().model_dump(mode="python")
    except ValidationError:
        defaults = {}
    default_text = yaml.safe_dump(defaults, default_flow_style=True).strip()
    while True:
        value = typer.prompt(
            f"{display_name} options as a YAML mapping",
            default=default_text,
        )
        try:
            payload = yaml.safe_load(value)
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                raise ValueError("options must be a mapping")
            return model_class.model_validate(payload).model_dump(mode="python")
        except (ValidationError, ValueError, yaml.YAMLError) as exc:
            typer.echo(f"Invalid options: {exc}")


__all__ = [
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
]
