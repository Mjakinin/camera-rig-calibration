"""Interactive editor for one method-queue row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ..config.models import (
    ColmapSettings,
    MarkerSettings,
    MethodSettings,
    ObservationQualitySettings,
)
from ..registry import calibration_methods


@dataclass(frozen=True)
class EditorBindings:
    """Wizard-owned hooks used by policies and navigation."""

    setting_rows: Callable[..., Any]
    format_setting_value: Callable[[object], str]
    public_policy_name: Callable[[object], object]
    clear_terminal: Callable[[], None]
    show_input_error: Callable[[str], None]
    configure_guided_selection: Callable[..., None]
    prompt_enum_choice: Callable[..., str]
    job_methods: Callable[..., Any]
    optional_positive_int: Callable[[str], int | None]
    parse_ids: Callable[[str], str | list[int]]
    refresh_job_selection_mode: Callable[..., None]
    refresh_method_job_label: Callable[[Any], str]
    sync_context_methods: Callable[[Any], None]
    wizard_back: type[Exception]
    guided_selection_keys: frozenset[str]


def edit_method_job(
    console: Console,
    job: Any,
    *,
    groups: set[str] | frozenset[str],
    title: str | None,
    selection_contexts: tuple[Any, ...],
    bindings: EditorBindings,
) -> Any:
    """Edit a job while delegating product-policy hooks to the facade."""
    _setting_rows = bindings.setting_rows
    _format_setting_value = bindings.format_setting_value
    _public_policy_name = bindings.public_policy_name
    _clear_terminal = bindings.clear_terminal
    _show_input_error = bindings.show_input_error
    _configure_guided_selection = bindings.configure_guided_selection
    _prompt_enum_choice = bindings.prompt_enum_choice
    _job_methods = bindings.job_methods
    _optional_positive_int = bindings.optional_positive_int
    _parse_ids = bindings.parse_ids
    _refresh_job_selection_mode = bindings.refresh_job_selection_mode
    _refresh_method_job_label = bindings.refresh_method_job_label
    _sync_context_methods = bindings.sync_context_methods
    WizardBack = bindings.wizard_back
    GUIDED_SELECTION_KEYS = bindings.guided_selection_keys
    while True:
        rows = _setting_rows(job, groups)
        table = Table(title=title or f"Method settings for {job.label}")
        table.add_column("#", justify="right")
        table.add_column("Group")
        table.add_column("Parameter")
        table.add_column("Current")
        table.add_column("Default")
        table.add_column("Meaning", overflow="fold")
        for index, (_, group, label, current, default, meaning) in enumerate(rows, 1):
            table.add_row(
                str(index),
                group,
                label,
                _format_setting_value(_public_policy_name(current)),
                _format_setting_value(_public_policy_name(default)),
                meaning,
            )
        console.print(table)
        selection = typer.prompt(
            "Setting numbers to change together "
            "(comma-separated; Enter = keep all; b = back)",
            default="",
            show_default=False,
        ).strip()
        if not selection:
            return job
        if selection.lower() in {"0", "b", "back"}:
            _clear_terminal()
            return job
        try:
            indices = [int(value.strip()) for value in selection.split(",")]
        except ValueError:
            _show_input_error(
                "Use comma-separated setting numbers, for example 2,5,9."
            )
            continue
        if not indices or min(indices) < 1 or max(indices) > len(rows):
            _show_input_error(
                f"Choose setting numbers between 1 and {len(rows)}."
            )
            continue
        back_to_table = False
        for index in dict.fromkeys(indices):
            key, _, label, current, _, _ = rows[index - 1]
            if key == "colmap_na":
                typer.echo("COLMAP is not applicable to AP02.")
                continue
            try:
                if key in GUIDED_SELECTION_KEYS:
                    _configure_guided_selection(
                        console,
                        job,
                        key=key,
                        label=label,
                        contexts=selection_contexts,
                    )
                    continue
                if key == "ap02_reference_display":
                    typer.echo(
                        "Change the reference-marker selection mode in the "
                        "preceding row; marker IDs are never entered freely."
                    )
                    continue
                if key == "evaluation_anchor":
                    value = _prompt_enum_choice(
                        label,
                        (
                            "manual"
                            if job.evaluation.anchor_selection_mode
                            == "review_once"
                            else "auto"
                        ),
                        (
                            (
                                "auto",
                                "freeze the strongest repeat-supported marker without pausing",
                            ),
                            (
                                "manual",
                                "show every detected marker once after preflight",
                            ),
                        ),
                    )
                elif key == "ap02_reference_mode":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            (
                                "baseline",
                                "canonical baseline contract; force marker 14",
                            ),
                            (
                                "auto",
                                "use the deterministic preflight recommendation",
                            ),
                            (
                                "manual",
                                "show all detected marker IDs once after preflight",
                            ),
                        ),
                    )
                elif key == "ap01_advanced_strategy":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("legacy_main_v1", "standard Direct/Relay behavior"),
                            ("wizard_robustness_v1", "configurable caps and consensus gates"),
                        ),
                    )
                elif key == "ap02_frame_strategy":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("legacy_smart_v1", "smart selection at the BA boundary"),
                            ("wizard_graph_preserving_v1", "advanced pre-initialization graph selection"),
                        ),
                    )
                elif key == "ap02_initialization_strategy":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("legacy_maximum_bottleneck_v1", "deterministic maximum-frontier tree"),
                            ("wizard_maximum_bottleneck_v2", "advanced path-level tie strategy"),
                            ("unweighted_bfs_diagnostic", "diagnostic unweighted breadth-first tree"),
                        ),
                    )
                elif key == "ap02_edge_weight_strategy":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("legacy_observation_quality_v1", "geometric observation quality"),
                            ("wizard_selection_score_v2", "advanced shared quality score"),
                        ),
                    )
                elif key == "ap02_reprojection_model":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("legacy_pinhole_v1", "zero-distortion pinhole projection"),
                            ("distortion_aware_v1", "advanced camera-info distortion projection"),
                        ),
                    )
                elif key == "ap03_feature_limit_policy":
                    value = _prompt_enum_choice(
                        label, str(current),
                        (
                            ("legacy_colmap_defaults_v1", "leave SIFT limits unset"),
                            ("wizard_explicit_limits_v1", "apply configured AP03 SIFT limits"),
                        ),
                    )
                elif key == "ap03_scale_input_policy":
                    value = _prompt_enum_choice(
                        label, str(current),
                        (
                            ("legacy_registered_image_redetection_v1", "re-detect every registered image"),
                            ("wizard_filtered_observations_v1", "gate re-detections through filtered observations"),
                        ),
                    )
                elif key == "matcher":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            (
                                "exhaustive",
                                "compare every image pair; baseline and best for unordered captures",
                            ),
                            (
                                "sequential",
                                "compare temporal neighbors; faster for ordered video frames",
                            ),
                        ),
                    )
                elif key == "compute_mode":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("cpu_baseline", "reproducible CPU baseline"),
                            ("gpu", "require a compatible GPU or fail preflight"),
                            ("auto", "resolve from available hardware"),
                        ),
                    )
                elif key == "ba_loss":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("soft_l1", "smooth robust baseline loss"),
                            ("huber", "piecewise robust loss"),
                            ("linear", "plain least squares without robust downweighting"),
                        ),
                    )
                elif key == "detection_mode":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            (
                                "baseline",
                                "unchanged OpenCV default detector",
                            ),
                            (
                                "subpixel_refined",
                                "baseline candidates with subpixel corner refinement",
                            ),
                            (
                                "high_sensitivity",
                                "confirmed two-gamma search for small or dark markers",
                            ),
                        ),
                    )
                else:
                    prompt_default = _format_setting_value(current)
                    if key in {
                        "quality_override_reprojection",
                        "quality_override_area",
                        "quality_override_positive_depth",
                        "quality_override_distance",
                    }:
                        override_field = {
                            "quality_override_reprojection": (
                                "maximum_pnp_reprojection_error_px"
                            ),
                            "quality_override_area": (
                                "minimum_marker_area_ratio"
                            ),
                            "quality_override_positive_depth": (
                                "require_positive_depth"
                            ),
                            "quality_override_distance": (
                                "maximum_marker_distance_m"
                            ),
                        }[key]
                        method_settings = getattr(
                            job.methods, job.method_id
                        )
                        if (
                            getattr(
                                method_settings.observation_quality,
                                override_field,
                            )
                            is None
                        ):
                            prompt_default = "inherit"
                    value = typer.prompt(
                        f"{label} (b = back)",
                        default=prompt_default,
                    ).strip()
                    if value.lower() in {"b", "back"}:
                        raise WizardBack()
            except WizardBack:
                _clear_terminal()
                back_to_table = True
                break
            if key == "evaluation_enabled":
                job.evaluation = job.evaluation.model_copy(
                    update={"enabled": _bool_value(value)}
                )
            elif key == "evaluation_anchor":
                job.evaluation = job.evaluation.model_copy(
                    update={
                        "anchor_marker_id": "auto",
                        "anchor_selection_mode": (
                            "review_once"
                            if value.lower() == "manual"
                            else "auto"
                        ),
                    }
                )
            elif key == "ap02_reference_mode":
                ap02_update: dict[str, object] = {
                    "reference_marker_selection_mode": value,
                    "reference_marker_id": (
                        14 if value == "baseline" else "auto"
                    ),
                }
                ap02 = job.methods.ap02.model_copy(update=ap02_update)
                job.methods = job.methods.model_copy(
                    update={"ap02": ap02}, deep=True
                )
                if value == "manual":
                    job.deferred_selection_keys.add("ap02_reference")
                else:
                    job.deferred_selection_keys.discard("ap02_reference")
                _refresh_job_selection_mode(job)
                for context in selection_contexts:
                    contextual = _job_methods(job, context.key)
                    contextual_ap02 = contextual.ap02.model_copy(
                        update=ap02_update
                    )
                    job.context_methods[context.key] = (
                        contextual.model_copy(
                            update={"ap02": contextual_ap02}, deep=True
                        )
                    )
                    pending = job.context_deferred_selection_keys.setdefault(
                        context.key, set()
                    )
                    if value == "manual":
                        pending.add("ap02_reference")
                    else:
                        pending.discard("ap02_reference")
                    _refresh_job_selection_mode(job, context.key)
                if value == "manual":
                    _configure_guided_selection(
                        console,
                        job,
                        key="ap02_reference",
                        label="Reference marker",
                        contexts=selection_contexts,
                        requested_mode="manual",
                    )
            elif key == "ap01_advanced_strategy":
                field = "advanced_strategy"
                updated = value
                ap01 = job.methods.ap01.model_copy(update={field: updated})
                job.methods = job.methods.model_copy(
                    update={"ap01": ap01}, deep=True
                )
            elif key in {
                "ap03_feature_limit_policy",
                "ap03_scale_input_policy",
                "ap03_minimum_marker_area",
            }:
                field = {
                    "ap03_feature_limit_policy": "feature_limit_policy",
                    "ap03_scale_input_policy": "scale_input_policy",
                    "ap03_minimum_marker_area": "minimum_marker_area_px2",
                }[key]
                typed = float(value) if key == "ap03_minimum_marker_area" else value
                ap03 = job.methods.ap03.model_copy(update={field: typed})
                job.methods = job.methods.model_copy(
                    update={"ap03": ap03}, deep=True
                )
            elif key in {
                "ap02_frame_strategy",
                "ap02_initialization_strategy",
                "ap02_edge_weight_strategy",
                "ap02_reprojection_model",
            }:
                field = {
                    "ap02_frame_strategy": "frame_selection_strategy",
                    "ap02_initialization_strategy": "initialization_strategy",
                    "ap02_edge_weight_strategy": "graph_edge_weight_strategy",
                    "ap02_reprojection_model": "reprojection_model",
                }[key]
                ap02 = job.methods.ap02.model_copy(update={field: value})
                job.methods = job.methods.model_copy(
                    update={"ap02": ap02}, deep=True
                )
            elif key == "ap01_direct_target":
                ap01 = job.methods.ap01.model_copy(
                    update={"direct_target_camera": str(value).strip()}
                )
                job.methods = job.methods.model_copy(
                    update={"ap01": ap01}, deep=True
                )
            elif key in {
                "evaluation_reprojection",
                "evaluation_inliers",
                "evaluation_ransac",
                "evaluation_angle",
                "evaluation_max_observations",
            }:
                field = {
                    "evaluation_reprojection": "reprojection_threshold_px",
                    "evaluation_inliers": "minimum_inliers",
                    "evaluation_ransac": "ransac_iterations",
                    "evaluation_angle": "minimum_triangulation_angle_deg",
                    "evaluation_max_observations": (
                        "maximum_moving_observations_per_marker"
                    ),
                }[key]
                typed_value: float | int = (
                    float(value)
                    if key in {"evaluation_reprojection", "evaluation_angle"}
                    else int(value)
                )
                job.evaluation = job.evaluation.model_copy(
                    update={field: typed_value}
                )
            elif key == "accepted_ids":
                parsed = _parse_ids(
                    "auto"
                    if value.lower() in {"all detected ids", "all_detected"}
                    else value
                )
                job.markers = job.markers.model_copy(
                    update={
                        "accepted_ids": (
                            "all_detected" if parsed == "auto" else parsed
                        )
                    }
                )
            elif key == "dictionary":
                job.markers = job.markers.model_copy(update={"dictionary": value})
            elif key == "detection_mode":
                job.markers = job.markers.model_copy(
                    update={"detection_mode": value}
                )
            elif key == "marker_length":
                job.markers = job.markers.model_copy(update={"length_m": float(value)})
            elif key in {
                "quality_reprojection",
                "quality_area",
                "quality_positive_depth",
                "quality_distance",
            }:
                field = {
                    "quality_reprojection": "maximum_pnp_reprojection_error_px",
                    "quality_area": "minimum_marker_area_ratio",
                    "quality_positive_depth": "require_positive_depth",
                    "quality_distance": "maximum_marker_distance_m",
                }[key]
                typed: str | float | bool
                if key == "quality_positive_depth":
                    typed = _bool_value(value)
                else:
                    typed = (
                        "disabled"
                        if value.lower() == "disabled"
                        else float(value)
                    )
                job.observation_quality = job.observation_quality.model_copy(
                    update={field: typed}
                )
            elif key in {
                "quality_override_reprojection",
                "quality_override_area",
                "quality_override_positive_depth",
                "quality_override_distance",
            }:
                field = {
                    "quality_override_reprojection": (
                        "maximum_pnp_reprojection_error_px"
                    ),
                    "quality_override_area": "minimum_marker_area_ratio",
                    "quality_override_positive_depth": (
                        "require_positive_depth"
                    ),
                    "quality_override_distance": (
                        "maximum_marker_distance_m"
                    ),
                }[key]
                normalized = value.strip().lower()
                if normalized.startswith("inherit"):
                    override_value: str | float | bool | None = None
                elif key == "quality_override_positive_depth":
                    override_value = _bool_value(value)
                elif normalized == "disabled":
                    override_value = "disabled"
                else:
                    override_value = float(value)
                method_settings = getattr(
                    job.methods, job.method_id
                )
                quality_override = (
                    method_settings.observation_quality.model_copy(
                        update={field: override_value}
                    )
                )
                updated_method = method_settings.model_copy(
                    update={"observation_quality": quality_override}
                )
                job.methods = job.methods.model_copy(
                    update={job.method_id: updated_method},
                    deep=True,
                )
            elif key in {
                "ap01_top_moving",
                "ap01_scale_top",
            }:
                field = {
                    "ap01_top_moving": "top_moving_per_marker",
                    "ap01_scale_top": "scale_top_per_marker",
                }[key]
                ap01 = job.methods.ap01.model_copy(
                    update={field: _optional_positive_int(value)}
                )
                job.methods = job.methods.model_copy(
                    update={"ap01": ap01}, deep=True
                )
            elif key.startswith("ap01_direct_"):
                field = {
                    "ap01_direct_markers": "minimum_independent_markers",
                    "ap01_direct_inlier_ratio": "minimum_inlier_ratio",
                    "ap01_direct_translation": (
                        "maximum_translation_dispersion_m"
                    ),
                    "ap01_direct_rotation": (
                        "maximum_rotation_dispersion_deg"
                    ),
                }[key]
                typed_gate_value: int | float = (
                    int(value)
                    if key == "ap01_direct_markers"
                    else float(value)
                )
                gate = job.methods.ap01.direct_quality_gate.model_copy(
                    update={field: typed_gate_value}
                )
                ap01 = job.methods.ap01.model_copy(
                    update={"direct_quality_gate": gate}
                )
                job.methods = job.methods.model_copy(
                    update={"ap01": ap01}, deep=True
                )
            elif key.startswith("ap01_relay_"):
                field = {
                    "ap01_relay_inlier_ratio": "minimum_inlier_ratio",
                    "ap01_relay_translation": (
                        "maximum_translation_dispersion_m"
                    ),
                    "ap01_relay_rotation": (
                        "maximum_rotation_dispersion_deg"
                    ),
                }[key]
                gate = job.methods.ap01.relay_quality_gate.model_copy(
                    update={field: float(value)}
                )
                ap01 = job.methods.ap01.model_copy(
                    update={"relay_quality_gate": gate}
                )
                job.methods = job.methods.model_copy(
                    update={"ap01": ap01}, deep=True
                )
            elif key.startswith("ap01_consistency_"):
                field = {
                    "ap01_consistency_translation": (
                        "maximum_translation_disagreement_m"
                    ),
                    "ap01_consistency_rotation": (
                        "maximum_rotation_disagreement_deg"
                    ),
                }[key]
                consistency = (
                    job.methods.ap01.direct_relay_consistency.model_copy(
                        update={field: float(value)}
                    )
                )
                ap01 = job.methods.ap01.model_copy(
                    update={"direct_relay_consistency": consistency}
                )
                job.methods = job.methods.model_copy(
                    update={"ap01": ap01}, deep=True
                )
            elif key in {
                "ap02_reference_frames",
                "ap02_top_marker",
                "ap02_top_pair",
                "ap02_total_frames",
            }:
                field = {
                    "ap02_reference_frames": (
                        "reference_marker_maximum_frames"
                    ),
                    "ap02_top_marker": "top_per_marker",
                    "ap02_top_pair": "top_per_marker_pair",
                    "ap02_total_frames": "maximum_total_frames",
                }[key]
                ap02 = job.methods.ap02.model_copy(
                    update={field: _optional_positive_int(value)}
                )
                job.methods = job.methods.model_copy(
                    update={"ap02": ap02}, deep=True
                )
            elif key in {"max_nfev_static", "max_nfev_moving"}:
                field = {
                    "max_nfev_static": "static_only_ba_max_function_evaluations",
                    "max_nfev_moving": "combined_ba_max_function_evaluations",
                }[key]
                job.methods = job.methods.model_copy(update={"ap02": job.methods.ap02.model_copy(update={field: int(value)})}, deep=True)
            elif key in {"ba_loss", "ba_loss_scale"}:
                field = (
                    "ba_robust_loss"
                    if key == "ba_loss"
                    else "ba_robust_loss_scale_px"
                )
                typed = value if key == "ba_loss" else float(value)
                job.methods = job.methods.model_copy(
                    update={
                        "ap02": job.methods.ap02.model_copy(
                            update={field: typed}
                        )
                    },
                    deep=True,
                )
            elif key in {
                "scale_reprojection",
                "scale_ransac",
                "scale_inliers",
                "scale_max_observations",
            }:
                if key.endswith("reprojection"):
                    field = "reprojection_threshold_px"
                elif key.endswith("ransac"):
                    field = "ransac_iterations"
                elif key.endswith("inliers"):
                    field = "minimum_inliers"
                else:
                    field = "maximum_observations_per_marker"
                typed = (
                    float(value)
                    if field == "reprojection_threshold_px"
                    else (
                        _optional_positive_int(value)
                        if field == "maximum_observations_per_marker"
                        else int(value)
                    )
                )
                ap03 = job.methods.ap03.model_copy(
                    update={
                        "scale": job.methods.ap03.scale.model_copy(
                            update={field: typed}
                        )
                    },
                    deep=True,
                )
                job.methods = job.methods.model_copy(update={"ap03": ap03}, deep=True)
            elif key in {"colmap_executable", "matcher"}:
                field = "executable" if key == "colmap_executable" else key
                job.colmap = job.colmap.model_copy(update={field: value})
            elif key == "compute_mode":
                job.colmap = job.colmap.model_copy(
                    update={"compute_mode": value.lower()}
                )
            elif key == "loop_detection":
                job.colmap = job.colmap.model_copy(update={key: _bool_value(value)})
            elif key in {"mapper_matches", "maximum_image_size", "maximum_features", "sequential_overlap"}:
                field = {
                    "mapper_matches": "mapper_minimum_matches",
                    "maximum_image_size": "maximum_image_size",
                    "maximum_features": "maximum_features",
                    "sequential_overlap": "sequential_overlap",
                }[key]
                job.colmap = job.colmap.model_copy(update={field: int(value)})
            elif key in {"ap03_image_size", "ap03_features"}:
                field = "ap03_maximum_image_size" if key == "ap03_image_size" else "ap03_maximum_features"
                job.colmap = job.colmap.model_copy(update={field: None if value.lower() in {"", "none", "runner default"} else int(value)})
            elif key == "extension":
                payload = yaml.safe_load(value) or {}
                method = calibration_methods.get(job.method_id)
                validated = method.config_model.model_validate(payload).model_dump(mode="python")
                extensions = dict(job.methods.extensions)
                extensions[job.method_id] = validated
                job.methods = job.methods.model_copy(update={"extensions": extensions}, deep=True)
        if back_to_table:
            continue
        # Validate all copied models and then show the resolved values again. This also
        # reveals sequential-only settings immediately after changing the matcher.
        job.methods = MethodSettings.model_validate(job.methods.model_dump(mode="python"))
        _sync_context_methods(job)
        job.markers = MarkerSettings.model_validate(job.markers.model_dump(mode="python"))
        job.colmap = ColmapSettings.model_validate(job.colmap.model_dump(mode="python"))
        job.observation_quality = ObservationQualitySettings.model_validate(
            job.observation_quality.model_dump(mode="python")
        )
        _refresh_method_job_label(job)
        if not typer.confirm("Change more values in this menu?", default=False):
            return job


__all__ = ["EditorBindings", "edit_method_job"]
