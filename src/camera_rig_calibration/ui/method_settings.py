"""Declarative rows for the calibration method settings UI."""

from __future__ import annotations

import json
from typing import Any, Callable

import yaml

from ..registry import calibration_methods


SettingRow = tuple[str, str, str, object, object, str]


def _ids_text(value: str | list[int]) -> str:
    """Render a marker selection without depending on the wizard facade."""
    if value == "auto":
        return "auto"
    return ",".join(str(item) for item in value)


def build_setting_rows(
    job: Any,
    groups: set[str] | frozenset[str] | None = None,
    *,
    defaults_factory: Callable[..., Any],
    selection_value: Callable[[Any, str], object],
) -> list[SettingRow]:
    """Describe editable rows without owning wizard navigation or policies."""
    defaults = defaults_factory(job.method_id, prompt_for_single_marker=False)
    accepted = (
        "all detected IDs"
        if job.markers.accepted_ids == "all_detected"
        else ",".join(map(str, job.markers.accepted_ids))
    )
    method_quality = getattr(
        getattr(job.methods, job.method_id, None),
        "observation_quality",
        None,
    )

    def effective_override_value(field_name: str) -> object:
        value = (
            getattr(method_quality, field_name)
            if method_quality is not None
            else None
        )
        return (
            getattr(job.observation_quality, field_name)
            if value is None
            else value
        )

    def default_quality_value(field_name: str) -> object:
        return getattr(defaults.observation_quality, field_name)

    evaluation_anchor_current = (
        "manual after preflight"
        if job.evaluation.anchor_selection_mode == "review_once"
        else (
            f"marker {job.evaluation.anchor_marker_id}"
            if isinstance(job.evaluation.anchor_marker_id, int)
            else "auto"
        )
    )
    rows: list[SettingRow] = [
        ("evaluation_enabled", "COMMON EVALUATION", "Common evaluation enabled", job.evaluation.enabled, defaults.evaluation.enabled, "Disable only when no repeat-supported common marker can be frozen; calibration results remain available without cross-method RMSE."),
        ("evaluation_anchor", "COMMON EVALUATION", "Common evaluation and export anchor", evaluation_anchor_current, "auto", "Auto freezes the strongest compatible marker. Manual lists every detected ID once after preflight, including warned candidates."),
        ("evaluation_reprojection", "COMMON EVALUATION", "Evaluation reprojection threshold [px]", job.evaluation.reprojection_threshold_px, defaults.evaluation.reprojection_threshold_px, "Smaller requires tighter common post-hoc triangulation support."),
        ("evaluation_inliers", "COMMON EVALUATION", "Evaluation minimum inliers", job.evaluation.minimum_inliers, defaults.evaluation.minimum_inliers, "Higher requires more common-support observations."),
        ("evaluation_ransac", "COMMON EVALUATION", "Evaluation RANSAC iterations", job.evaluation.ransac_iterations, defaults.evaluation.ransac_iterations, "Higher tests more hypotheses and increases evaluation runtime."),
        ("evaluation_angle", "COMMON EVALUATION", "Minimum triangulation angle [deg]", job.evaluation.minimum_triangulation_angle_deg, defaults.evaluation.minimum_triangulation_angle_deg, "Larger rejects weak-baseline triangulation geometry."),
        ("evaluation_max_observations", "COMMON EVALUATION", "Maximum moving observations per marker", job.evaluation.maximum_moving_observations_per_marker, defaults.evaluation.maximum_moving_observations_per_marker, "Caps deterministic evaluation work per marker."),
        ("detection_mode", "QUEUE-WIDE ARUCO", "Detection mode", job.markers.detection_mode, defaults.markers.detection_mode, "One versioned detector mode is shared by every method in this queue."),
        ("accepted_ids", "QUEUE-WIDE ARUCO", "Accepted marker IDs", accepted, "all detected IDs", "Use all detections or a comma-separated ID list."),
        ("dictionary", "QUEUE-WIDE ARUCO", "ArUco dictionary", job.markers.dictionary, defaults.markers.dictionary, "Must match the printed marker family."),
        ("marker_length", "QUEUE-WIDE ARUCO", "Marker edge length [m]", job.markers.length_m, defaults.markers.length_m, "Positive physical size used for metric scale."),
        ("quality_reprojection", "OBSERVATION QUALITY BASELINE", "Global maximum PnP reprojection RMSE [px]", job.observation_quality.maximum_pnp_reprojection_error_px, 25.0, "PnP quality only; smaller rejects observations before any method runs."),
        ("quality_area", "OBSERVATION QUALITY BASELINE", "Global minimum marker area ratio", job.observation_quality.minimum_marker_area_ratio, 0.000008, "Marker pixels divided by image pixels; larger rejects small/distant detections independently of resolution."),
        ("quality_positive_depth", "OBSERVATION QUALITY BASELINE", "Global require positive marker depth", job.observation_quality.require_positive_depth, True, "Rejects PnP poses behind the camera; disable only for a documented diagnostic."),
        ("quality_distance", "OBSERVATION QUALITY BASELINE", "Global maximum marker distance [m]", job.observation_quality.maximum_marker_distance_m, "disabled", "Smaller retains only near PnP observations; disabled applies no distance cap."),
        ("quality_override_reprojection", "OBSERVATION QUALITY OVERRIDE", "Method maximum PnP reprojection RMSE [px]", effective_override_value("maximum_pnp_reprojection_error_px"), default_quality_value("maximum_pnp_reprojection_error_px"), "An unset method override internally uses the queue baseline; an explicit value affects only this method row."),
        ("quality_override_area", "OBSERVATION QUALITY OVERRIDE", "Method minimum marker area ratio", effective_override_value("minimum_marker_area_ratio"), default_quality_value("minimum_marker_area_ratio"), "An unset method override internally uses the queue baseline; larger values reject smaller marker detections."),
        ("quality_override_positive_depth", "OBSERVATION QUALITY OVERRIDE", "Method require positive marker depth", effective_override_value("require_positive_depth"), default_quality_value("require_positive_depth"), "An unset method override internally uses the queue baseline; explicit yes/no affects only this method."),
        ("quality_override_distance", "OBSERVATION QUALITY OVERRIDE", "Method maximum marker distance [m]", effective_override_value("maximum_marker_distance_m"), default_quality_value("maximum_marker_distance_m"), "An unset method override internally uses the queue baseline; disabled explicitly removes the distance cap."),
    ]
    if method_quality is None:
        rows = [
            row
            for row in rows
            if row[1] != "OBSERVATION QUALITY OVERRIDE"
        ]
    def guided_current(key: str, value: object) -> object:
        contextual = [
            selection_value(methods, key)
            for methods in job.context_methods.values()
        ]
        values = {
            json.dumps(item, sort_keys=True)
            for item in contextual
        }
        review_pending = (
            job.selection.mode == "review_once"
            or any(
                selection.mode == "review_once"
                for selection in job.context_selections.values()
            )
        )
        if review_pending:
            return "manual after preflight"
        if len(values) > 1:
            return "manual per experiment"
        if contextual:
            return contextual[0]
        return value

    if job.method_id == "ap01":
        value, base = job.methods.ap01, defaults.methods.ap01
        rows.extend([
            ("ap01_advanced_strategy", "METHOD-SPECIFIC SETTINGS", "AP01 strategy", value.advanced_strategy, base.advanced_strategy, "The baseline strategy uses Direct/Relay selection; the robustness strategy enables configurable caps and consensus gates."),
            ("ap01_direct_target", "METHOD-SPECIFIC SETTINGS", "Direct target camera", value.direct_target_camera, base.direct_target_camera, "One configurable camera is calibrated through the Direct path; other cameras use Relay support."),
            ("root_camera", "METHOD-SPECIFIC SETTINGS", "Root camera", guided_current("root_camera", value.root_camera), base.root_camera, "Coordinate origin; auto is resolved from filtered graph coverage."),
        ])
        if value.advanced_strategy == "wizard_robustness_v1":
            rows.extend([
                ("ap01_top_moving", "METHOD-SPECIFIC SETTINGS", "Relay observations per marker", value.top_moving_per_marker, base.top_moving_per_marker, "Quality-ranked moving observations kept per marker; null keeps all."),
                ("ap01_scale_top", "METHOD-SPECIFIC SETTINGS", "Scale observations per marker", value.scale_top_per_marker, base.scale_top_per_marker, "Quality-ranked observations kept before scale-pair construction; null keeps all."),
                ("ap01_direct_markers", "METHOD-SPECIFIC SETTINGS", "Direct minimum independent inlier markers", value.direct_quality_gate.minimum_independent_markers, base.direct_quality_gate.minimum_independent_markers, "Higher requires more independent marker evidence."),
                ("ap01_direct_inlier_ratio", "METHOD-SPECIFIC SETTINGS", "Direct minimum inlier ratio", value.direct_quality_gate.minimum_inlier_ratio, base.direct_quality_gate.minimum_inlier_ratio, "Fraction in [0,1]; higher rejects less-consistent direct candidate sets."),
                ("ap01_direct_translation", "METHOD-SPECIFIC SETTINGS", "Direct maximum translation dispersion [m]", value.direct_quality_gate.maximum_translation_dispersion_m, base.direct_quality_gate.maximum_translation_dispersion_m, "Lower requires tighter direct-pose consensus."),
                ("ap01_direct_rotation", "METHOD-SPECIFIC SETTINGS", "Direct maximum rotation dispersion [deg]", value.direct_quality_gate.maximum_rotation_dispersion_deg, base.direct_quality_gate.maximum_rotation_dispersion_deg, "Lower requires tighter direct orientation consensus."),
                ("ap01_relay_inlier_ratio", "METHOD-SPECIFIC SETTINGS", "Relay minimum inlier ratio", value.relay_quality_gate.minimum_inlier_ratio, base.relay_quality_gate.minimum_inlier_ratio, "Fraction in [0,1]; higher rejects less-consistent relay candidates."),
                ("ap01_relay_translation", "METHOD-SPECIFIC SETTINGS", "Relay maximum translation dispersion [m]", value.relay_quality_gate.maximum_translation_dispersion_m, base.relay_quality_gate.maximum_translation_dispersion_m, "Lower requires tighter Relay consensus."),
                ("ap01_relay_rotation", "METHOD-SPECIFIC SETTINGS", "Relay maximum rotation dispersion [deg]", value.relay_quality_gate.maximum_rotation_dispersion_deg, base.relay_quality_gate.maximum_rotation_dispersion_deg, "Lower requires tighter Relay orientation consensus."),
                ("ap01_consistency_translation", "METHOD-SPECIFIC SETTINGS", "Direct/Relay maximum translation disagreement [m]", value.direct_relay_consistency.maximum_translation_disagreement_m, base.direct_relay_consistency.maximum_translation_disagreement_m, "If both paths are stable, larger disagreement publishes Direct with a visible warning."),
                ("ap01_consistency_rotation", "METHOD-SPECIFIC SETTINGS", "Direct/Relay maximum rotation disagreement [deg]", value.direct_relay_consistency.maximum_rotation_disagreement_deg, base.direct_relay_consistency.maximum_rotation_disagreement_deg, "If both paths are stable, larger disagreement publishes Direct with a visible warning."),
            ])
    elif job.method_id == "ap02":
        value, base = job.methods.ap02, defaults.methods.ap02
        reference_current = (
            "marker 14"
            if value.reference_marker_selection_mode == "baseline"
            else "manual after preflight"
            if (
                value.reference_marker_selection_mode == "manual"
                and value.reference_marker_id == "auto"
            )
            else guided_current(
                "ap02_reference", value.reference_marker_id
            )
        )
        rows.extend([
            ("ap02_frame_strategy", "METHOD-SPECIFIC SETTINGS", "Frame-selection strategy", value.frame_selection_strategy, base.frame_selection_strategy, "Smart frame budgets are applied at the bundle-adjustment boundary; graph-preserving preselection is an advanced alternative."),
            ("ap02_initialization_strategy", "METHOD-SPECIFIC SETTINGS", "Initialization strategy", value.initialization_strategy, base.initialization_strategy, "Selects the deterministic reference-rooted graph tree policy."),
            ("ap02_edge_weight_strategy", "METHOD-SPECIFIC SETTINGS", "Graph edge weights", value.graph_edge_weight_strategy, base.graph_edge_weight_strategy, "Geometric observation quality is the baseline; a shared selection score is available for experiments."),
            ("ap02_reprojection_model", "METHOD-SPECIFIC SETTINGS", "Reprojection model", value.reprojection_model, base.reprojection_model, "Pinhole projection is the baseline; distortion-aware projection is an advanced alternative."),
            ("ap02_reference_mode", "METHOD-SPECIFIC SETTINGS", "Reference-marker selection mode", value.reference_marker_selection_mode, base.reference_marker_selection_mode, "Baseline uses canonical marker 14; auto uses the deterministic recommendation; manual pauses once after preflight; explicit is retained for compatible schema-v5 files."),
            ("ap02_reference_display", "METHOD-SPECIFIC SETTINGS", "Resolved/reference marker", reference_current, "auto", "Read-only preview. Auto and manual choices are resolved from the detected marker inventory during preflight."),
            ("ap02_reference_frames", "METHOD-SPECIFIC SETTINGS", "Reference-marker frame limit", value.reference_marker_maximum_frames, base.reference_marker_maximum_frames, "Quality-ranked reference-marker frames; null is unlimited."),
            ("ap02_top_marker", "METHOD-SPECIFIC SETTINGS", "Top frames per marker", value.top_per_marker, base.top_per_marker, "Quality-ranked frames retained for each marker; null keeps all."),
            ("ap02_top_pair", "METHOD-SPECIFIC SETTINGS", "Top frames per marker pair", value.top_per_marker_pair, base.top_per_marker_pair, "Prioritizes cross-marker bridge frames; null keeps all."),
            ("ap02_total_frames", "METHOD-SPECIFIC SETTINGS", "Maximum total moving frames", value.maximum_total_frames, base.maximum_total_frames, "Hard cap; preflight rejects values below the graph-preserving minimum."),
            ("max_nfev_static", "METHOD-SPECIFIC SETTINGS", "Static-only BA function evaluation limit", value.max_nfev_static, base.max_nfev_static, "Bundle-adjustment optimizer budget for the diagnostic static-only result."),
            ("max_nfev_moving", "METHOD-SPECIFIC SETTINGS", "Combined static + moving BA function evaluation limit", value.max_nfev_moving, base.max_nfev_moving, "Bundle-adjustment optimizer budget for the primary combined result."),
            ("ba_loss", "METHOD-SPECIFIC SETTINGS", "BA robust loss", value.ba_robust_loss, base.ba_robust_loss, "soft_l1 baseline; huber is piecewise robust; linear disables robustness."),
            ("ba_loss_scale", "METHOD-SPECIFIC SETTINGS", "BA robust loss scale [px]", value.ba_robust_loss_scale_px, base.ba_robust_loss_scale_px, "Smaller downweights residuals earlier; typical tests: 1, 3, 5 px."),
        ])
    elif job.method_id == "ap03":
        single, multi, scale = (
            job.methods.ap03.single,
            job.methods.ap03.multi,
            job.methods.ap03.scale,
        )
        base_single, base_multi, base_scale = (
            defaults.methods.ap03.single,
            defaults.methods.ap03.multi,
            defaults.methods.ap03.scale,
        )
        rows.extend([
            ("ap03_feature_limit_policy", "METHOD-SPECIFIC SETTINGS", "Feature-limit policy", job.methods.ap03.feature_limit_policy, defaults.methods.ap03.feature_limit_policy, "COLMAP defaults leave SIFT limits unset; explicit limits use the AP03 values shown below."),
            ("ap03_scale_input_policy", "METHOD-SPECIFIC SETTINGS", "Scale input policy", job.methods.ap03.scale_input_policy, defaults.methods.ap03.scale_input_policy, "Registered-image detection uses all registered images; filtered detection also requires accepted observations."),
            ("ap03_minimum_marker_area", "METHOD-SPECIFIC SETTINGS", "Scale marker minimum area [px^2]", job.methods.ap03.minimum_marker_area_px2, defaults.methods.ap03.minimum_marker_area_px2, "Reject scale detections below this image-area threshold."),
            ("single_marker", "METHOD-SPECIFIC SETTINGS", "Single diagnostic scale marker", guided_current("single_marker", single.scale_marker_id), base_single.scale_marker_id, "Diagnostic marker chosen after filtered observations."),
            ("multi_markers", "METHOD-SPECIFIC SETTINGS", "Multi primary marker set", guided_current("multi_markers", _ids_text(multi.marker_ids)), _ids_text(base_multi.marker_ids), "The baseline uses markers 0-14; auto uses every compatible filtered marker."),
            ("scale_reprojection", "METHOD-SPECIFIC SETTINGS", "Scale RANSAC threshold [px]", scale.reprojection_threshold_px, base_scale.reprojection_threshold_px, "Shared by Single and Multi; smaller requires tighter triangulation support."),
            ("scale_ransac", "METHOD-SPECIFIC SETTINGS", "Scale RANSAC iterations", scale.ransac_iterations, base_scale.ransac_iterations, "Shared by Single and Multi; higher explores more hypotheses but increases runtime."),
            ("scale_inliers", "METHOD-SPECIFIC SETTINGS", "Scale minimum inliers", scale.minimum_inliers, base_scale.minimum_inliers, "Shared by Single and Multi; higher requires more supporting views per marker corner."),
        ])
        if job.methods.ap03.scale_input_policy == "wizard_filtered_observations_v1":
            rows.append(("scale_max_observations", "METHOD-SPECIFIC SETTINGS", "Maximum observations per marker", scale.maximum_observations_per_marker, base_scale.maximum_observations_per_marker, "Quality-ranked cap before corner triangulation; null keeps all."))
    else:
        current_extension = job.methods.extensions.get(job.method_id, {})
        default_extension = defaults.methods.extensions.get(job.method_id, {})
        rows.append(
            (
                "extension",
                "METHOD-SPECIFIC SETTINGS",
                "Extension options (YAML)",
                yaml.safe_dump(
                    current_extension, default_flow_style=True
                ).strip(),
                yaml.safe_dump(
                    default_extension, default_flow_style=True
                ).strip(),
                "Validated against the registered method config model.",
            )
        )
    if job.method_id in {"ap01", "ap03"}:
        rows.extend([
            ("matcher", "COLMAP SETTINGS", "Matcher", job.colmap.matcher, "exhaustive", "Exhaustive compares all image pairs; sequential limits temporal pairs."),
            ("compute_mode", "COLMAP SETTINGS", "COLMAP compute mode", job.colmap.compute_mode, "cpu_baseline", "CPU baseline is reproducible; GPU is explicit; auto probes capability and records the resolved device."),
            ("mapper_matches", "COLMAP SETTINGS", "Mapper minimum matches", job.colmap.mapper_minimum_matches, 8, "Higher requires stronger image pairs; lower may add weak registrations."),
        ])
        if job.method_id == "ap01":
            rows.extend([
                ("maximum_image_size", "COLMAP SETTINGS", "Maximum feature image size", job.colmap.maximum_image_size, 1600, "Maximum AP01 moving-reconstruction image dimension used during SIFT extraction."),
                ("maximum_features", "COLMAP SETTINGS", "Maximum features per image", job.colmap.maximum_features, 4096, "Maximum AP01 SIFT features extracted per moving image."),
            ])
        if (
            job.method_id == "ap03"
            and job.methods.ap03.feature_limit_policy
            == "wizard_explicit_limits_v1"
        ):
            rows.extend([
                ("ap03_image_size", "COLMAP SETTINGS", "Maximum feature image size", job.colmap.ap03_maximum_image_size, 2400, "Maximum AP03 image dimension used during SIFT extraction."),
                ("ap03_features", "COLMAP SETTINGS", "Maximum features per image", job.colmap.ap03_maximum_features, 8192, "Maximum AP03 SIFT features extracted per image."),
            ])
        if job.colmap.matcher == "sequential":
            rows.append(("sequential_overlap", "COLMAP SETTINGS", "Sequential overlap", job.colmap.sequential_overlap, 20, "Number of temporal neighbors; higher widens matching."))
            if (
                job.method_id == "ap01"
                or job.methods.ap03.feature_limit_policy
                == "wizard_explicit_limits_v1"
            ):
                rows.append(("loop_detection", "COLMAP SETTINGS", "Sequential loop detection", job.colmap.loop_detection, True, "Adds non-local loop candidates for sequential matching."))
    return rows if groups is None else [row for row in rows if row[1] in groups]



__all__ = ["SettingRow", "build_setting_rows"]
