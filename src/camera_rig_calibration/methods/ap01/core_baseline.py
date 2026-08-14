"""Baseline AP01 candidate aggregation."""
from __future__ import annotations

import numpy as np

from .contracts import AP01MethodContract, resolve_ap01_method_contract


CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]
ROOT_CAMERA = "cam_edge_3"



from .core_candidates import rotation_difference_deg, weighted_rotation_mean
from .core_geometry import make_T
def weighted_transform_mean(
    candidates: list[dict], indices: list[int]
) -> np.ndarray:
    if not indices:
        raise RuntimeError("No AP01 transforms to average")
    weights = np.asarray(
        [max(1e-12, float(candidates[index]["quality"])) for index in indices],
        dtype=np.float64,
    )
    weights /= weights.sum()
    translation = np.sum(
        np.asarray(
            [candidates[index]["T"][:3, 3] for index in indices],
            dtype=np.float64,
        )
        * weights[:, None],
        axis=0,
    )
    rotation = weighted_rotation_mean(
        [candidates[index]["T"][:3, :3] for index in indices], weights
    )
    return make_T(rotation, translation)


def se3_medoid(candidates: list[dict]) -> tuple[int, float]:
    """Return the first-on-equal SE(3) medoid using t + 0.02*r distance."""

    if not candidates:
        raise RuntimeError("No AP01 direct candidates")
    best_index = 0
    best_score: float | None = None
    for index, candidate in enumerate(candidates):
        distances = []
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            translation = float(
                np.linalg.norm(candidate["T"][:3, 3] - other["T"][:3, 3])
            )
            rotation = rotation_difference_deg(
                candidate["T"][:3, :3], other["T"][:3, :3]
            )
            distances.append(translation + 0.02 * rotation)
        score = float(np.median(distances)) if distances else 0.0
        if best_score is None or score < best_score:
            best_index = index
            best_score = score
    return best_index, float(best_score or 0.0)


def medoid_inliers(
    candidates: list[dict],
    medoid_index: int,
    *,
    translation_floor: float,
    rotation_floor: float,
) -> tuple[list[int], dict]:
    center = candidates[medoid_index]["T"]
    translation = np.asarray(
        [
            np.linalg.norm(candidate["T"][:3, 3] - center[:3, 3])
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [
            rotation_difference_deg(
                candidate["T"][:3, :3], center[:3, :3]
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    t_median = float(np.median(translation))
    r_median = float(np.median(rotation))
    t_mad = 1.4826 * float(np.median(np.abs(translation - t_median)))
    r_mad = 1.4826 * float(np.median(np.abs(rotation - r_median)))
    t_threshold = max(translation_floor, t_median + 3.0 * t_mad)
    r_threshold = max(rotation_floor, r_median + 3.0 * r_mad)
    indices = [
        index
        for index, (t_value, r_value) in enumerate(zip(translation, rotation))
        if t_value <= t_threshold and r_value <= r_threshold
    ]
    return indices, {
        "translation_deviation_median_m": t_median,
        "translation_deviation_mad_scaled_m": t_mad,
        "translation_inlier_threshold_m": t_threshold,
        "rotation_deviation_median_deg": r_median,
        "rotation_deviation_mad_scaled_deg": r_mad,
        "rotation_inlier_threshold_deg": r_threshold,
    }


def aggregate_baseline_direct_candidates(
    candidates: list[dict], contract: AP01MethodContract
) -> tuple[np.ndarray, dict]:
    """Build the baseline quality-filtered Direct aggregate."""

    if not candidates:
        raise RuntimeError("No AP01 direct candidates")
    quality_indices = [
        index
        for index, candidate in enumerate(candidates)
        if float(candidate.get("root_area_px2", float("nan")))
        >= float(contract.direct_minimum_area_px2 or 0.0)
        and float(candidate.get("target_area_px2", float("nan")))
        >= float(contract.direct_minimum_area_px2 or 0.0)
        and float(candidate.get("root_distance_m", float("inf")))
        <= float(contract.direct_maximum_distance_m or float("inf"))
        and float(candidate.get("target_distance_m", float("inf")))
        <= float(contract.direct_maximum_distance_m or float("inf"))
        and float(candidate["quality"])
        >= float(contract.direct_minimum_combined_quality or 0.0)
    ]
    fallback = False
    if not quality_indices:
        fallback = True
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: float(candidates[index]["quality"]),
            reverse=True,
        )
        fallback_count = int(contract.direct_quality_fallback_count or 1)
        quality_indices = ranked[: max(1, min(fallback_count, len(ranked)))]
    quality_candidates = [candidates[index] for index in quality_indices]
    medoid_index, medoid_score = se3_medoid(quality_candidates)
    inlier_local, inlier_stats = medoid_inliers(
        quality_candidates,
        medoid_index,
        translation_floor=contract.direct_translation_mad_floor_m,
        rotation_floor=contract.direct_rotation_mad_floor_deg,
    )
    if not inlier_local:
        inlier_local = list(range(len(quality_candidates)))
    inlier_indices = {quality_indices[index] for index in inlier_local}
    quality_set = set(quality_indices)
    preferred_index = next(
        (
            index
            for index in quality_indices
            if int(candidates[index]["root_marker"])
            == contract.preferred_direct_marker_id
        ),
        None,
    )
    if preferred_index is None:
        selected_index = quality_indices[medoid_index]
        selection_note = "quality_filtered_se3_medoid_fallback"
    else:
        selected_index = preferred_index
        selection_note = "marker14_visible_and_passed_no_gt_quality_filter"
    weighted_diagnostic = weighted_transform_mean(
        candidates, sorted(inlier_indices)
    )
    for index, candidate in enumerate(candidates):
        candidate["quality_filter_eligible"] = index in quality_set
        candidate["quality_filter_fallback_used"] = fallback
        candidate["inlier"] = index in inlier_indices
        candidate["pose_support"] = index in inlier_indices
        candidate["preferred_marker_selected"] = index == selected_index
    return candidates[selected_index]["T"], {
        "selected_aggregate_type": (
            "quality_filtered_preferred_marker_no_gt_selection"
        ),
        "aggregate_priority": [
            "quality_filtered_preferred_marker_no_gt_selection",
            "quality_filtered_weighted_mean_no_gt_selection",
            "weighted_mean_of_mad_inliers_no_gt_selection",
            "se3_medoid_no_gt_selection",
        ],
        "selected_marker_id": candidates[selected_index]["root_marker"],
        "selected_candidate_index": selected_index,
        "selection_note": selection_note,
        "quality_filter_fallback_used": fallback,
        "num_candidates": len(candidates),
        "num_quality_candidates": len(quality_indices),
        "num_quality_mad_inliers": len(inlier_indices),
        "quality_subset_medoid_score": medoid_score,
        "quality_subset_mad": inlier_stats,
        "quality_filtered_weighted_mean_diagnostic": (
            weighted_diagnostic.tolist()
        ),
        "ground_truth_used": False,
    }


def aggregate_baseline_relay_candidates(
    candidates: list[dict], contract: AP01MethodContract
) -> tuple[np.ndarray, dict]:
    """Build the baseline one-level flat MAD Relay aggregate."""

    if not candidates:
        raise RuntimeError("No AP01 relay candidates")
    translations = np.asarray(
        [candidate["T"][:3, 3] for candidate in candidates], dtype=np.float64
    )
    all_indices = list(range(len(candidates)))
    weighted = weighted_transform_mean(candidates, all_indices)
    initial = make_T(weighted[:3, :3], np.median(translations, axis=0))
    translation_deviation = np.asarray(
        [
            np.linalg.norm(candidate["T"][:3, 3] - initial[:3, 3])
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    rotation_deviation = np.asarray(
        [
            rotation_difference_deg(
                candidate["T"][:3, :3], initial[:3, :3]
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    t_median = float(np.median(translation_deviation))
    r_median = float(np.median(rotation_deviation))
    t_mad = 1.4826 * float(
        np.median(np.abs(translation_deviation - t_median))
    )
    r_mad = 1.4826 * float(
        np.median(np.abs(rotation_deviation - r_median))
    )
    t_threshold = max(
        contract.relay_translation_mad_floor_m, t_median + 3.0 * t_mad
    )
    r_threshold = max(
        contract.relay_rotation_mad_floor_deg, r_median + 3.0 * r_mad
    )
    inlier_indices = [
        index
        for index, (t_value, r_value) in enumerate(
            zip(translation_deviation, rotation_deviation)
        )
        if t_value <= t_threshold and r_value <= r_threshold
    ]
    fallback = False
    minimum = int(contract.relay_fallback_minimum_count or 0)
    if len(inlier_indices) < minimum:
        fallback = True
        ranked = sorted(
            all_indices,
            key=lambda index: float(candidates[index]["quality"]),
            reverse=True,
        )
        fraction = float(contract.relay_fallback_fraction or 0.5)
        keep = max(minimum, int(len(ranked) * fraction))
        inlier_indices = ranked[:keep]
    inlier_set = set(inlier_indices)
    for index, candidate in enumerate(candidates):
        candidate["inlier"] = index in inlier_set
        candidate["pose_support"] = index in inlier_set
        candidate["translation_deviation_m"] = float(
            translation_deviation[index]
        )
        candidate["rotation_deviation_deg"] = float(rotation_deviation[index])
    transform = weighted_transform_mean(candidates, inlier_indices)
    return transform, {
        "aggregate_type": "weighted_mean_of_mad_inliers_no_gt_selection",
        "num_candidates": len(candidates),
        "num_inliers": len(inlier_indices),
        "num_outliers": len(candidates) - len(inlier_indices),
        "translation_deviation_median_m": t_median,
        "translation_deviation_mad_scaled_m": t_mad,
        "translation_inlier_threshold_m": t_threshold,
        "rotation_deviation_median_deg": r_median,
        "rotation_deviation_mad_scaled_deg": r_mad,
        "rotation_inlier_threshold_deg": r_threshold,
        "fallback_top_half_by_quality": fallback,
        "ground_truth_used": False,
    }
