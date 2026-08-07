"""Schema checks for the parity/recommended/fast preset separation."""

from __future__ import annotations

from typing import Any


PRESET_NAMES = {
    "main_route2_parity_v1",
    "recommended_wizard_v1",
    "fast_50x50",
}


def validate_presets(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("preset schema_version must be 1")
    presets = payload.get("presets")
    if not isinstance(presets, dict) or set(presets) != PRESET_NAMES:
        raise ValueError("the three named preset contracts are required")
    parity = presets["main_route2_parity_v1"]
    required_locks = {
        "dataset_fingerprint",
        "root_camera",
        "ap02_reference_marker_id",
        "evaluation_anchor_marker_id",
        "observation_semantics",
        "ap01_aggregate_selection",
        "ap02_frame_selection",
        "ap02_graph_initialization",
        "ap02_static_max_nfev",
        "ap02_combined_max_nfev",
        "colmap_compute",
        "intrinsics_refinement",
        "implementation_versions",
    }
    missing = required_locks - set(parity.get("locks", {}))
    if missing:
        raise ValueError(f"parity preset is missing locks: {sorted(missing)}")
    if parity.get("parity") is not True or parity.get("locked") is not True:
        raise ValueError("main_route2_parity_v1 must be locked parity")
    if presets["fast_50x50"].get("parity") is not False:
        raise ValueError("fast_50x50 must remain explicitly non-parity")
    if presets["recommended_wizard_v1"].get("parity") is not False:
        raise ValueError("recommended_wizard_v1 is not a parity promise")

