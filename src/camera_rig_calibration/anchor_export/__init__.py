"""Method-independent camera-pose export in the common marker frame."""

from pathlib import Path
from typing import Any

from .compact import write_compact_method_anchor_yaml
from .exporter import (
    ensure_experiment_anchor_exports as _ensure_experiment_anchor_exports,
    export_method_anchor_poses as _export_method_anchor_poses,
)


def export_method_anchor_poses(method_root: Path) -> dict[str, Any]:
    """Publish the canonical export and its compact deployment view."""
    outcome = _export_method_anchor_poses(method_root)
    write_compact_method_anchor_yaml(method_root)
    return outcome


def ensure_experiment_anchor_exports(
    experiment_root: Path,
) -> dict[str, dict[str, Any]]:
    """Ensure canonical exports, then mirror each one to compact YAML."""
    experiment_root = Path(experiment_root)
    outcomes = _ensure_experiment_anchor_exports(experiment_root)
    methods_root = experiment_root / "methods"
    if methods_root.is_dir():
        for source in sorted(
            methods_root.glob("*/*/camera_extrinsics_anchor.json")
        ):
            write_compact_method_anchor_yaml(source.parent)
    return outcomes


__all__ = [
    "ensure_experiment_anchor_exports",
    "export_method_anchor_poses",
    "write_compact_method_anchor_yaml",
]
