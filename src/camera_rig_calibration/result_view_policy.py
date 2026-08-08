from __future__ import annotations

from pathlib import Path
from typing import Any


_INSTALLED = False


def install_result_view_policy() -> None:
    """Avoid expensive scientific re-finalization before showing the result menu.

    Completed layout-v2 runs already materialize RESULTS/COMPARISON during
    publication. Browsing those immutable artifacts must be immediate. Older or
    incomplete experiments still fall back to the original reconciliation path.
    RViz itself keeps its own lazy ensure_visualization_artifacts() call when the
    user explicitly chooses the visualization option.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from . import wizard

    original = wizard.reconcile_existing_experiment
    if getattr(original, "_rigcal_fast_result_view", False):
        _INSTALLED = True
        return

    def reconcile_for_view(
        root: Path,
        *,
        dataset_root: Path,
        category: str,
    ) -> dict[str, Any]:
        root = Path(root)
        if (root / "RESULTS.txt").is_file() and (root / "COMPARISON.json").is_file():
            return {
                "status": "already_materialized",
                "reason": "completed published result opened read-only",
                "experiment_root": str(root.resolve()),
                "category": category,
            }
        return original(
            root,
            dataset_root=dataset_root,
            category=category,
        )

    reconcile_for_view._rigcal_fast_result_view = True  # type: ignore[attr-defined]
    wizard.reconcile_existing_experiment = reconcile_for_view
    _INSTALLED = True
