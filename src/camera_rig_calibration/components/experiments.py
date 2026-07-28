"""Built-in experiment variant providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config.models import RigConfig


@dataclass(frozen=True)
class ColmapMatcherExperiments:
    """Generate controlled exhaustive/sequential matcher variants."""

    id: str = "colmap_matcher"
    display_name: str = "COLMAP matcher comparison"
    description: str = "Create separate exhaustive and sequential matcher runs."

    def variants(self, config: RigConfig) -> Sequence[tuple[str, RigConfig]]:
        variants = []
        for matcher in ("exhaustive", "sequential"):
            variants.append(
                (
                    f"colmap_{matcher}",
                    config.model_copy(
                        update={
                            "project": config.project.model_copy(
                                update={"run_label": f"colmap_{matcher}"}
                            ),
                            "colmap": config.colmap.model_copy(
                                update={"matcher": matcher}
                            ),
                        },
                        deep=True,
                    ),
                )
            )
        return variants
