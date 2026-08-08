from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_INSTALLED = False


def _authoritative_anchor(root: Path) -> int | None:
    from .result_output_policy import _authoritative_anchor as resolve

    return resolve(root)


def _marker_report(root: Path) -> Path | None:
    preferred = (
        root
        / "evaluations"
        / "method_anchors_reconciled"
        / "REAL_DATA_MARKER_CONSISTENCY.txt"
    )
    if preferred.is_file():
        return preferred
    candidates = sorted(
        (root / "evaluations").rglob("REAL_DATA_MARKER_CONSISTENCY.txt"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def install_final_reporting_frontdoor_policy() -> None:
    """Make the published real-data RESULTS front door self-contained and canonical.

    Native method RESULT.txt sections intentionally retain their own gauges
    (AP01 root camera, AP02 reference marker, AP03 COLMAP gauge).  The experiment
    front door additionally states the authoritative common export frame and
    embeds the marker-length/cross-camera reprojection evaluation exactly once.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from .evaluation import reporting

    original = reporting._real_results_text
    if getattr(original, "_rigcal_final_reporting_frontdoor", False):
        _INSTALLED = True
        return

    def real_results_text(
        experiment_root: Path,
        method_payloads: list[dict[str, Any]],
        dataset_root: Path | None = None,
    ):
        root = Path(experiment_root)
        text, payload = original(experiment_root, method_payloads, dataset_root)

        anchor = _authoritative_anchor(root)
        if anchor is not None:
            canonical = f"Common evaluation/export anchor: marker {anchor}"
            text = re.sub(
                r"^Common evaluation(?:/export)? anchor: marker .*?$",
                canonical,
                text,
                count=1,
                flags=re.MULTILINE,
            )
            payload.setdefault("evaluation_anchor", {})
            payload["evaluation_anchor"]["selected"] = anchor
            payload["evaluation_anchor"]["ground_truth_used"] = False

        report = _marker_report(root)
        title = "REAL-DATA MARKER LENGTH AND REPROJECTION RESULTS"
        if report is not None:
            payload["marker_consistency_path"] = str(report.relative_to(root))
            marker_text = report.read_text(encoding="utf-8").strip()
            if title not in text and marker_text:
                insertion = text.find("METHOD / VARIANT OVERVIEW")
                if insertion >= 0:
                    text = (
                        text[:insertion].rstrip()
                        + "\n\n"
                        + marker_text
                        + "\n\n"
                        + text[insertion:]
                    )
                else:
                    text = text.rstrip() + "\n\n" + marker_text + "\n"

        return text, payload

    real_results_text._rigcal_final_reporting_frontdoor = True  # type: ignore[attr-defined]
    reporting._real_results_text = real_results_text
    _INSTALLED = True
