"""Human-readable reports for real-data marker-consistency evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable


def format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def text_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    rendered = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    heading = " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rendered
    ]
    return "\n".join([heading, separator, *body])


def report(
    path: Path,
    dataset: Path,
    anchor: int,
    marker_length_m: float,
    summaries: list[dict[str, Any]],
    marker_rows: list[dict[str, Any]],
) -> None:
    """Write the stable text front door for all evaluated method variants."""
    del dataset, anchor
    width = 138
    lines = [
        "REAL-DATA MARKER LENGTH AND REPROJECTION RESULTS",
        "=" * width,
        "",
        f"Expected marker edge length: {100 * marker_length_m:.2f} cm",
        "",
        "METHOD / VARIANT RESULTS",
        "-" * width,
        text_table(
            [
                "Method / variant",
                "Status",
                "Anchor",
                "Cams",
                "Moving",
                "Markers",
                "Marker RMSE [cm]",
                "Marker RMSE [%]",
                "Moving RMSE [px]",
                "Cross-camera RMSE [px]",
                "Cross corners",
            ],
            [
                [
                    summary.get("method", "-"),
                    summary.get("status", "-"),
                    summary.get("anchor_marker_id", "-"),
                    summary.get("available_static_camera_count", 0),
                    summary.get("registered_moving_frames", 0),
                    summary.get("evaluated_non_anchor_markers", 0),
                    format_value(summary.get("marker_length_rmse_cm")),
                    format_value(summary.get("marker_length_rmse_percent")),
                    format_value(summary.get("moving_fit_reprojection_rmse_px")),
                    format_value(summary.get("moving_to_static_reprojection_rmse_px")),
                    summary.get("moving_to_static_reprojection_observations", 0),
                ]
                for summary in summaries
            ],
        ),
    ]
    for summary in summaries:
        method = summary.get("method", "-")
        rows = sorted(
            (row for row in marker_rows if row.get("method") == method),
            key=lambda row: int(row.get("marker_id", 0)),
        )
        lines.extend(["", f"{method}: MARKER RESULTS", "-" * width])
        if not rows:
            error = summary.get("error")
            lines.append(
                f"Unavailable: {error}"
                if error
                else "Unavailable: no marker could be evaluated."
            )
            continue
        lines.append(
            text_table(
                [
                    "Marker",
                    "Role",
                    "Estimated length [cm]",
                    "Expected length [cm]",
                    "Abs error [cm]",
                    "Rel error [%]",
                    "Moving RMSE [px]",
                    "Cross-camera RMSE [px]",
                    "Static cams",
                    "Cross corners",
                ],
                [
                    [
                        row.get("marker_id", "-"),
                        (
                            "scale anchor"
                            if row.get("is_scale_anchor")
                            else "validation"
                        ),
                        format_value(row.get("estimated_marker_size_cm")),
                        format_value(row.get("expected_marker_size_cm")),
                        format_value(row.get("absolute_size_error_cm")),
                        format_value(row.get("relative_size_error_percent")),
                        format_value(row.get("moving_fit_reprojection_rmse_px")),
                        format_value(row.get("moving_to_static_reprojection_rmse_px")),
                        row.get("static_validation_camera_count", 0),
                        row.get("moving_to_static_reprojection_observations", 0),
                    ]
                    for row in rows
                ],
            )
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


__all__ = ["format_value", "report", "text_table"]
