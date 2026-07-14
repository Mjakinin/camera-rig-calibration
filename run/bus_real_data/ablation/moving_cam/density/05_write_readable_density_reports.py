#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path("results/bus_real_data/ablation/moving_cam/density")
FINAL99 = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")
METHODS = ("AP01", "AP02", "AP03")
VARIANTS = (
    ("density_route2_125pct_recaptured", "125% recaptured"),
    ("density_stride_1_100pct", "100%"),
    ("density_stride_2_50pct", "50%"),
    ("density_stride_4_25pct", "25%"),
    ("density_stride_8_12p5pct", "12.5%"),
    ("density_stride_8_offset4", "12.5% offset 4"),
    ("density_stride_16_6p25pct", "6.25%"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    parsed = num(value)
    return "-" if parsed is None else f"{parsed:.{digits}f}{suffix}"


def safe(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text if text else "-"


primary = [
    "MOVING-CAMERA FRAME DENSITY — DETAILED CAMERA-TO-CAMERA RESULTS",
    "=" * 112,
    "",
    "Primary metric: static camera-to-camera extrinsics against simulation GT.",
    "Missing-camera rows remain visible. Partial means are not compared with six-pair means.",
    "",
]
secondary = [
    "MOVING-CAMERA FRAME DENSITY — DETAILED OPTIONAL MAP-TO-GT RESULTS",
    "=" * 124,
    "",
    "Method-independent section: best-fit SE(3)-aligned static-camera map against GT.",
    "AP02 REF14-anchored available marker maps are appended by 33_write_ref14_available_maps.py.",
    "",
]

for variant, parameter in VARIANTS:
    final = ROOT / variant / "FINAL_RESULTS"
    if not final.is_dir():
        raise RuntimeError(f"Missing FINAL_RESULTS: {final}")

    p_summary = read_csv(final / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv")
    p_detail = read_csv(final / "BASELINE_FINAL_PAIRWISE_DETAIL.csv")
    s_summary = read_csv(final / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv")
    s_detail = read_csv(final / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv")

    primary += [
        "#" * 112,
        f"PARAMETER: {parameter}",
        f"VARIANT:   {variant}",
        "#" * 112,
        "",
    ]
    secondary += [
        "#" * 124,
        f"PARAMETER: {parameter}",
        f"VARIANT:   {variant}",
        "#" * 124,
        "",
    ]

    for method in METHODS:
        ps = next((row for row in p_summary if row.get("method") == method), {})
        pd = [row for row in p_detail if row.get("method") == method]
        ss = next((row for row in s_summary if row.get("method") == method), {})
        sd = [row for row in s_detail if row.get("method") == method]

        primary += [
            f"{method} — {safe(ps.get('status'))} — "
            f"{safe(ps.get('camera_count'))}/4 cameras — "
            f"{safe(ps.get('pair_count_ok'))}/6 pairs",
            "-" * 112,
            (
                f"{'Pair':13s}{'Status':18s}{'t error':>12s}{'r error':>12s}"
                f"{'GT base':>12s}{'Est base':>12s}{'Base err':>12s}{'Dir err':>12s}"
            ),
            "-" * 112,
        ]
        if not pd:
            primary.append("- no pair rows")
        for row in pd:
            primary.append(
                f"{safe(row.get('pair'))[:13]:13s}"
                f"{safe(row.get('status'))[:18]:18s}"
                f"{fmt(row.get('translation_error_cm'), 2, ' cm'):>12s}"
                f"{fmt(row.get('rotation_error_deg'), 2, ' deg'):>12s}"
                f"{fmt(row.get('gt_baseline_m'), 3, ' m'):>12s}"
                f"{fmt(row.get('est_baseline_m'), 3, ' m'):>12s}"
                f"{fmt(row.get('baseline_error_cm'), 2, ' cm'):>12s}"
                f"{fmt(row.get('direction_error_deg'), 2, ' deg'):>12s}"
            )
        primary.append("")

        secondary += [
            f"{method} — {safe(ss.get('status'))} — {safe(ss.get('camera_count'))}/4 cameras",
            "-" * 124,
            (
                f"{'Camera':14s}{'Status':24s}{'t error':>13s}{'r error':>13s}"
                f"{'Aligned XYZ [m]':>30s}{'GT XYZ [m]':>30s}"
            ),
            "-" * 124,
        ]
        if not sd:
            secondary.append("- no camera-map rows")
        for row in sd:
            aligned = (
                f"({fmt(row.get('aligned_est_x_m'), 3)}, "
                f"{fmt(row.get('aligned_est_y_m'), 3)}, "
                f"{fmt(row.get('aligned_est_z_m'), 3)})"
            )
            gt = (
                f"({fmt(row.get('gt_x_m'), 3)}, "
                f"{fmt(row.get('gt_y_m'), 3)}, "
                f"{fmt(row.get('gt_z_m'), 3)})"
            )
            secondary.append(
                f"{safe(row.get('camera'))[:14]:14s}"
                f"{safe(row.get('status'))[:24]:24s}"
                f"{fmt(row.get('translation_error_cm'), 2, ' cm'):>13s}"
                f"{fmt(row.get('rotation_error_deg'), 2, ' deg'):>13s}"
                f"{aligned:>30s}{gt:>30s}"
            )
        secondary.append("")

(FINAL99 / "details/primary").mkdir(parents=True, exist_ok=True)
(FINAL99 / "details/secondary").mkdir(parents=True, exist_ok=True)
(FINAL99 / "details/primary/06_FRAME_DENSITY_CAM_TO_CAM.txt").write_text(
    "\n".join(primary) + "\n",
    encoding="utf-8",
)
(FINAL99 / "details/secondary/06_FRAME_DENSITY_MAP_TO_GT.txt").write_text(
    "\n".join(secondary) + "\n",
    encoding="utf-8",
)

print("[OK] readable density reports written")
