#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path("results/bus_real_data/ablation/world/route")
FINAL99 = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")

VARIANTS = [
    ("route1", "Route 1"),
    ("route2", "Route 2"),
]

METHODS = ["AP01", "AP02", "AP03"]


def read_csv(path):
    if not path.is_file():
        return []

    with path.open(newline="", errors="replace") as file:
        return list(csv.DictReader(file))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, digits=2):
    value = number(value)
    return "-" if value is None else f"{value:.{digits}f}"


def mean_field(rows, field):
    values = [
        number(row.get(field))
        for row in rows
        if number(row.get(field)) is not None
    ]

    return mean(values) if values else None


def max_field(rows, field):
    values = [
        number(row.get(field))
        for row in rows
        if number(row.get(field)) is not None
    ]

    return max(values) if values else None


def diagnostic_flag(rows):
    """
    Diagnostic only. This does not replace the official coverage/execution status.

    A >90 degree pair or >100 cm pair is treated as an obvious catastrophic
    geometric outlier for the simulation benchmark.
    """
    max_t = max_field(rows, "translation_error_cm")
    max_r = max_field(rows, "rotation_error_deg")
    mean_t = mean_field(rows, "translation_error_cm")
    mean_r = mean_field(rows, "rotation_error_deg")

    if (
        (max_t is not None and max_t >= 100.0)
        or (max_r is not None and max_r >= 90.0)
    ):
        return "CATASTROPHIC_OUTLIER"

    if (
        (mean_t is not None and mean_t >= 20.0)
        or (mean_r is not None and mean_r >= 5.0)
    ):
        return "HIGH_ERROR"

    return "NOMINAL"


def read_status(path):
    result = {}

    if not path.is_file():
        return result

    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()

    return result


summary_lines = [
    "ROUTE ABLATION — AP01 / AP02 / AP03",
    "=" * 112,
    "",
    "Execution/coverage status and geometric accuracy are reported separately.",
    "OK_FULL means complete output coverage; it does not by itself guarantee an accurate solution.",
    "",
    (
        f"{'Route':9s}"
        f"{'Frames':>9s}"
        f"{'Method':>10s}"
        f"{'Coverage':>18s}"
        f"{'Diagnostic':>25s}"
        f"{'Mean t':>13s}"
        f"{'Median/Max t':>17s}"
        f"{'Mean r':>13s}"
        f"{'Max r':>13s}"
    ),
    "-" * 112,
]

primary_lines = [
    "ROUTE ABLATION — DETAILED CAMERA-TO-CAMERA RESULTS",
    "=" * 112,
    "",
    "Primary metric: direct static camera-to-camera extrinsics against simulation ground truth.",
    "",
]

secondary_lines = [
    "ROUTE ABLATION — DETAILED REF14/WORLD MAP-TO-GT RESULTS",
    "=" * 100,
    "",
    "Secondary metric: static camera map after SE(3) alignment against simulation ground truth.",
    "",
]

decision_data = {}

for variant, label in VARIANTS:
    final = ROOT / variant / "FINAL_RESULTS"

    if not final.is_dir():
        raise RuntimeError(f"Missing FINAL_RESULTS directory: {final}")

    metadata_path = ROOT / variant / "VARIANT_METADATA.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.is_file()
        else {}
    )

    frames = metadata.get("num_route_frames", "-")
    status = read_status(final / "RUN_STATUS.txt")

    primary_summary = read_csv(
        final / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    )
    primary_detail = read_csv(
        final / "BASELINE_FINAL_PAIRWISE_DETAIL.csv"
    )
    secondary_summary = read_csv(
        final / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv"
    )
    secondary_detail = read_csv(
        final / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv"
    )

    primary_lines += [
        "",
        "#" * 112,
        f"{label} — {variant} — {frames} moving frames",
        "#" * 112,
    ]

    secondary_lines += [
        "",
        "#" * 100,
        f"{label} — {variant} — {frames} moving frames",
        "#" * 100,
    ]

    decision_data[variant] = {}

    for method in METHODS:
        p_summary = next(
            (
                row
                for row in primary_summary
                if row.get("method") == method
            ),
            {},
        )

        s_summary = next(
            (
                row
                for row in secondary_summary
                if row.get("method") == method
            ),
            {},
        )

        p_rows = [
            row
            for row in primary_detail
            if row.get("method") == method
        ]

        s_rows = [
            row
            for row in secondary_detail
            if row.get("method") == method
        ]

        quality = diagnostic_flag(p_rows)

        mean_t = mean_field(p_rows, "translation_error_cm")
        max_t = max_field(p_rows, "translation_error_cm")
        mean_r = mean_field(p_rows, "rotation_error_deg")
        max_r = max_field(p_rows, "rotation_error_deg")

        coverage = p_summary.get("status", "-")

        summary_lines.append(
            f"{label:9s}"
            f"{str(frames):>9s}"
            f"{method:>10s}"
            f"{coverage:>18s}"
            f"{quality:>25s}"
            f"{fmt(mean_t) + ' cm':>13s}"
            f"{fmt(max_t) + ' cm':>17s}"
            f"{fmt(mean_r) + ' deg':>13s}"
            f"{fmt(max_r) + ' deg':>13s}"
        )

        decision_data[variant][method] = {
            "coverage": coverage,
            "diagnostic": quality,
            "primary_mean_translation_cm": mean_t,
            "primary_mean_rotation_deg": mean_r,
            "primary_max_translation_cm": max_t,
            "primary_max_rotation_deg": max_r,
            "secondary_mean_translation_cm": mean_field(
                s_rows,
                "translation_error_cm",
            ),
            "secondary_mean_rotation_deg": mean_field(
                s_rows,
                "rotation_error_deg",
            ),
        }

        primary_lines += [
            "",
            method,
            "-" * 112,
            (
                f"{'Pair':13s}"
                f"{'Status':13s}"
                f"{'t error [cm]':>15s}"
                f"{'r error [deg]':>16s}"
                f"{'baseline [cm]':>16s}"
                f"{'direction [deg]':>18s}"
                f"{'GT base [m]':>14s}"
                f"{'Est base [m]':>14s}"
            ),
            "-" * 112,
        ]

        for row in p_rows:
            primary_lines.append(
                f"{row.get('pair', '-'):13s}"
                f"{row.get('status', '-'):13s}"
                f"{fmt(row.get('translation_error_cm')):>15s}"
                f"{fmt(row.get('rotation_error_deg')):>16s}"
                f"{fmt(row.get('baseline_error_cm')):>16s}"
                f"{fmt(row.get('direction_error_deg')):>18s}"
                f"{fmt(row.get('gt_baseline_m'), 3):>14s}"
                f"{fmt(row.get('est_baseline_m'), 3):>14s}"
            )

        secondary_lines += [
            "",
            method,
            "-" * 100,
            (
                f"{'Camera':18s}"
                f"{'Status':13s}"
                f"{'t error [cm]':>17s}"
                f"{'r error [deg]':>18s}"
                f"{'Aligned X':>13s}"
                f"{'Aligned Y':>13s}"
                f"{'Aligned Z':>13s}"
            ),
            "-" * 100,
        ]

        for row in s_rows:
            secondary_lines.append(
                f"{row.get('camera', '-'):18s}"
                f"{row.get('status', '-'):13s}"
                f"{fmt(row.get('translation_error_cm')):>17s}"
                f"{fmt(row.get('rotation_error_deg')):>18s}"
                f"{fmt(row.get('aligned_est_x_m'), 3):>13s}"
                f"{fmt(row.get('aligned_est_y_m'), 3):>13s}"
                f"{fmt(row.get('aligned_est_z_m'), 3):>13s}"
            )

summary_lines += [
    "",
    "INTERPRETATION",
    "-" * 112,
    "",
    "1. Route 2 is selected as the canonical baseline route.",
    "2. Route 2 provides complete, non-catastrophic results for AP01, AP02 and AP03.",
    "3. Route 1 remains a documented acquisition-robustness failure case.",
    "4. Route 1 AP03 contains a near-180-degree static-camera orientation failure.",
    "5. AP02 is comparatively stable because explicit marker identities constrain the graph.",
    "6. More frames do not guarantee a better result when the route introduces ambiguous revisits.",
    "",
]

FINAL99.mkdir(parents=True, exist_ok=True)
(FINAL99 / "ablations").mkdir(parents=True, exist_ok=True)
(FINAL99 / "details/primary").mkdir(parents=True, exist_ok=True)
(FINAL99 / "details/secondary").mkdir(parents=True, exist_ok=True)

(FINAL99 / "ablations/05_ROUTE_PATH_ALL_METHODS.txt").write_text(
    "\n".join(summary_lines) + "\n"
)

(
    FINAL99
    / "details/primary/05_ROUTE_PATH_CAM_TO_CAM.txt"
).write_text(
    "\n".join(primary_lines) + "\n"
)

(
    FINAL99
    / "details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt"
).write_text(
    "\n".join(secondary_lines) + "\n"
)

decision_lines = [
    "CANONICAL BASELINE ROUTE SELECTION",
    "=" * 88,
    "",
    "Selected route: route2",
    "",
    "Reason:",
    "- complete coverage for AP01, AP02 and AP03",
    "- no catastrophic static-camera orientation failure",
    "- shorter and computationally cheaper than route1",
    "- controlled overlap with fewer ambiguous revisits",
    "- suitable as a common baseline without optimizing for one individual method",
    "",
    "Route 1 is retained as a robustness/failure-case ablation.",
    "",
]

(FINAL99 / "BASELINE_ROUTE_SELECTION.txt").write_text(
    "\n".join(decision_lines) + "\n"
)

print("[OK] readable route reports written")
print(FINAL99 / "ablations/05_ROUTE_PATH_ALL_METHODS.txt")
print(FINAL99 / "details/primary/05_ROUTE_PATH_CAM_TO_CAM.txt")
print(FINAL99 / "details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt")
print(FINAL99 / "BASELINE_ROUTE_SELECTION.txt")
