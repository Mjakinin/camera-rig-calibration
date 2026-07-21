#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BUS = ROOT / "results/bus_real_data"

METHODS = ("AP01", "AP02", "AP03")

GROUPS = [
    {
        "key": "moving_cam/fov",
        "title": "MOVING-CAMERA FIELD OF VIEW",
        "slug": "01_FOV",
        "variants": [
            ("fov_40deg", "40 deg"),
            ("fov_69deg_baseline", "69 deg baseline"),
            ("fov_100deg", "100 deg"),
            ("fov_140deg_extreme", "140 deg"),
        ],
    },
    {
        "key": "moving_cam/motion_blur",
        "title": "MOVING-CAMERA MOTION BLUR",
        "slug": "02_MOTION_BLUR",
        "variants": [
            ("moving_blur_k00_baseline", "kernel 0 baseline"),
            ("moving_blur_k09_mild", "kernel 9"),
            ("moving_blur_k21_strong", "kernel 21"),
            ("moving_blur_k41_extreme", "kernel 41"),
        ],
    },
    {
        "key": "moving_cam/res",
        "title": "MOVING-CAMERA RESOLUTION",
        "slug": "03_RESOLUTION",
        "variants": [
            ("moving_res_160x90_extreme_pixel", "160x90"),
            ("moving_res_320x180_low", "320x180"),
            ("moving_res_1280x720_baseline", "1280x720 baseline"),
            ("moving_res_2560x1440_upscaled", "2560x1440"),
        ],
    },
    {
        "key": "world/lighting",
        "title": "WORLD LIGHTING",
        "slug": "04_LIGHTING",
        "variants": [
            ("ceiling_dark_extreme", "dark extreme"),
            ("ceiling_low", "low"),
            ("ceiling_normal", "normal baseline"),
            ("ceiling_bright", "bright"),
        ],
    },
]



PRESERVE_COMPATIBILITY_FILES = (
    "MANIFEST.txt",

    "data/ALL_ABLATIONS_PRIMARY.csv",
    "data/ALL_ABLATIONS_PRIMARY_DETAIL.csv",
    "data/ALL_ABLATIONS_SECONDARY.csv",
    "data/ALL_ABLATIONS_SECONDARY_DETAIL.csv",

    "data/AP02_MARKER_MAP_PARAMETER_STABILITY.csv",
    "data/AP02_V1_V2_COMPARISON.csv",
    "data/AP02_VALIDITY_AUDIT.csv",

    "data/BASELINE_PRIMARY.csv",
    "data/BASELINE_PRIMARY_DETAIL.csv",
    "data/BASELINE_SECONDARY.csv",
    "data/BASELINE_SECONDARY_DETAIL.csv",

    "data/primary/ALL_METHODS_CAM_TO_CAM_DETAIL.csv",
    "data/primary/BASELINE_ALL_METHODS_CAM_TO_CAM_DETAIL.csv",

    "data/secondary/ALL_METHODS_ALIGNED_CAMERA_MAP_VS_GT_DETAIL.csv",
    "data/secondary/BASELINE_ALL_METHODS_ALIGNED_CAMERA_MAP_VS_GT_DETAIL.csv",

    "data/secondary/AP02_GT_ALIGNED_FULL_MAP_VS_GT_DETAIL.csv",
    "data/secondary/AP02_GT_ALIGNED_FULL_MAP_VS_GT_SUMMARY.csv",

    "data/secondary/AP02_REF14_RELATIVE_MAP_VS_GT_DETAIL.csv",
    "data/secondary/AP02_REF14_RELATIVE_MAP_VS_GT_SUMMARY.csv",
)

def divider(
    char: str = "=",
    width: int = 112,
) -> str:
    return char * width


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open(
        newline="",
        errors="replace",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv_union(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("")
        return

    fields: list[str] = []

    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def number(
    value: Any,
) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def fmt(
    value: Any,
    digits: int = 2,
) -> str:
    parsed = number(value)

    if parsed is None:
        return "-"

    return f"{parsed:.{digits}f}"


def integer(
    value: Any,
) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe(
    value: Any,
) -> str:
    text = "" if value is None else str(value).strip()
    return text if text else "-"


def valid_pair_rows(
    rows: list[dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if (
            row.get("method") == method
            and row.get("status") == "OK"
            and number(
                row.get("translation_error_cm")
            ) is not None
            and number(
                row.get("rotation_error_deg")
            ) is not None
        )
    ]


def normalise_primary_summary(
    summary_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    summary_index = {
        row.get("method", ""): dict(row)
        for row in summary_rows
    }

    result = []

    for method in METHODS:
        row: dict[str, Any] = summary_index.get(
            method,
            {"method": method},
        )

        pairs = valid_pair_rows(
            detail_rows,
            method,
        )

        cameras = set()

        for pair in pairs:
            for key in (
                "from_camera",
                "to_camera",
            ):
                camera = pair.get(key)

                if camera:
                    cameras.add(camera)

        camera_count = integer(
            row.get("camera_count")
        )

        pair_count = integer(
            row.get("pair_count_ok")
        )

        if camera_count is None:
            camera_count = len(cameras)

        if pair_count is None:
            pair_count = len(pairs)

        row["camera_count"] = camera_count
        row["pair_count_ok"] = pair_count
        row["pair_count_total"] = 6

        if not row.get("status"):
            if (
                camera_count == 4
                and pair_count == 6
            ):
                row["status"] = "OK_FULL"
            elif pair_count:
                row["status"] = (
                    f"PARTIAL_{camera_count}_OF_4"
                )
            else:
                row["status"] = "FAILED_NO_PAIR"

        if pairs:
            translations = [
                float(pair["translation_error_cm"])
                for pair in pairs
            ]

            rotations = [
                float(pair["rotation_error_deg"])
                for pair in pairs
            ]

            row.setdefault(
                "mean_pair_t_cm",
                statistics.fmean(translations),
            )

            row.setdefault(
                "median_pair_t_cm",
                statistics.median(translations),
            )

            row.setdefault(
                "mean_pair_r_deg",
                statistics.fmean(rotations),
            )

            row.setdefault(
                "median_pair_r_deg",
                statistics.median(rotations),
            )

        result.append(row)

    return result


def normalise_secondary_summary(
    summary_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    index = {
        row.get("method", ""): dict(row)
        for row in summary_rows
    }

    result = []

    for method in METHODS:
        row: dict[str, Any] = index.get(
            method,
            {"method": method},
        )

        method_rows = [
            detail
            for detail in detail_rows
            if detail.get("method") == method
        ]

        cameras = {
            detail.get("camera")
            for detail in method_rows
            if detail.get("camera")
        }

        if integer(row.get("camera_count")) is None:
            row["camera_count"] = len(cameras)

        result.append(row)

    return result



def canonical_variant_name(
    value: Any,
) -> str:
    return str(value or "").strip().lower()


def detect_audit_variant(
    row: dict[str, Any],
) -> str:
    preferred_fields = (
        "variant",
        "variant_name",
        "dataset",
        "dataset_name",
        "case",
        "case_name",
        "name",
    )

    for field in preferred_fields:
        value = canonical_variant_name(
            row.get(field)
        )

        if value:
            return value

    known_variants = {
        variant.lower()
        for group in GROUPS
        for variant, _parameter in group["variants"]
    }

    for value in row.values():
        candidate = canonical_variant_name(
            value
        )

        for variant in known_variants:
            if variant in candidate:
                return variant

    return ""


def audit_row_is_invalid(
    variant: str,
    row: dict[str, Any],
) -> bool:
    variant = canonical_variant_name(
        variant
    )

    false_fields = (
        "valid",
        "is_valid",
        "gt_valid",
        "evaluation_valid",
    )

    for field in false_fields:
        if field not in row:
            continue

        value = canonical_variant_name(
            row.get(field)
        )

        if value in {
            "0",
            "false",
            "no",
            "invalid",
            "failed",
        }:
            return True

    blob = " ".join(
        canonical_variant_name(value)
        for value in row.values()
    )

    invalid_tokens = (
        "invalid_full",
        "invalid full",
        "gt_failed",
        "gt failed",
        "failed_gt",
        "failed gt",
        "structurally full but gt failed",
        "globally incorrect",
        "mirrored",
        "wrong geometry",
    )

    if any(
        token in blob
        for token in invalid_tokens
    ):
        return True

    # Known result from the committed AP02 validity audit:
    # complete coverage, but globally invalid geometry.
    if variant == "fov_40deg":
        return True

    return False


def audit_reason(
    variant: str,
    row: dict[str, Any],
) -> str:
    preferred_fields = (
        "validity_reason",
        "failure_reason",
        "reason",
        "diagnosis",
        "notes",
        "note",
        "comment",
        "classification",
        "validity",
        "status",
    )

    for field in preferred_fields:
        value = str(
            row.get(field, "")
        ).strip()

        if value:
            return value

    if canonical_variant_name(
        variant
    ) == "fov_40deg":
        return (
            "Full graph coverage but globally invalid "
            "rear-camera/rear-marker geometry."
        )

    return "Marked invalid by AP02 validity audit."


def load_ap02_validity_index(
    preserve_root: Path | None,
) -> dict[str, dict[str, Any]]:
    if preserve_root is None:
        return {}

    path = (
        preserve_root
        / "data"
        / "AP02_VALIDITY_AUDIT.csv"
    )

    rows = read_csv(path)

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in rows:
        variant = detect_audit_variant(
            row
        )

        if variant:
            result[variant] = dict(row)

    return result


def apply_ap02_validity(
    summary_rows: list[dict[str, Any]],
    variant: str,
    validity_index: dict[str, dict[str, Any]],
) -> None:
    key = canonical_variant_name(
        variant
    )

    audit_row = validity_index.get(
        key
    )

    if audit_row is None:
        return

    if not audit_row_is_invalid(
        key,
        audit_row,
    ):
        return

    for row in summary_rows:
        if row.get("method") != "AP02":
            continue

        previous_status = safe(
            row.get("status")
        )

        camera_count = integer(
            row.get("camera_count")
        ) or 0

        pair_count = integer(
            row.get("pair_count_ok")
        ) or 0

        row[
            "coverage_status_before_validity"
        ] = previous_status

        row["validity"] = "INVALID"

        row["validity_reason"] = (
            audit_reason(
                key,
                audit_row,
            )
        )

        if (
            camera_count == 4
            and pair_count == 6
        ):
            row["status"] = (
                "INVALID_FULL_COVERAGE"
            )
        else:
            row["status"] = (
                "INVALID_PARTIAL"
            )

def method_index(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("method")): row
        for row in rows
        if row.get("method") in METHODS
    }


def primary_summary_table(
    rows: list[dict[str, Any]],
) -> list[str]:
    by_method = method_index(rows)

    lines = [
        (
            f"{'Method':8s}"
            f"{'Status':24s}"
            f"{'Cameras':>10s}"
            f"{'Pairs':>9s}"
            f"{'Mean t':>13s}"
            f"{'Median t':>13s}"
            f"{'Mean r':>13s}"
        ),
        divider("-", 90),
    ]

    for method in METHODS:
        row = by_method.get(method, {})

        cameras = integer(
            row.get("camera_count")
        )

        pairs = integer(
            row.get("pair_count_ok")
        )

        lines.append(
            f"{method:8s}"
            f"{safe(row.get('status'))[:24]:24s}"
            f"{str(cameras if cameras is not None else '-') + '/4':>10s}"
            f"{str(pairs if pairs is not None else '-') + '/6':>9s}"
            f"{fmt(row.get('mean_pair_t_cm')) + ' cm':>13s}"
            f"{fmt(row.get('median_pair_t_cm')) + ' cm':>13s}"
            f"{fmt(row.get('mean_pair_r_deg')) + ' deg':>13s}"
        )

    return lines


def secondary_summary_table(
    rows: list[dict[str, Any]],
) -> list[str]:
    by_method = method_index(rows)

    lines = [
        (
            f"{'Method':8s}"
            f"{'Status':24s}"
            f"{'Cameras':>10s}"
            f"{'Mean t':>13s}"
            f"{'Median t':>13s}"
            f"{'Mean r':>13s}"
        ),
        divider("-", 81),
    ]

    for method in METHODS:
        row = by_method.get(method, {})

        cameras = integer(
            row.get("camera_count")
        )

        lines.append(
            f"{method:8s}"
            f"{safe(row.get('status'))[:24]:24s}"
            f"{str(cameras if cameras is not None else '-') + '/4':>10s}"
            f"{fmt(row.get('mean_translation_error_cm')) + ' cm':>13s}"
            f"{fmt(row.get('median_translation_error_cm')) + ' cm':>13s}"
            f"{fmt(row.get('mean_rotation_error_deg')) + ' deg':>13s}"
        )

    return lines


def primary_detail_report(
    title: str,
    records: list[dict[str, Any]],
) -> str:
    lines = [
        title,
        divider(),
        "",
        "Primary metric: static camera-to-camera extrinsics against simulation GT.",
        "No global map alignment is used.",
        "",
    ]

    for record in records:
        lines += [
            divider("#"),
            f"PARAMETER: {record['parameter']}",
            f"VARIANT:   {record['variant']}",
            divider("#"),
            "",
        ]

        for method in METHODS:
            rows = valid_pair_rows(
                record["primary_detail"],
                method,
            )

            lines += [
                method,
                "-" * len(method),
            ]

            if not rows:
                lines += [
                    "- no evaluable pair",
                    "",
                ]
                continue

            lines += [
                (
                    f"{'Pair':12s}"
                    f"{'t error':>12s}"
                    f"{'r error':>12s}"
                    f"{'GT base':>12s}"
                    f"{'Est base':>12s}"
                    f"{'Base err':>12s}"
                    f"{'Dir err':>12s}"
                ),
                divider("-", 84),
            ]

            for row in rows:
                lines.append(
                    f"{safe(row.get('pair')):12s}"
                    f"{fmt(row.get('translation_error_cm')) + ' cm':>12s}"
                    f"{fmt(row.get('rotation_error_deg')) + ' deg':>12s}"
                    f"{fmt(row.get('gt_baseline_m'), 3) + ' m':>12s}"
                    f"{fmt(row.get('est_baseline_m'), 3) + ' m':>12s}"
                    f"{fmt(row.get('baseline_error_cm')) + ' cm':>12s}"
                    f"{fmt(row.get('direction_error_deg')) + ' deg':>12s}"
                )

            lines.append("")

    return "\n".join(lines) + "\n"


def secondary_detail_report(
    title: str,
    records: list[dict[str, Any]],
) -> str:
    lines = [
        title,
        divider(),
        "",
        "Optional secondary:",
        "- best-fit SE(3)-aligned static-camera map vs GT",
        "- no additional scale fitting",
        "- Primary camera-to-camera metrics remain the main comparison",
        "",
    ]

    for record in records:
        lines += [
            divider("#"),
            f"PARAMETER: {record['parameter']}",
            f"VARIANT:   {record['variant']}",
            divider("#"),
            "",
        ]

        for method in METHODS:
            rows = [
                row
                for row in record["secondary_detail"]
                if row.get("method") == method
            ]

            lines += [
                method,
                "-" * len(method),
            ]

            if not rows:
                lines += [
                    "- no map-to-GT result",
                    "",
                ]
                continue

            lines += [
                (
                    f"{'Camera':14s}"
                    f"{'Status':24s}"
                    f"{'t error':>13s}"
                    f"{'r error':>13s}"
                    f"{'Aligned XYZ [m]':>30s}"
                    f"{'GT XYZ [m]':>30s}"
                ),
                divider("-", 124),
            ]

            for row in rows:
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

                lines.append(
                    f"{safe(row.get('camera')):14s}"
                    f"{safe(row.get('status'))[:24]:24s}"
                    f"{fmt(row.get('translation_error_cm')) + ' cm':>13s}"
                    f"{fmt(row.get('rotation_error_deg')) + ' deg':>13s}"
                    f"{aligned:>30s}"
                    f"{gt:>30s}"
                )

            lines.append("")

        full_map = record["ap02_full_map"]

        lines += [
            "AP02 OPTIONAL GT-ALIGNED FULL MAP",
            "-" * 36,
        ]

        if not full_map:
            lines += [
                "- status: NOT_AVAILABLE",
                "- no complete held-out full-map evaluation exists for this run",
                "",
            ]
        else:
            lines += [
                (
                    f"{'Entity':28s}"
                    f"{'Type':18s}"
                    f"{'Used align':>12s}"
                    f"{'t error':>13s}"
                    f"{'r error':>13s}"
                ),
                divider("-", 84),
            ]

            for row in full_map:
                lines.append(
                    f"{safe(row.get('entity_id'))[:28]:28s}"
                    f"{safe(row.get('entity_type'))[:18]:18s}"
                    f"{safe(row.get('used_for_alignment')):>12s}"
                    f"{fmt(row.get('translation_error_cm')) + ' cm':>13s}"
                    f"{fmt(row.get('rotation_error_deg')) + ' deg':>13s}"
                )

            lines.append("")

    return "\n".join(lines) + "\n"


def compact_group_report(
    group: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    lines = [
        f"ABLATION — {group['title']}",
        divider(),
        "",
        "Parameters are listed in physical order.",
        "",
    ]

    for record in records:
        lines += [
            divider("#"),
            f"{record['parameter']} — {record['variant']}",
            divider("#"),
            "",
            "PRIMARY",
            *primary_summary_table(
                record["primary_summary"]
            ),
            "",
            "OPTIONAL SECONDARY",
            *secondary_summary_table(
                record["secondary_summary"]
            ),
            "",
        ]

    lines += [
        "Exact values:",
        (
            f"- ../details/primary/"
            f"{group['slug']}_CAM_TO_CAM.txt"
        ),
        (
            f"- ../details/secondary/"
            f"{group['slug']}_MAP_TO_GT.txt"
        ),
        "",
    ]

    return "\n".join(lines) + "\n"


def method_report(
    method: str,
    baseline_primary: list[dict[str, Any]],
    baseline_secondary: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> str:
    primary_base = method_index(
        baseline_primary
    ).get(method, {})

    secondary_base = method_index(
        baseline_secondary
    ).get(method, {})

    lines = [
        f"{method} RESULTS",
        divider(),
        "",
        "BASELINE PRIMARY",
        divider("-"),
        (
            f"- status: "
            f"{safe(primary_base.get('status'))}"
        ),
        (
            f"- cameras: "
            f"{safe(primary_base.get('camera_count'))}/4"
        ),
        (
            f"- pairs: "
            f"{safe(primary_base.get('pair_count_ok'))}/6"
        ),
        (
            f"- mean translation: "
            f"{fmt(primary_base.get('mean_pair_t_cm'))} cm"
        ),
        (
            f"- mean rotation: "
            f"{fmt(primary_base.get('mean_pair_r_deg'))} deg"
        ),
        "",
        "BASELINE OPTIONAL SECONDARY",
        divider("-"),
        (
            f"- status: "
            f"{safe(secondary_base.get('status'))}"
        ),
        (
            f"- mean translation: "
            f"{fmt(secondary_base.get('mean_translation_error_cm'))} cm"
        ),
        (
            f"- mean rotation: "
            f"{fmt(secondary_base.get('mean_rotation_error_deg'))} deg"
        ),
        "",
        "ABLATIONS",
        divider("-"),
        (
            f"{'Group':24s}"
            f"{'Parameter':22s}"
            f"{'Status':24s}"
            f"{'Cams':>7s}"
            f"{'Pairs':>8s}"
            f"{'Mean t':>12s}"
            f"{'Mean r':>12s}"
        ),
        divider("-", 109),
    ]

    for record in records:
        row = method_index(
            record["primary_summary"]
        ).get(method, {})

        lines.append(
            f"{record['group'][:24]:24s}"
            f"{record['parameter'][:22]:22s}"
            f"{safe(row.get('status'))[:24]:24s}"
            f"{safe(row.get('camera_count')) + '/4':>7s}"
            f"{safe(row.get('pair_count_ok')) + '/6':>8s}"
            f"{fmt(row.get('mean_pair_t_cm')) + ' cm':>12s}"
            f"{fmt(row.get('mean_pair_r_deg')) + ' deg':>12s}"
        )

    lines += [
        "",
        "Detailed primary values:",
        "- details/primary/",
        "",
        "Detailed optional secondary values:",
        "- details/secondary/",
        "",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--core-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--preserve-from",
        type=Path,
    )

    args = parser.parse_args()

    core = args.core_root
    output = args.output_root

    validity_index = load_ap02_validity_index(
        args.preserve_from
    )

    required_core = {
        "primary_summary": (
            core
            / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
        ),
        "primary_detail": (
            core
            / "BASELINE_FINAL_PAIRWISE_DETAIL.csv"
        ),
        "secondary_summary": (
            core
            / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv"
        ),
        "secondary_detail": (
            core
            / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv"
        ),
    }

    for name, path in required_core.items():
        if not path.is_file():
            raise RuntimeError(
                f"Missing core file {name}: {path}"
            )

    if output.exists():
        shutil.rmtree(output)

    for directory in [
        output,
        output / "ablations",
        output / "details/primary",
        output / "details/secondary",
        output / "data/primary",
        output / "data/secondary",
        output / "diagnostics",
        output / "plots",
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if args.preserve_from:
        for relative in PRESERVE_COMPATIBILITY_FILES:
            source = (
                args.preserve_from
                / relative
            )

            if not source.is_file():
                continue

            destination = (
                output
                / relative
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                source,
                destination,
            )

    if args.preserve_from:
        for name in (
            "diagnostics",
            "plots",
        ):
            source = args.preserve_from / name
            destination = output / name

            if source.is_dir():
                shutil.rmtree(
                    destination,
                    ignore_errors=True,
                )

                shutil.copytree(
                    source,
                    destination,
                )

    baseline_primary_detail = read_csv(
        required_core["primary_detail"]
    )

    baseline_primary_summary = (
        normalise_primary_summary(
            read_csv(
                required_core["primary_summary"]
            ),
            baseline_primary_detail,
        )
    )

    baseline_secondary_detail = read_csv(
        required_core["secondary_detail"]
    )

    baseline_secondary_summary = (
        normalise_secondary_summary(
            read_csv(
                required_core["secondary_summary"]
            ),
            baseline_secondary_detail,
        )
    )

    baseline_full_map_path = (
        BUS
        / "02_ref_marker_graph_ba"
        / "08_final_results"
        / "AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv"
    )

    baseline_record = {
        "group": "baseline",
        "variant": "clean_baseline",
        "parameter": "clean baseline",
        "primary_detail": baseline_primary_detail,
        "primary_summary": baseline_primary_summary,
        "secondary_detail": baseline_secondary_detail,
        "secondary_summary": baseline_secondary_summary,
        "ap02_full_map": read_csv(
            baseline_full_map_path
        ),
    }

    records = []

    all_primary_detail = []
    all_primary_summary = []
    all_secondary_detail = []
    all_secondary_summary = []
    all_run_status = []
    all_ap02_full_map = []

    group_records: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for group in GROUPS:
        current_group_records = []

        for variant, parameter in group["variants"]:
            variant_final = (
                BUS
                / "ablation"
                / group["key"]
                / variant
                / "FINAL_RESULTS"
            )

            if not variant_final.is_dir():
                raise RuntimeError(
                    "Missing variant result package: "
                    f"{variant_final}"
                )

            primary_detail = read_csv(
                variant_final
                / "BASELINE_FINAL_PAIRWISE_DETAIL.csv"
            )

            primary_summary = (
                normalise_primary_summary(
                    read_csv(
                        variant_final
                        / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
                    ),
                    primary_detail,
                )
            )

            apply_ap02_validity(
                primary_summary,
                variant,
                validity_index,
            )

            secondary_detail = read_csv(
                variant_final
                / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv"
            )

            secondary_summary = (
                normalise_secondary_summary(
                    read_csv(
                        variant_final
                        / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv"
                    ),
                    secondary_detail,
                )
            )

            full_map = read_csv(
                variant_final
                / "DIAGNOSTIC_AP02_GT_ALIGNED_FULL_MARKER_MAP.csv"
            )

            record = {
                "group": group["key"],
                "variant": variant,
                "parameter": parameter,
                "primary_detail": primary_detail,
                "primary_summary": primary_summary,
                "secondary_detail": secondary_detail,
                "secondary_summary": secondary_summary,
                "ap02_full_map": full_map,
            }

            records.append(record)
            current_group_records.append(record)

            for row in primary_detail:
                all_primary_detail.append({
                    "group": group["key"],
                    "variant": variant,
                    "parameter": parameter,
                    **row,
                })

            for row in primary_summary:
                all_primary_summary.append({
                    "group": group["key"],
                    "variant": variant,
                    "parameter": parameter,
                    **row,
                })

            for row in secondary_detail:
                all_secondary_detail.append({
                    "group": group["key"],
                    "variant": variant,
                    "parameter": parameter,
                    **row,
                })

            for row in secondary_summary:
                all_secondary_summary.append({
                    "group": group["key"],
                    "variant": variant,
                    "parameter": parameter,
                    **row,
                })

            status_path = (
                variant_final
                / "RUN_STATUS.txt"
            )

            status_row: dict[str, Any] = {
                "group": group["key"],
                "variant": variant,
                "parameter": parameter,
            }

            if status_path.is_file():
                for line in status_path.read_text(
                    errors="replace"
                ).splitlines():
                    if "=" not in line:
                        continue

                    key, value = line.split(
                        "=",
                        1,
                    )

                    status_row[
                        key.strip()
                    ] = value.strip()

            all_run_status.append(
                status_row
            )

            for row in full_map:
                all_ap02_full_map.append({
                    "group": group["key"],
                    "variant": variant,
                    "parameter": parameter,
                    **row,
                })
        group_records[
            group["key"]
        ] = current_group_records

    if len(records) != 16:
        raise RuntimeError(
            f"Expected 16 variants, found {len(records)}"
        )

    # Machine-readable baseline.
    write_csv_union(
        output
        / "data/primary/BASELINE_SUMMARY.csv",
        baseline_primary_summary,
    )

    write_csv_union(
        output
        / "data/primary/BASELINE_DETAIL.csv",
        baseline_primary_detail,
    )

    write_csv_union(
        output
        / "data/secondary/"
        "BASELINE_ALIGNED_CAMERA_MAP_SUMMARY.csv",
        baseline_secondary_summary,
    )

    write_csv_union(
        output
        / "data/secondary/"
        "BASELINE_ALIGNED_CAMERA_MAP_DETAIL.csv",
        baseline_secondary_detail,
    )

    write_csv_union(
        output
        / "data/secondary/"
        "BASELINE_AP02_GT_ALIGNED_FULL_MAP.csv",
        baseline_record["ap02_full_map"],
    )

    # Machine-readable ablations.
    write_csv_union(
        output
        / "data/primary/"
        "ALL_ABLATIONS_SUMMARY.csv",
        all_primary_summary,
    )

    write_csv_union(
        output
        / "data/primary/"
        "ALL_ABLATIONS_DETAIL.csv",
        all_primary_detail,
    )

    write_csv_union(
        output
        / "data/secondary/"
        "ALL_ABLATIONS_ALIGNED_CAMERA_MAP_SUMMARY.csv",
        all_secondary_summary,
    )

    write_csv_union(
        output
        / "data/secondary/"
        "ALL_ABLATIONS_ALIGNED_CAMERA_MAP_DETAIL.csv",
        all_secondary_detail,
    )

    write_csv_union(
        output
        / "data/secondary/"
        "ALL_ABLATIONS_AP02_GT_ALIGNED_FULL_MAP.csv",
        all_ap02_full_map,
    )

    write_csv_union(
        output
        / "data/ALL_ABLATIONS_RUN_STATUS.csv",
        all_run_status,
    )

    # Baseline readable reports.
    (
        output
        / "details/primary/"
        "00_BASELINE_CAM_TO_CAM.txt"
    ).write_text(
        primary_detail_report(
            "BASELINE — DETAILED CAMERA-TO-CAMERA RESULTS",
            [baseline_record],
        )
    )

    (
        output
        / "details/secondary/"
        "00_BASELINE_MAP_TO_GT.txt"
    ).write_text(
        secondary_detail_report(
            "BASELINE — DETAILED OPTIONAL MAP-TO-GT RESULTS",
            [baseline_record],
        )
    )

    # Group reports.
    for group in GROUPS:
        current = group_records[
            group["key"]
        ]

        (
            output
            / "details/primary"
            / f"{group['slug']}_CAM_TO_CAM.txt"
        ).write_text(
            primary_detail_report(
                (
                    f"{group['title']} — "
                    "DETAILED CAMERA-TO-CAMERA RESULTS"
                ),
                current,
            )
        )

        (
            output
            / "details/secondary"
            / f"{group['slug']}_MAP_TO_GT.txt"
        ).write_text(
            secondary_detail_report(
                (
                    f"{group['title']} — "
                    "DETAILED OPTIONAL MAP-TO-GT RESULTS"
                ),
                current,
            )
        )

        (
            output
            / "ablations"
            / f"{group['slug']}_ALL_METHODS.txt"
        ).write_text(
            compact_group_report(
                group,
                current,
            )
        )

    # Baseline overview.
    baseline_lines = [
        "BASELINE — AP01 / AP02 / AP03",
        divider(),
        "",
        "PRIMARY",
        divider("-"),
        *primary_summary_table(
            baseline_primary_summary
        ),
        "",
        "OPTIONAL SECONDARY",
        divider("-"),
        *secondary_summary_table(
            baseline_secondary_summary
        ),
        "",
        "Exact values:",
        "- details/primary/00_BASELINE_CAM_TO_CAM.txt",
        "- details/secondary/00_BASELINE_MAP_TO_GT.txt",
        "",
    ]

    (
        output
        / "01_BASELINE_ALL_METHODS.txt"
    ).write_text(
        "\n".join(baseline_lines) + "\n"
    )

    # Overall ablation overview.
    overall_lines = [
        "ALL ABLATIONS — AP01 / AP02 / AP03",
        divider(),
        "",
        "All parameters are listed in physical order.",
        "",
        "Validity note:",
        "- INVALID_FULL_COVERAGE means complete numeric coverage,",
        "  but the result failed the independent geometric validity audit.",
        "",
        (
            f"{'Group':24s}"
            f"{'Parameter':22s}"
            f"{'Method':8s}"
            f"{'Status':24s}"
            f"{'Cams':>7s}"
            f"{'Pairs':>8s}"
            f"{'Mean t':>12s}"
            f"{'Mean r':>12s}"
        ),
        divider("-", 109),
    ]

    for record in records:
        by_method = method_index(
            record["primary_summary"]
        )

        for method in METHODS:
            row = by_method.get(method, {})

            overall_lines.append(
                f"{record['group'][:24]:24s}"
                f"{record['parameter'][:22]:22s}"
                f"{method:8s}"
                f"{safe(row.get('status'))[:24]:24s}"
                f"{safe(row.get('camera_count')) + '/4':>7s}"
                f"{safe(row.get('pair_count_ok')) + '/6':>8s}"
                f"{fmt(row.get('mean_pair_t_cm')) + ' cm':>12s}"
                f"{fmt(row.get('mean_pair_r_deg')) + ' deg':>12s}"
            )

        overall_lines.append("")

    overall_lines += [
        "Detailed reports:",
        "- ablations/",
        "- details/primary/",
        "- details/secondary/",
        "",
    ]

    (
        output
        / "02_ALL_ABLATIONS_ALL_METHODS.txt"
    ).write_text(
        "\n".join(overall_lines) + "\n"
    )

    # Method reports.
    method_outputs = {
        "AP01": "03_AP01_RESULTS.txt",
        "AP02": "04_AP02_RESULTS.txt",
        "AP03": "05_AP03_RESULTS.txt",
    }

    for method, filename in method_outputs.items():
        (
            output / filename
        ).write_text(
            method_report(
                method,
                baseline_primary_summary,
                baseline_secondary_summary,
                records,
            )
        )

    # Start document.
    (
        output / "00_READ_ME_FIRST.txt"
    ).write_text(
        "\n".join([
            "CAMERA-RIG CALIBRATION — FINAL RESULTS",
            divider(),
            "",
            "Main:",
            "- 01_BASELINE_ALL_METHODS.txt",
            "- 02_ALL_ABLATIONS_ALL_METHODS.txt",
            "- 03_AP01_RESULTS.txt",
            "- 04_AP02_RESULTS.txt",
            "- 05_AP03_RESULTS.txt",
            "",
            "Detailed primary:",
            "- details/primary/",
            "",
            "Optional secondary:",
            "- details/secondary/",
            "",
            "Machine-readable:",
            "- data/primary/",
            "- data/secondary/",
            "",
            "Primary remains the direct camera-to-camera comparison.",
            "Secondary remains an optional map-level evaluation.",
            "",
        ])
    )

    # Deterministic manifest.
    files = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    )

    (
        output / "MANIFEST.json"
    ).write_text(
        json.dumps(
            {
                "schema": "camera_rig_final_results_v1",
                "baseline_methods": list(METHODS),
                "ablation_variant_count": len(records),
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "[OK] Built canonical final results:"
        f" {output}"
    )

    print(
        f"[OK] Variants: {len(records)} / 16"
    )


if __name__ == "__main__":
    main()
