#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_FINAL = Path(
    "results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
)

AP02_ROOT = Path(
    "results/bus_real_data/02_ref_marker_graph_ba"
)

TABLES = [
    "BASELINE_FINAL_PAIRWISE_SUMMARY.csv",
    "BASELINE_FINAL_PAIRWISE_DETAIL.csv",
    "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv",
    "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv",
]

METHOD_ORDER = {
    "AP01": 0,
    "AP02": 1,
    "AP03": 2,
}

PAIR_ORDER = {
    "cam0-cam1": 0,
    "cam0-cam3": 1,
    "cam0-cam5": 2,
    "cam1-cam3": 3,
    "cam1-cam5": 4,
    "cam3-cam5": 5,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(
        newline="",
        errors="replace",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields: list[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open(
        "w",
        newline="",
    ) as file:
        if not fields:
            return

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            METHOD_ORDER.get(
                row.get("method", ""),
                99,
            ),
            PAIR_ORDER.get(
                row.get("pair", ""),
                99,
            ),
            row.get("camera", ""),
        ),
    )


def archive_v1(final_root: Path) -> None:
    archive = final_root / "AP02_V1_ARCHIVE"

    if archive.exists():
        return

    archive.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_ap02 = final_root / "AP02"

    if old_ap02.is_dir():
        shutil.copytree(
            old_ap02,
            archive / "AP02",
            dirs_exist_ok=True,
        )

    for name in TABLES:
        rows = [
            row
            for row in read_csv(final_root / name)
            if row.get("method") == "AP02"
        ]

        if rows:
            write_csv(
                archive / name,
                rows,
            )

    status = final_root / "RUN_STATUS.txt"

    if status.exists():
        shutil.copy2(
            status,
            archive / "RUN_STATUS.txt",
        )


def merge_table(
    final_root: Path,
    name: str,
) -> list[dict[str, str]]:
    source = CANONICAL_FINAL / name
    target = final_root / name

    new_ap02 = [
        row
        for row in read_csv(source)
        if row.get("method") == "AP02"
    ]

    if not new_ap02:
        raise RuntimeError(
            f"No AP02 rows generated in {source}"
        )

    previous = read_csv(target)

    retained = [
        row
        for row in previous
        if row.get("method") != "AP02"
    ]

    merged = sort_rows(
        retained + new_ap02
    )

    write_csv(
        target,
        merged,
    )

    return new_ap02


def format_value(value: str) -> str:
    if value in ("", None):
        return "-"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def regenerate_primary_report(final_root: Path) -> None:
    summary = read_csv(
        final_root
        / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    )

    detail = read_csv(
        final_root
        / "BASELINE_FINAL_PAIRWISE_DETAIL.csv"
    )

    lines = [
        "PARTIAL-AWARE PAIRWISE STATIC-CAMERA EVALUATION",
        "================================================",
        "",
        (
            "OK_FULL requires all four cameras and all six "
            "camera pairs. PARTIAL results retain every "
            "evaluable camera pair."
        ),
        "",
        "Summary:",
    ]

    for row in sort_rows(summary):
        lines.append(
            f"- {row.get('method', '')}: "
            f"{row.get('status', '')} | "
            f"cameras={row.get('camera_count', '-')}/4 | "
            f"pairs={row.get('pair_count_ok', '-')}/6 | "
            f"mean={format_value(row.get('mean_pair_t_cm', ''))} cm / "
            f"{format_value(row.get('mean_pair_r_deg', ''))} deg | "
            f"missing={row.get('missing_cameras', '') or '-'}"
        )

        if row.get("failure_reason"):
            lines.append(
                f"  reason: {row['failure_reason']}"
            )

    lines += [
        "",
        "Pairwise detail:",
    ]

    for method in [
        "AP01",
        "AP02",
        "AP03",
    ]:
        lines += [
            "",
            method,
        ]

        for row in sort_rows([
            item
            for item in detail
            if item.get("method") == method
        ]):
            if row.get("status") == "OK":
                lines.append(
                    f"  {row.get('pair', '')}: "
                    f"{format_value(row.get('translation_error_cm', ''))} cm / "
                    f"{format_value(row.get('rotation_error_deg', ''))} deg"
                )
            else:
                lines.append(
                    f"  {row.get('pair', '')}: "
                    f"{row.get('status', '')} "
                    f"({row.get('note', '')})"
                )

    (
        final_root
        / "BASELINE_FINAL_CLEAN_COMPARISON.txt"
    ).write_text(
        "\n".join(lines) + "\n"
    )


def regenerate_secondary_report(final_root: Path) -> None:
    summary = read_csv(
        final_root
        / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv"
    )

    detail = read_csv(
        final_root
        / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv"
    )

    lines = [
        "PARTIAL-AWARE SECONDARY REF14/WORLD CAMERA-MAP EVALUATION",
        "==========================================================",
        "",
        (
            "SE(3) alignment is evaluated only when at least "
            "three static cameras are available."
        ),
    ]

    for method in [
        "AP01",
        "AP02",
        "AP03",
    ]:
        method_summary = next(
            (
                row
                for row in summary
                if row.get("method") == method
            ),
            None,
        )

        lines += [
            "",
            method,
            "-" * len(method),
        ]

        if method_summary is None:
            lines.append("status: MISSING")
            continue

        lines.append(
            "available cameras: "
            + str(
                method_summary
                .get("available_cameras", "")
                .split(";")
                if method_summary.get(
                    "available_cameras"
                )
                else []
            )
        )

        lines.append(
            "missing cameras: "
            + str(
                method_summary
                .get("missing_cameras", "")
                .split(";")
                if method_summary.get(
                    "missing_cameras"
                )
                else []
            )
        )

        method_detail = [
            row
            for row in detail
            if row.get("method") == method
        ]

        for row in method_detail:
            if row.get("status") == "OK":
                lines.append(
                    f"- {row.get('camera', '')}: "
                    f"{format_value(row.get('translation_error_cm', ''))} cm / "
                    f"{format_value(row.get('rotation_error_deg', ''))} deg"
                )

        lines.append(
            "summary: "
            f"{method_summary.get('status', '')} | mean "
            f"{format_value(method_summary.get('mean_translation_error_cm', ''))} cm / "
            f"{format_value(method_summary.get('mean_rotation_error_deg', ''))} deg"
        )

    (
        final_root
        / "SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt"
    ).write_text(
        "\n".join(lines) + "\n"
    )


def update_status(
    final_root: Path,
    ap02_status: str,
) -> None:
    path = final_root / "RUN_STATUS.txt"

    values: dict[str, str] = {}

    if path.exists():
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue

            key, value = line.split(
                "=",
                1,
            )

            values[key] = value

    values["variant"] = values.get(
        "variant",
        final_root.parent.name,
    )

    values["AP01_STATUS"] = values.get(
        "AP01_STATUS",
        "UNKNOWN",
    )

    values["AP02_STATUS"] = ap02_status

    values["AP03_STATUS"] = values.get(
        "AP03_STATUS",
        "UNKNOWN",
    )

    values["PAIRWISE_STATUS"] = "OK"
    values["SECONDARY_STATUS"] = "OK"

    ordered = [
        "variant",
        "AP01_STATUS",
        "AP02_STATUS",
        "AP03_STATUS",
        "PAIRWISE_STATUS",
        "SECONDARY_STATUS",
    ]

    path.write_text(
        "\n".join(
            f"{key}={values[key]}"
            for key in ordered
        )
        + "\n"
    )


def copy_outputs(
    final_root: Path,
) -> None:
    canonical_ap02 = CANONICAL_FINAL / "AP02"

    if canonical_ap02.is_dir():
        target = final_root / "AP02"

        shutil.rmtree(
            target,
            ignore_errors=True,
        )

        shutil.copytree(
            canonical_ap02,
            target,
        )

    diagnostics = final_root / "AP02_V2_DIAGNOSTICS"

    shutil.rmtree(
        diagnostics,
        ignore_errors=True,
    )

    diagnostics.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in [
        "05_graph_initialization",
        "07_graph_ba",
        "08_final_results",
    ]:
        source = AP02_ROOT / name

        if source.is_dir():
            shutil.copytree(
                source,
                diagnostics / name,
                dirs_exist_ok=True,
            )

    diagnostic_files = {
        (
            AP02_ROOT
            / "08_final_results"
            / "AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.txt"
        ): (
            final_root
            / "DIAGNOSTIC_AP02_GT_ALIGNED_FULL_MARKER_MAP.txt"
        ),
        (
            AP02_ROOT
            / "08_final_results"
            / "AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv"
        ): (
            final_root
            / "DIAGNOSTIC_AP02_GT_ALIGNED_FULL_MARKER_MAP.csv"
        ),
    }

    # Remove outputs from an older AP02 evaluator before copying the
    # current run. Otherwise an evaluator failure can leave a stale CSV
    # that looks like a current partial-map result.
    for destination in diagnostic_files.values():
        destination.unlink(missing_ok=True)

    for source, destination in diagnostic_files.items():
        if source.exists():
            shutil.copy2(
                source,
                destination,
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant-root",
        required=True,
    )

    parser.add_argument(
        "--pipeline-rc",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    variant_root = Path(
        args.variant_root
    )

    final_root = (
        variant_root
        / "FINAL_RESULTS"
    )

    final_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_v1(final_root)

    new_rows: dict[str, list[dict[str, str]]] = {}

    for name in TABLES:
        new_rows[name] = merge_table(
            final_root,
            name,
        )

    primary_summary = new_rows[
        "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    ][0]

    ap02_status = primary_summary.get(
        "status",
        "UNKNOWN",
    )

    copy_outputs(final_root)
    update_status(
        final_root,
        ap02_status,
    )

    regenerate_primary_report(final_root)
    regenerate_secondary_report(final_root)

    metadata = {
        "variant": variant_root.name,
        "method": "AP02",
        "version": "v2",
        "pipeline_return_code": args.pipeline_rc,
        "primary_status": ap02_status,
        "graph_initialization": (
            "rooted_maximum_spanning_tree"
        ),
        "moving_frame_selection": (
            "marker_aware_smart_selection"
        ),
        "top_per_marker": 8,
        "top_per_marker_pair": 4,
        "uniform_stride_used": False,
        "updated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    (
        final_root
        / "AP02_V2_METADATA.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print(
        f"[OK] Updated AP02 v2 results: {variant_root}"
    )

    print(
        "[OK] AP02 status:",
        ap02_status,
    )

    print(
        "[OK] mean pair translation [cm]:",
        primary_summary.get(
            "mean_pair_t_cm",
            "",
        ),
    )

    print(
        "[OK] mean pair rotation [deg]:",
        primary_summary.get(
            "mean_pair_r_deg",
            "",
        ),
    )


if __name__ == "__main__":
    main()
