#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BUS = ROOT / "results/bus_real_data"
FINAL99 = BUS / "99_FINAL_RESULTS_FOR_REPORT"
EVALUATOR = (
    ROOT
    / "run/bus_real_data/approach2_ref_marker_graph_ba/"
    / "09_eval_ap02_gt_aligned_full_map.py"
)

SPEC = importlib.util.spec_from_file_location("ap02_eval09", EVALUATOR)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import evaluator: {EVALUATOR}")
EV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EV)

REF_KEY = ("aruco_marker", int(EV.REF_MARKER_ID))
EXPECTED_MARKERS = list(EV.SIM_MARKER_IDS)
STATIC_CAMERAS = list(EV.STATIC_CAMERAS)
BEGIN = "=== AP02 REF14-ANCHORED AVAILABLE MAPS BEGIN ==="
END = "=== AP02 REF14-ANCHORED AVAILABLE MAPS END ==="

GROUPS = [
    {
        "report": FINAL99 / "details/secondary/00_BASELINE_MAP_TO_GT.txt",
        "records": [
            (
                "clean_baseline",
                "clean baseline",
                BUS / "02_ref_marker_graph_ba/08_final_results",
                BUS / "02_ref_marker_graph_ba/08_final_results",
            )
        ],
    },
    {
        "report": FINAL99 / "details/secondary/01_FOV_MAP_TO_GT.txt",
        "root": BUS / "ablation/moving_cam/fov",
        "variants": [
            ("fov_40deg", "40 deg"),
            ("fov_69deg_baseline", "69 deg baseline"),
            ("fov_100deg", "100 deg"),
            ("fov_140deg_extreme", "140 deg"),
        ],
    },
    {
        "report": FINAL99 / "details/secondary/02_MOTION_BLUR_MAP_TO_GT.txt",
        "root": BUS / "ablation/moving_cam/motion_blur",
        "variants": [
            ("moving_blur_k00_baseline", "kernel 0 baseline"),
            ("moving_blur_k09_mild", "kernel 9"),
            ("moving_blur_k21_strong", "kernel 21"),
            ("moving_blur_k41_extreme", "kernel 41"),
        ],
    },
    {
        "report": FINAL99 / "details/secondary/03_RESOLUTION_MAP_TO_GT.txt",
        "root": BUS / "ablation/moving_cam/res",
        "variants": [
            ("moving_res_160x90_extreme_pixel", "160x90"),
            ("moving_res_320x180_low", "320x180"),
            ("moving_res_1280x720_baseline", "1280x720 baseline"),
            ("moving_res_2560x1440_upscaled", "2560x1440"),
        ],
    },
    {
        "report": FINAL99 / "details/secondary/04_LIGHTING_MAP_TO_GT.txt",
        "root": BUS / "ablation/world/lighting",
        "variants": [
            ("ceiling_dark_extreme", "dark extreme"),
            ("ceiling_low", "low"),
            ("ceiling_normal", "normal baseline"),
            ("ceiling_bright", "bright"),
        ],
    },
    {
        "report": FINAL99 / "details/secondary/05_ROUTE_PATH_MAP_TO_GT.txt",
        "root": BUS / "ablation/world/route",
        "variants": [("route1", "Route 1"), ("route2", "Route 2")],
    },
    {
        "report": FINAL99 / "details/secondary/06_FRAME_DENSITY_MAP_TO_GT.txt",
        "root": BUS / "ablation/moving_cam/density",
        "variants": [
            ("density_route2_125pct_recaptured", "125% recaptured"),
            ("density_stride_1_100pct", "100%"),
            ("density_stride_2_50pct", "50%"),
            ("density_stride_4_25pct", "25%"),
            ("density_stride_8_12p5pct", "12.5%"),
            ("density_stride_8_offset4", "12.5% offset 4"),
            ("density_stride_16_6p25pct", "6.25%"),
        ],
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    parsed = number(value)
    return "-" if parsed is None else f"{parsed:.{digits}f}{suffix}"


def load_estimates(diag: Path) -> dict[tuple[str, Any], Any]:
    cameras = diag / "ap02_with_moving_static_camera_poses_ref_marker.csv"
    markers = diag / "ap02_with_moving_marker_poses_ref_marker.csv"
    if not cameras.is_file() or not markers.is_file():
        return {}
    est: dict[tuple[str, Any], Any] = {}
    for row in read_csv(cameras):
        camera = row.get("entity_id", "")
        if camera:
            est[("static_camera", camera)] = EV.T_from_ap02_row(row)
    for row in read_csv(markers):
        try:
            marker_id = int(float(row.get("entity_id", "")))
        except (TypeError, ValueError):
            continue
        est[("aruco_marker", marker_id)] = EV.T_from_ap02_row(row)
    return est


def evaluate(diag: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt = EV.load_gt_world_entities()
    est = load_estimates(diag)
    available_cameras = [
        camera for camera in STATIC_CAMERAS
        if ("static_camera", camera) in est
    ]
    available_markers = [
        marker for marker in EXPECTED_MARKERS
        if ("aruco_marker", marker) in est
    ]
    meta = {
        "available_cameras": available_cameras,
        "missing_cameras": [
            camera for camera in STATIC_CAMERAS
            if camera not in available_cameras
        ],
        "available_marker_ids": available_markers,
        "missing_marker_ids": [
            marker for marker in EXPECTED_MARKERS
            if marker not in available_markers
        ],
        "ref14_available": REF_KEY in est,
    }
    if REF_KEY not in est:
        return [], meta

    T_world_from_local = gt[REF_KEY] @ EV.invT(est[REF_KEY])
    keys = [
        ("static_camera", camera) for camera in available_cameras
    ] + [
        ("aruco_marker", marker) for marker in available_markers
    ]
    rows: list[dict[str, Any]] = []
    for key in keys:
        entity_type, entity_raw = key
        T_local = est[key]
        T_aligned = T_world_from_local @ T_local
        T_gt = gt[key]
        marker_id: Any = ""
        entity_id = str(entity_raw)
        if entity_type == "aruco_marker":
            marker_id = int(entity_raw)
            entity_id = EV.marker_name(marker_id)
        row: dict[str, Any] = {
            "approach": "AP02_ref_marker_graph_ba",
            "evaluation": "ref14_anchored_available_map",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "marker_id": marker_id,
            "used_for_alignment": "yes" if key == REF_KEY else "no",
            "alignment_frame": "GT_world_from_est_local_using_ref14_pose_only",
            "translation_error_cm": EV.trans_error_cm(T_aligned, T_gt),
            "rotation_error_deg": EV.rot_error_deg(T_aligned, T_gt),
        }
        row.update(EV.delta_columns(T_aligned, T_gt))
        row.update(EV.pose_columns("est_local", T_local))
        row.update(EV.pose_columns("est_gt_aligned", T_aligned))
        row.update(EV.pose_columns("gt_world", T_gt))
        rows.append(row)
    return rows, meta


def table(rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    lines = [
        f"available cameras: {meta['available_cameras']}",
        f"missing cameras: {meta['missing_cameras']}",
        f"available marker IDs: {meta['available_marker_ids']}",
        f"missing marker IDs: {meta['missing_marker_ids']}",
        "",
    ]
    if not rows:
        lines += [
            "NOT AVAILABLE: reference marker 14 is not present in the AP02 estimate.",
            "A REF14-anchored map cannot be defined without marker 14.",
            "",
        ]
        return lines
    lines += [
        (
            f"{'Entity':24s}{'Type':16s}{'Used align':>12s}"
            f"{'t error':>12s}{'r error':>12s}"
            f"{'Aligned XYZ [m]':>31s}{'GT XYZ [m]':>31s}"
            f"{'Delta XYZ [cm]':>31s}"
        ),
        "-" * 169,
    ]
    for row in rows:
        aligned = (
            f"({fmt(row.get('est_gt_aligned_x_m'), 3)}, "
            f"{fmt(row.get('est_gt_aligned_y_m'), 3)}, "
            f"{fmt(row.get('est_gt_aligned_z_m'), 3)})"
        )
        gt = (
            f"({fmt(row.get('gt_world_x_m'), 3)}, "
            f"{fmt(row.get('gt_world_y_m'), 3)}, "
            f"{fmt(row.get('gt_world_z_m'), 3)})"
        )
        delta = (
            f"({fmt(row.get('delta_x_cm'), 2)}, "
            f"{fmt(row.get('delta_y_cm'), 2)}, "
            f"{fmt(row.get('delta_z_cm'), 2)})"
        )
        lines.append(
            f"{str(row.get('entity_id', '-'))[:24]:24s}"
            f"{str(row.get('entity_type', '-'))[:16]:16s}"
            f"{str(row.get('used_for_alignment', '-')):>12s}"
            f"{fmt(row.get('translation_error_cm'), 2, ' cm'):>12s}"
            f"{fmt(row.get('rotation_error_deg'), 2, ' deg'):>12s}"
            f"{aligned:>31s}{gt:>31s}{delta:>31s}"
        )
    lines.append("")
    return lines


def strip_old_section(text: str) -> str:
    if BEGIN not in text:
        return text.rstrip() + "\n"
    return text.split(BEGIN, 1)[0].rstrip() + "\n"


def records_for_group(group: dict[str, Any]) -> list[tuple[str, str, Path, Path]]:
    if "records" in group:
        return list(group["records"])
    records = []
    root = group["root"]
    for variant, parameter in group["variants"]:
        final = root / variant / "FINAL_RESULTS"
        diag = final / "AP02_V2_DIAGNOSTICS/08_final_results"
        records.append((variant, parameter, final, diag))
    return records


def main() -> None:
    FINAL99.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, Any]] = []

    for group in GROUPS:
        report = group["report"]
        if not report.is_file():
            raise RuntimeError(f"Missing readable secondary report: {report}")
        section = [
            BEGIN,
            "",
            "AP02 REF14-ANCHORED AVAILABLE CAMERA + MARKER MAPS",
            "=" * 169,
            "",
            "Marker 14 alone defines the world transform in this section.",
            "Therefore marker 14 has zero residual by construction and is marked 'yes'.",
            "Every camera and every other available marker is marked 'no' and evaluated relative to marker 14.",
            "This is separate from the preceding best-fit SE(3) camera-map/full-map diagnostic.",
            "",
        ]

        for variant, parameter, final, diag in records_for_group(group):
            rows, meta = evaluate(diag)
            output_csv = final / "DIAGNOSTIC_AP02_REF14_ANCHORED_AVAILABLE_MAP.csv"
            output_txt = final / "DIAGNOSTIC_AP02_REF14_ANCHORED_AVAILABLE_MAP.txt"
            output_meta = final / "DIAGNOSTIC_AP02_REF14_ANCHORED_AVAILABLE_MAP_metadata.json"
            write_csv(output_csv, rows)
            body = [
                "AP02 REF14-ANCHORED AVAILABLE CAMERA + MARKER MAP",
                "=" * 169,
                "",
                f"variant: {variant}",
                *table(rows, meta),
            ]
            output_txt.write_text("\n".join(body) + "\n", encoding="utf-8")
            output_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

            section += [
                "#" * 169,
                f"PARAMETER: {parameter}",
                f"VARIANT:   {variant}",
                "#" * 169,
                "",
                *table(rows, meta),
            ]
            audit.append({
                "variant": variant,
                "available_cameras": ";".join(meta["available_cameras"]),
                "missing_cameras": ";".join(meta["missing_cameras"]),
                "available_marker_ids": ";".join(map(str, meta["available_marker_ids"])),
                "missing_marker_ids": ";".join(map(str, meta["missing_marker_ids"])),
                "ref14_available": "yes" if meta["ref14_available"] else "no",
                "row_count": len(rows),
            })

        section += [END, ""]
        base = strip_old_section(report.read_text(encoding="utf-8", errors="replace"))
        report.write_text(base + "\n".join(section), encoding="utf-8")
        print(f"[OK] updated {report}")

    write_csv(FINAL99 / "AP02_REF14_AVAILABLE_MAP_AUDIT.csv", audit)
    audit_lines = [
        "AP02 REF14-ANCHORED AVAILABLE MAP AUDIT",
        "=" * 120,
        "",
    ]
    for row in audit:
        audit_lines += [
            row["variant"],
            f"  cameras: {row['available_cameras'] or '-'}",
            f"  marker IDs: {row['available_marker_ids'] or '-'}",
            f"  missing marker IDs: {row['missing_marker_ids'] or '-'}",
            f"  ref14 available: {row['ref14_available']}",
            f"  evaluated rows: {row['row_count']}",
            "",
        ]
    (FINAL99 / "AP02_REF14_AVAILABLE_MAP_AUDIT.txt").write_text(
        "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )
    print("[OK] REF14-anchored partial maps written for all available variants")


if __name__ == "__main__":
    main()
