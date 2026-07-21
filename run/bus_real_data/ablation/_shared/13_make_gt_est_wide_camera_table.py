#!/usr/bin/env python3
from pathlib import Path
import os
import re
import csv
import math

STUDY_ROOT = Path(os.environ.get("STUDY_ROOT", "results/bus_real_data/ablation/moving_cam/res"))
FINAL = STUDY_ROOT / "final_results"

OUT_CSV = FINAL / "GT_EST_WIDE_CAMERA_ERROR_TABLE.csv"
OUT_TXT = FINAL / "GT_EST_WIDE_CAMERA_ERROR_TABLE.txt"
OUT_LONG_CSV = FINAL / "GT_EST_LONG_CAMERA_ERROR_TABLE.csv"

CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

GT_REF14_XYZ = {
    "cam_edge_0": (0.588, -1.735, 2.087),
    "cam_edge_1": (-3.974, 0.475, 2.132),
    "cam_edge_3": (-3.227, -1.563, 2.054),
    "cam_edge_5": (5.859, 0.348, 2.139),
}

CANONICAL_AP01 = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP01/AP01_FINAL_RESULT.txt")

VARIANT_ORDER = {
    "res_320x180_extreme": 0,
    "res_640x360": 1,
    "res_960x540": 2,
    "res_1280x720_baseline": 3,
    "res_1920x1080": 4,
    "fov_40deg": 10,
    "fov_50deg": 11,
    "fov_60deg": 12,
    "fov_69deg_baseline": 13,
    "fov_80deg": 14,
    "fov_90deg": 15,
    "fov_100deg": 16,
    "fov_110deg": 17,
    "fov_120deg": 18,
    "fov_140deg_extreme": 19,
}

APPROACH_ORDER = {
    "AP01": 0,
    "AP02": 1,
    "AP03_MULTI": 2,
    "AP03_SINGLE_REF14": 3,
}

FLOAT = r"([0-9.+\-eE]+)"

PIPE_ERROR_ROW_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*" + FLOAT + r"\s*\|\s*" + FLOAT + r"\s*\|"
)


METHOD_ERROR_ROW_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    r"([^|]+?)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|",
    re.IGNORECASE,
)


AP03_COMBINED_ROW_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|\s*"   # single_err_cm
    r"([0-9.+\-eE]+)\s*\|\s*"   # multi_err_cm
    r"([0-9.+\-eE]+)\s*\|\s*"   # gain_cm
    r"([0-9.+\-eE]+)\s*\|\s*"   # single_rot
    r"([0-9.+\-eE]+)\s*\|",     # multi_rot
    re.IGNORECASE,
)

EST_XYZ_ROW_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    r"\(\s*([0-9.+\-eE]+)\s*,\s*([0-9.+\-eE]+)\s*,\s*([0-9.+\-eE]+)\s*\)\s*\|",
    re.IGNORECASE,
)

def read_text(p: Path) -> str:
    return p.read_text(errors="replace")

def variant_from_report(p: Path) -> str:
    return p.name.replace("_FINAL_RESULT.txt", "")

def fmt(v):
    if v == "" or v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)

def dist_cm(a, b):
    return 100.0 * math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )

def mean(vals):
    clean = []
    for v in vals:
        if v == "" or v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isnan(fv):
            clean.append(fv)
    if not clean:
        return ""
    return round(sum(clean) / len(clean), 3)

def extract_block(text: str, start_marker: str, end_markers):
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = len(text)
    for marker in end_markers:
        idx = text.find(marker, start + len(start_marker))
        if idx >= 0:
            end = min(end, idx)
    return text[start:end]

def extract_ap01_section(text: str):
    starts = [
        "AP01 FINAL RESULT",
        "FINAL CAMERA RIG CALIBRATION REPORT",
    ]
    start = -1
    for marker in starts:
        idx = text.find(marker)
        if idx >= 0:
            start = idx
            break
    if start < 0:
        return ""

    end = len(text)
    for marker in [
        "================ AP02",
        "AP02 FINAL",
        "3. AP02",
        "=== AP02",
        "\nAP02 ",
    ]:
        idx = text.find(marker, start + 1)
        if idx >= 0:
            end = min(end, idx)

    return text[start:end]

def parse_pipe_error_rows(block: str):
    rows = {}
    for line in block.splitlines():
        m = PIPE_ERROR_ROW_RE.search(line)
        if not m:
            continue
        cam = m.group(1)
        rows[cam] = {
            "camera": cam,
            "t_cm": float(m.group(2)),
            "r_deg": float(m.group(3)),
            "note": "",
        }
    return rows

def parse_ap01_true_eval_table(text: str):
    section = extract_ap01_section(text)
    if not section:
        section = text

    idx = section.find("Evaluation-only: comparison against GT camera map in Ref14 frame")
    if idx < 0:
        idx = section.find("cam_gt_vs_est_cm")
    if idx < 0:
        return {}

    block = section[idx:idx + 30000]
    end = len(block)
    for marker in [
        "Method per camera:",
        "================ AP02",
        "AP02 FINAL",
        "3. AP02",
        "AP03 FINAL",
    ]:
        j = block.find(marker)
        if j >= 0 and j > 0:
            end = min(end, j)

    rows = parse_pipe_error_rows(block[:end])
    for r in rows.values():
        r["note"] = "AP01 true Ref14-frame GT evaluation table"
    return rows

def parse_ap01_est_map_translation_only(text: str):
    section = extract_ap01_section(text)
    if not section:
        return {}

    idx = section.find("AP01 real-life method output: estimated camera map in local Ref14 frame")
    if idx < 0:
        idx = section.find("est_ref14_xyz")
    if idx < 0:
        return {}

    block = section[idx:idx + 20000]

    end = len(block)
    for marker in [
        "Evaluation-only:",
        "Method per camera:",
        "================ AP02",
        "AP02 FINAL",
        "3. AP02",
        "AP03 FINAL",
    ]:
        j = block.find(marker)
        if j >= 0 and j > 0:
            end = min(end, j)

    block = block[:end]
    rows = {}

    for line in block.splitlines():
        m = EST_XYZ_ROW_RE.search(line)
        if not m:
            continue

        cam = m.group(1)
        est = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        gt = GT_REF14_XYZ.get(cam)
        if gt is None:
            continue

        rows[cam] = {
            "camera": cam,
            "t_cm": round(dist_cm(est, gt), 3),
            "r_deg": "",
            "note": "AP01 translation recomputed from archived est_ref14_xyz; rotation not archived in this ablation final file",
        }

    return rows


def parse_method_error_rows(block: str):
    rows = {}
    for line in block.splitlines():
        m = METHOD_ERROR_ROW_RE.search(line)
        if not m:
            continue

        cam = m.group(1)
        method = m.group(2).strip()

        # Avoid parsing markdown separator/header rows accidentally.
        if cam not in CAMERAS:
            continue
        if method.lower() in {"method", "---", "----"}:
            continue

        rows[cam] = {
            "camera": cam,
            "t_cm": float(m.group(3)),
            "r_deg": float(m.group(4)),
            "note": f"AP01 Ref14-origin static pose table; per-camera method={method}",
        }
    return rows

def parse_ap01_ref14_origin_static_table(text: str):
    section = extract_ap01_section(text)
    if not section:
        return {}

    idx = section.find("Static camera poses relative to Ref14:")
    if idx < 0:
        return {}

    block = section[idx:idx + 30000]

    end = len(block)
    for marker in [
        "Method-internal cam_edge_3-rooted details:",
        "================ AP02",
        "AP02 FINAL",
        "3. AP02",
        "AP03 FINAL",
    ]:
        j = block.find(marker)
        if j >= 0 and j > 0:
            end = min(end, j)

    rows = parse_method_error_rows(block[:end])
    return rows


def parse_ap01(text: str, variant: str):
    # 1) Current ablation AP01 format:
    # Static camera poses relative to Ref14:
    # camera | method | t_cm | r_deg | ...
    rows = parse_ap01_ref14_origin_static_table(text)
    if rows:
        return {
            "approach": "AP01",
            "method": "marker_direct_relay_ref14_origin_static_pose_table",
            "rows": rows,
        }

    # 2) Older/canonical AP01 format:
    # Evaluation-only: comparison against GT camera map in Ref14 frame
    rows = parse_ap01_true_eval_table(text)
    if rows:
        return {
            "approach": "AP01",
            "method": "marker_direct_relay_multichain_ref14_eval",
            "rows": rows,
        }

    # 3) Baseline import if needed.
    if variant in {"res_1280x720_baseline", "fov_69deg_baseline"} and CANONICAL_AP01.exists():
        rows = parse_ap01_true_eval_table(read_text(CANONICAL_AP01))
        if rows:
            return {
                "approach": "AP01",
                "method": "marker_direct_relay_multichain_ref14_eval_imported_baseline",
                "rows": rows,
            }

    # 4) Last fallback: translation only from estimated XYZ.
    rows = parse_ap01_est_map_translation_only(text)
    if rows:
        return {
            "approach": "AP01",
            "method": "marker_direct_relay_ref14_est_map_t_only",
            "rows": rows,
        }

    return None


def parse_ap02(text: str):
    block = extract_block(
        text,
        "AP02 FINAL RESULT — GT-Aligned Full-Map Evaluation",
        [
            "Markers:",
            "MARKER MAP",
            "AP03 FINAL",
            "================ AP03",
            "3. AP03",
        ],
    )
    rows = parse_pipe_error_rows(block)
    if rows:
        return {
            "approach": "AP02",
            "method": "ref_marker_graph_ba_gt_aligned_full_map",
            "rows": rows,
        }

    block = extract_block(
        text,
        "STATIC CAMERA EXTRINSICS VS GT",
        [
            "MARKER MAP VS GT",
            "Markers:",
            "AP03 FINAL",
            "================ AP03",
        ],
    )
    rows = parse_pipe_error_rows(block)
    if rows:
        return {
            "approach": "AP02",
            "method": "ref_marker_graph_ba_ref14_frame",
            "rows": rows,
        }

    return None


def parse_ap03_combined_error_rows(text: str, which: str):
    # Parse AP03 final combined table:
    # cam | single_err_cm | multi_err_cm | gain_cm | single_rot | multi_rot | ...
    idx = text.find("Evaluation-only: camera GT-vs-estimated errors:")
    if idx < 0:
        idx = text.find("single_err_cm")
    if idx < 0:
        return {}

    block = text[idx:idx + 30000]

    end = len(block)
    for marker in [
        "Evaluation-only: GT camera map in Ref14 frame:",
        "Final interpretation:",
        "AP02 FINAL",
        "AP01 FINAL",
        "================",
    ]:
        j = block.find(marker)
        if j >= 0 and j > 0:
            end = min(end, j)

    block = block[:end]
    rows = {}

    for line in block.splitlines():
        m = AP03_COMBINED_ROW_RE.search(line)
        if not m:
            continue

        cam = m.group(1)
        single_err = float(m.group(2))
        multi_err = float(m.group(3))
        single_rot = float(m.group(5))
        multi_rot = float(m.group(6))

        if which == "single":
            rows[cam] = {
                "camera": cam,
                "t_cm": single_err,
                "r_deg": single_rot,
                "note": "AP03 combined final table: Single Ref14 columns",
            }
        elif which == "multi":
            rows[cam] = {
                "camera": cam,
                "t_cm": multi_err,
                "r_deg": multi_rot,
                "note": "AP03 combined final table: Multi-ArUco columns",
            }
        else:
            raise ValueError(which)

    return rows


def parse_ap03_multi(text: str):
    # Preferred: final AP03 combined table with explicit multi_err_cm/multi_rot columns.
    rows = parse_ap03_combined_error_rows(text, "multi")
    if rows:
        return {
            "approach": "AP03_MULTI",
            "method": "targetless_colmap_multi_aruco_sim3",
            "rows": rows,
        }

    # Fallback for older reports with a dedicated Multi-ArUco camera table.
    lower = text.lower()
    markers = [
        "ap03 final result — targetless colmap + multi-aruco scale registration",
        "ap03 final result - targetless colmap + multi-aruco scale registration",
        "targetless colmap + multi-aruco",
        "multi-aruco scale registration",
        "multi aruco scale registration",
        "ap03_multi",
        "multi_aruco",
        "multi-aruco",
    ]

    start = -1
    for marker in markers:
        idx = lower.find(marker.lower())
        if idx >= 0:
            start = idx
            break

    if start < 0:
        return None

    block = text[start:start + 120000]

    # Important: do not parse the combined table with the generic 2-column parser.
    if "single_err_cm" in block and "multi_err_cm" in block:
        return None

    for marker in [
        "Static camera results relative to Ref14:",
        "Static camera GT evaluation",
        "Static camera results",
        "camera     |",
        "camera |",
    ]:
        idx = block.find(marker)
        if idx >= 0:
            block = block[idx:]
            break

    rows = parse_pipe_error_rows(block)
    if rows:
        return {
            "approach": "AP03_MULTI",
            "method": "targetless_colmap_multi_aruco_sim3",
            "rows": rows,
        }

    return None


def parse_ap03_single(text: str):
    # Preferred: final AP03 combined table with explicit single_err_cm/single_rot columns.
    rows = parse_ap03_combined_error_rows(text, "single")
    if rows:
        return {
            "approach": "AP03_SINGLE_REF14",
            "method": "targetless_colmap_single_ref14_sim3",
            "rows": rows,
        }

    # Fallback for older AP03 Single-Ref14 reports.
    start_markers = [
        "AP03 FINAL REPO-LIKE SCALE REGISTRATION REPORT",
        "AP03 FINAL RESULT — Targetless COLMAP + Ref14 Scale Registration",
        "AP03 FINAL RESULT - Targetless COLMAP + Ref14 Scale Registration",
    ]

    start = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0:
            start = idx
            break

    if start < 0:
        return None

    block = text[start:start + 120000]

    end = len(block)
    for marker in [
        "AP03 FINAL RESULT — Targetless COLMAP + Multi-ArUco",
        "AP03 FINAL RESULT - Targetless COLMAP + Multi-ArUco",
        "[OK] AP03 full pipeline complete",
        "Traceback",
    ]:
        idx = block.find(marker)
        if idx >= 0 and idx > 0:
            end = min(end, idx)

    block = block[:end]

    idx = block.find("Static camera GT evaluation")
    if idx >= 0:
        block = block[idx:]

    rows = parse_pipe_error_rows(block)
    if rows:
        return {
            "approach": "AP03_SINGLE_REF14",
            "method": "targetless_colmap_single_ref14_sim3",
            "rows": rows,
        }

    return None


def status_from_rows(rows):
    n = len(rows)
    if n == 0:
        return "MISSING"

    has_blank_rot = any(rows[c].get("r_deg", "") == "" for c in rows)
    if has_blank_rot:
        return f"T_ONLY_{n}/4"

    if n == 4:
        return "OK"
    return f"PARTIAL_{n}/4"

def make_wide_row(variant, parsed):
    rows = parsed["rows"]

    t_vals = [rows[c]["t_cm"] for c in CAMERAS if c in rows]
    r_vals = [rows[c]["r_deg"] for c in CAMERAS if c in rows]

    out = {
        "variant": variant,
        "approach": parsed["approach"],
        "method": parsed["method"],
        "status": status_from_rows(rows),
        "mean_t_cm": mean(t_vals),
        "mean_r_deg": mean(r_vals),
        "registered_static": f"{len(rows)}/4",
    }

    for cam in CAMERAS:
        short = cam.replace("cam_edge_", "cam")
        if cam in rows:
            out[f"{short}_t_cm"] = rows[cam]["t_cm"]
            out[f"{short}_r_deg"] = rows[cam]["r_deg"]
        else:
            out[f"{short}_t_cm"] = ""
            out[f"{short}_r_deg"] = ""

    return out

def make_long_rows(variant, parsed):
    out = []
    for cam in CAMERAS:
        r = parsed["rows"].get(cam)
        if not r:
            out.append({
                "variant": variant,
                "approach": parsed["approach"],
                "method": parsed["method"],
                "status": status_from_rows(parsed["rows"]),
                "camera": cam,
                "t_cm": "",
                "r_deg": "",
                "note": "missing estimate",
            })
        else:
            out.append({
                "variant": variant,
                "approach": parsed["approach"],
                "method": parsed["method"],
                "status": status_from_rows(parsed["rows"]),
                "camera": cam,
                "t_cm": r["t_cm"],
                "r_deg": r["r_deg"],
                "note": r.get("note", ""),
            })
    return out

def write_csv(path, rows):
    if not rows:
        path.write_text("[NO ROWS]\n")
        return
    with path.open("w", newline="", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_txt_table(path, rows):
    with path.open("w", errors="replace") as f:
        f.write("GT EST WIDE CAMERA ERROR TABLE\n")
        f.write("==============================\n\n")
        f.write("Meaning: each row is one variant + approach. Columns cam0/cam1/cam3/cam5 are translation/rotation errors vs GT.\n")
        f.write("AP01 uses the true Ref14-frame GT-evaluation table if present. Otherwise AP01 translation can be recomputed from archived est_ref14_xyz and marked T_ONLY.\n")
        f.write("AP03_MULTI and AP03_SINGLE_REF14 are reported as separate rows when available.\n\n")

        if not rows:
            f.write("[NO ROWS]\n")
            return

        cols = list(rows[0].keys())
        table_rows = [{c: fmt(r.get(c, "")) for c in cols} for r in rows]
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in table_rows)) for c in cols}

        f.write(" | ".join(c.ljust(widths[c]) for c in cols) + "\n")
        f.write("-+-".join("-" * widths[c] for c in cols) + "\n")

        last_variant = None
        for r in table_rows:
            if last_variant is not None and r.get("variant") != last_variant:
                f.write("\n")
            f.write(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + "\n")
            last_variant = r.get("variant")

def main():
    if not FINAL.exists():
        raise FileNotFoundError(FINAL)

    reports = sorted(
        [p for p in FINAL.glob("*_FINAL_RESULT.txt") if not p.name.startswith("ALL_")],
        key=lambda p: (
            VARIANT_ORDER.get(variant_from_report(p), 999),
            variant_from_report(p),
        ),
    )

    wide_rows = []
    long_rows = []

    for report in reports:
        variant = variant_from_report(report)
        text = read_text(report)

        parsed_items = []

        ap01 = parse_ap01(text, variant)
        parsed_items.append(ap01 if ap01 else {
            "approach": "AP01",
            "method": "marker_direct_relay_multichain_ref14_eval",
            "rows": {},
        })

        ap02 = parse_ap02(text)
        parsed_items.append(ap02 if ap02 else {
            "approach": "AP02",
            "method": "ref_marker_graph_ba_gt_aligned_full_map",
            "rows": {},
        })

        ap03m = parse_ap03_multi(text)
        parsed_items.append(ap03m if ap03m else {
            "approach": "AP03_MULTI",
            "method": "targetless_colmap_multi_aruco_sim3",
            "rows": {},
        })

        ap03s = parse_ap03_single(text)
        parsed_items.append(ap03s if ap03s else {
            "approach": "AP03_SINGLE_REF14",
            "method": "targetless_colmap_single_ref14_sim3",
            "rows": {},
        })

        for parsed in parsed_items:
            wide_rows.append(make_wide_row(variant, parsed))
            long_rows.extend(make_long_rows(variant, parsed))

    wide_rows.sort(key=lambda r: (
        VARIANT_ORDER.get(r["variant"], 999),
        r["variant"],
        APPROACH_ORDER.get(r["approach"], 999),
    ))

    write_csv(OUT_CSV, wide_rows)
    write_csv(OUT_LONG_CSV, long_rows)
    write_txt_table(OUT_TXT, wide_rows)

    print("[OK] wrote:")
    print(" ", OUT_TXT)
    print(" ", OUT_CSV)
    print(" ", OUT_LONG_CSV)

if __name__ == "__main__":
    main()
