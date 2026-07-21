#!/usr/bin/env python3
from pathlib import Path
import os
import re
import csv
import math

STUDY_ROOT = Path(os.environ.get("STUDY_ROOT", "results/bus_real_data/ablation/moving_cam/res"))
COMPACT = STUDY_ROOT / "99_summary" / "compact_final_outputs"
OUTDIR = STUDY_ROOT / "final_results" / "clean_tables"

OUT_TXT = OUTDIR / "RES_VARIANT_CAMERA_GT_COMPARISON.txt"
OUT_CSV = OUTDIR / "RES_VARIANT_CAMERA_GT_COMPARISON.csv"
OUT_LONG = OUTDIR / "RES_VARIANT_CAMERA_GT_COMPARISON_LONG.csv"
OUT_SOURCES = OUTDIR / "RES_VARIANT_CAMERA_GT_COMPARISON_SOURCES.txt"

CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

VARIANT_ORDER = {
    "res_320x180_extreme": 0,
    "res_640x360": 1,
    "res_960x540": 2,
    "res_1280x720_baseline": 3,
    "res_1920x1080": 4,
}

APPROACH_ORDER = {
    "AP01": 0,
    "AP02": 1,
    "AP03_MULTI": 2,
    "AP03_SINGLE_REF14": 3,
}

FLOAT = r"([0-9.+\-eE]+)"

# Generic rows:
# cam_edge_0 | 3.618 | 1.161 | ...
GENERIC_CAM_ERROR_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    + FLOAT + r"\s*\|\s*"
    + FLOAT + r"\s*\|",
    re.IGNORECASE,
)

# AP01 current result:
# cam_edge_0 | method | t_cm | r_deg | ...
METHOD_CAM_ERROR_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    r"([^|]+?)\s*\|\s*"
    + FLOAT + r"\s*\|\s*"
    + FLOAT + r"\s*\|",
    re.IGNORECASE,
)

# AP03 combined final:
# cam | single_err_cm | multi_err_cm | gain_cm | single_rot | multi_rot | ...
AP03_COMBINED_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    + FLOAT + r"\s*\|\s*"   # single_err
    + FLOAT + r"\s*\|\s*"   # multi_err
    + FLOAT + r"\s*\|\s*"   # gain
    + FLOAT + r"\s*\|\s*"   # single_rot
    + FLOAT + r"\s*\|",     # multi_rot
    re.IGNORECASE,
)

def read(p: Path) -> str:
    return p.read_text(errors="replace")

def fmt(v):
    if v == "" or v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)

def mean(vals):
    clean = []
    for v in vals:
        if v == "" or v is None:
            continue
        clean.append(float(v))
    if not clean:
        return ""
    return round(sum(clean) / len(clean), 3)

def find_one(root: Path, patterns):
    hits = []
    for pat in patterns:
        hits.extend(root.glob(pat))
    hits = sorted(set(hits))
    return hits[0] if hits else None

def variant_dirs():
    if not COMPACT.exists():
        raise FileNotFoundError(COMPACT)
    dirs = [p for p in COMPACT.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: (VARIANT_ORDER.get(p.name, 999), p.name))

def parse_generic_rows(block: str):
    rows = {}
    for line in block.splitlines():
        m = GENERIC_CAM_ERROR_RE.search(line)
        if not m:
            continue
        cam = m.group(1)
        if cam not in CAMERAS:
            continue
        rows[cam] = {
            "camera": cam,
            "t_cm": float(m.group(2)),
            "r_deg": float(m.group(3)),
        }
    return rows

def parse_method_rows(block: str):
    rows = {}
    for line in block.splitlines():
        m = METHOD_CAM_ERROR_RE.search(line)
        if not m:
            continue
        cam = m.group(1)
        if cam not in CAMERAS:
            continue
        rows[cam] = {
            "camera": cam,
            "t_cm": float(m.group(3)),
            "r_deg": float(m.group(4)),
        }
    return rows

def parse_ap01(path: Path):
    if not path:
        return {}

    text = read(path)

    # Preferred old/canonical AP01 table:
    idx = text.find("Evaluation-only: comparison against GT camera map in Ref14 frame")
    if idx >= 0:
        block = text[idx:idx + 20000]
        rows = parse_generic_rows(block)
        if rows:
            return rows

    # Current ablation AP01 table:
    idx = text.find("Static camera poses relative to Ref14:")
    if idx >= 0:
        block = text[idx:idx + 20000]
        end = block.find("Method-internal cam_edge_3-rooted details:")
        if end > 0:
            block = block[:end]
        rows = parse_method_rows(block)
        if rows:
            return rows

    return {}

def parse_ap02(path: Path):
    if not path:
        return {}

    text = read(path)
    idx = text.find("Evaluation-only: GT-aligned static camera comparison")
    if idx < 0:
        idx = text.find("STATIC CAMERA EXTRINSICS VS GT")
    if idx < 0:
        idx = 0

    block = text[idx:idx + 30000]
    return parse_generic_rows(block)

def parse_ap03_combined(path: Path, which: str):
    if not path:
        return {}

    text = read(path)
    idx = text.find("Evaluation-only: camera GT-vs-estimated errors:")
    if idx < 0:
        idx = text.find("single_err_cm")
    if idx < 0:
        return {}

    block = text[idx:idx + 30000]
    rows = {}

    for line in block.splitlines():
        m = AP03_COMBINED_RE.search(line)
        if not m:
            continue

        cam = m.group(1)
        if cam not in CAMERAS:
            continue

        single_err = float(m.group(2))
        multi_err = float(m.group(3))
        single_rot = float(m.group(5))
        multi_rot = float(m.group(6))

        if which == "single":
            rows[cam] = {"camera": cam, "t_cm": single_err, "r_deg": single_rot}
        elif which == "multi":
            rows[cam] = {"camera": cam, "t_cm": multi_err, "r_deg": multi_rot}
        else:
            raise ValueError(which)

    return rows

def parse_ap03_dedicated(path: Path):
    if not path:
        return {}

    text = read(path)

    markers = [
        "Static camera GT evaluation",
        "Static camera results relative to Ref14:",
        "Static camera results",
        "Evaluation-only: camera GT-vs-estimated errors:",
    ]

    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            rows = parse_generic_rows(text[idx:idx + 30000])
            if rows:
                return rows

    return parse_generic_rows(text)

def status(rows):
    n = len(rows)
    if n == 4:
        return "OK"
    if n == 0:
        return "MISSING"
    return f"PARTIAL_{n}/4"

def make_row(variant, approach, method, rows, source):
    out = {
        "variant": variant,
        "approach": approach,
        "method": method,
        "status": status(rows),
        "mean_t_cm": mean([rows[c]["t_cm"] for c in CAMERAS if c in rows]),
        "mean_r_deg": mean([rows[c]["r_deg"] for c in CAMERAS if c in rows]),
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

    out["_source"] = str(source) if source else ""
    return out

def make_long(wide_rows):
    out = []
    for r in wide_rows:
        for cam in CAMERAS:
            short = cam.replace("cam_edge_", "cam")
            out.append({
                "variant": r["variant"],
                "approach": r["approach"],
                "method": r["method"],
                "status": r["status"],
                "camera": cam,
                "t_cm": r[f"{short}_t_cm"],
                "r_deg": r[f"{short}_r_deg"],
                "source": r["_source"],
            })
    return out

def write_csv(path, rows):
    if not rows:
        path.write_text("[NO ROWS]\n")
        return

    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
    with path.open("w", newline="", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def write_long_csv(path, rows):
    if not rows:
        path.write_text("[NO ROWS]\n")
        return
    with path.open("w", newline="", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def write_txt(path, rows):
    with path.open("w", errors="replace") as f:
        f.write("RES VARIANT CAMERA GT COMPARISON\n")
        f.write("================================\n\n")
        f.write("Meaning: one row = one resolution variant + one method.\n")
        f.write("Every camera entry is GT-vs-estimated camera-pose error in the same evaluation frame.\n")
        f.write("No cam3-rooted AP01 fallback rows are used.\n")
        f.write("Source: 99_summary/compact_final_outputs/<variant>/<approach>/...\n\n")

        cols = [k for k in rows[0].keys() if not k.startswith("_")]
        table_rows = [{c: fmt(r.get(c, "")) for c in cols} for r in rows]
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in table_rows)) for c in cols}

        f.write(" | ".join(c.ljust(widths[c]) for c in cols) + "\n")
        f.write("-+-".join("-" * widths[c] for c in cols) + "\n")

        last_variant = None
        for r in table_rows:
            if last_variant is not None and r["variant"] != last_variant:
                f.write("\n")
            f.write(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + "\n")
            last_variant = r["variant"]

def write_sources(path, rows):
    with path.open("w", errors="replace") as f:
        f.write("SOURCE FILES USED\n")
        f.write("=================\n\n")
        for r in rows:
            f.write(f"{r['variant']} | {r['approach']} | {r['_source']}\n")

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    wide = []

    for vdir in variant_dirs():
        variant = vdir.name

        ap01_path = find_one(vdir / "AP01", ["*AP01_FINAL_RESULT.txt"])
        ap02_path = find_one(vdir / "AP02", ["*AP02_FINAL_RESULT.txt"])

        ap03_multi_path = find_one(vdir / "AP03", ["*AP03_FINAL_MULTI_ARUCO_RESULT.txt"])
        ap03_single_path = find_one(vdir / "AP03", ["*AP03_FINAL_SINGLE_REF14_RESULT.txt"])
        ap03_combined_path = find_one(vdir / "AP03", ["*AP03_FINAL_RESULT.txt"])

        ap01_rows = parse_ap01(ap01_path)
        wide.append(make_row(
            variant,
            "AP01",
            "marker_direct_relay_ref14_origin_gt_eval",
            ap01_rows,
            ap01_path,
        ))

        ap02_rows = parse_ap02(ap02_path)
        wide.append(make_row(
            variant,
            "AP02",
            "ref_marker_graph_ba_gt_aligned_full_map",
            ap02_rows,
            ap02_path,
        ))

        ap03m_rows = parse_ap03_dedicated(ap03_multi_path)
        if not ap03m_rows:
            ap03m_rows = parse_ap03_combined(ap03_combined_path, "multi")
            ap03m_source = ap03_combined_path
        else:
            ap03m_source = ap03_multi_path

        wide.append(make_row(
            variant,
            "AP03_MULTI",
            "targetless_colmap_multi_aruco_sim3",
            ap03m_rows,
            ap03m_source,
        ))

        ap03s_rows = parse_ap03_dedicated(ap03_single_path)
        if not ap03s_rows:
            ap03s_rows = parse_ap03_combined(ap03_combined_path, "single")
            ap03s_source = ap03_combined_path
        else:
            ap03s_source = ap03_single_path

        wide.append(make_row(
            variant,
            "AP03_SINGLE_REF14",
            "targetless_colmap_single_ref14_sim3",
            ap03s_rows,
            ap03s_source,
        ))

    wide.sort(key=lambda r: (
        VARIANT_ORDER.get(r["variant"], 999),
        r["variant"],
        APPROACH_ORDER.get(r["approach"], 999),
    ))

    write_txt(OUT_TXT, wide)
    write_csv(OUT_CSV, wide)
    write_long_csv(OUT_LONG, make_long(wide))
    write_sources(OUT_SOURCES, wide)

    print("[OK] wrote:")
    print(" ", OUT_TXT)
    print(" ", OUT_CSV)
    print(" ", OUT_LONG)
    print(" ", OUT_SOURCES)

if __name__ == "__main__":
    main()
