#!/usr/bin/env python3
from pathlib import Path
import csv
import re
import math

ROOT = Path("results/bus_real_data/ablation/moving_cam/res")
COMPACT = ROOT / "99_summary" / "compact_final_outputs"
OUTDIR = ROOT / "final_results" / "clean_tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_TXT = OUTDIR / "RES_CLEAN_ABLATION_CAMERA_GT_TABLE.txt"
OUT_CSV = OUTDIR / "RES_CLEAN_ABLATION_CAMERA_GT_TABLE.csv"
OUT_SOURCES = OUTDIR / "RES_CLEAN_ABLATION_SOURCES.txt"

VARIANTS = [
    "res_320x180_extreme",
    "res_640x360",
    "res_960x540",
    "res_1280x720_baseline",
    "res_1920x1080",
]

CAMS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

FLOAT = r"([0-9.+\-eE]+)"

GENERIC_ROW = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*" + FLOAT + r"\s*\|\s*" + FLOAT + r"\s*\|"
)

METHOD_ROW = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*([^|]+?)\s*\|\s*" + FLOAT + r"\s*\|\s*" + FLOAT + r"\s*\|"
)

AP03_COMBINED = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    + FLOAT + r"\s*\|\s*"   # single_err
    + FLOAT + r"\s*\|\s*"   # multi_err
    + FLOAT + r"\s*\|\s*"   # gain
    + FLOAT + r"\s*\|\s*"   # single_rot
    + FLOAT + r"\s*\|"      # multi_rot
)

def read(p):
    return p.read_text(errors="replace")

def find_one(base, patterns):
    if not base.exists():
        return None
    hits = []
    for pat in patterns:
        hits.extend(base.rglob(pat))
    hits = sorted(set(hits))
    return hits[0] if hits else None

def fmt(x):
    if x == "" or x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)

def mean(vals):
    vals = [float(v) for v in vals if v != "" and v is not None]
    return round(sum(vals) / len(vals), 3) if vals else ""

def parse_pipe_rows(text):
    rows = {}
    for line in text.splitlines():
        m = GENERIC_ROW.search(line)
        if not m:
            continue
        cam = m.group(1)
        if cam in CAMS:
            rows[cam] = {"t_cm": float(m.group(2)), "r_deg": float(m.group(3))}
    return rows

def parse_method_rows(text):
    rows = {}
    for line in text.splitlines():
        m = METHOD_ROW.search(line)
        if not m:
            continue
        cam = m.group(1)
        if cam in CAMS:
            rows[cam] = {"t_cm": float(m.group(3)), "r_deg": float(m.group(4))}
    return rows

def parse_ap01(vdir):
    # Preferred: AP01 Ref14-origin GT evaluation regenerated per resolution variant.
    generated = ROOT / "final_results" / "clean_tables" / "ap01_ref14_by_variant" / vdir.name / "AP01_FINAL_RESULT.csv"
    if generated.exists():
        rows = {}
        with generated.open(errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cam = r.get("entity_id") or r.get("camera") or r.get("entity")
                if cam not in CAMS:
                    continue
                rows[cam] = {
                    "t_cm": float(r["translation_error_cm"]),
                    "r_deg": float(r["rotation_error_deg"]),
                }
        if rows:
            return rows, generated, "VARIANT_SPECIFIC"

    p = find_one(vdir / "AP01", ["*AP01_FINAL_RESULT.txt"])
    if not p:
        return {}, None, "MISSING"

    text = read(p)

    idx = text.find("Evaluation-only: comparison against GT camera map in Ref14 frame")
    if idx >= 0:
        rows = parse_pipe_rows(text[idx:idx + 20000])
        if rows:
            return rows, p, "REUSED_GLOBAL_OR_CONSTANT"

    idx = text.find("Static camera poses relative to Ref14:")
    if idx >= 0:
        block = text[idx:idx + 20000]
        cut = block.find("Method-internal cam_edge_3-rooted details:")
        if cut > 0:
            block = block[:cut]
        rows = parse_method_rows(block)
        if rows:
            return rows, p, "REUSED_GLOBAL_OR_CONSTANT"

    return {}, p, "MISSING"

def parse_ap02(vdir):
    # Preferred: true variant-specific AP02 camera CSV.
    p = find_one(vdir / "AP02", ["*ap02_static_cameras_ref_aruco_vs_gt.csv"])
    if p:
        rows = {}
        with p.open(errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cam = r.get("entity") or r.get("camera") or r.get("entity_id")
                if cam not in CAMS:
                    continue
                t = r.get("t_cm") or r.get("translation_error_cm") or r.get("translation_error")
                rr = r.get("r_deg") or r.get("rotation_error_deg") or r.get("rotation_error")
                if t is not None and rr is not None:
                    rows[cam] = {"t_cm": float(t), "r_deg": float(rr)}
        if rows:
            return rows, p, "VARIANT_SPECIFIC"

    # Fallback: readable report table.
    p = find_one(vdir / "AP02", ["*ap02_ref_aruco_eval_report.txt", "*AP02_FINAL_READABLE_REF_ARUCO_REPORT.txt"])
    if p:
        rows = parse_pipe_rows(read(p))
        if rows:
            return rows, p, "VARIANT_SPECIFIC"

    return {}, None, "MISSING"

def parse_ap03_combined(path, which):
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
        m = AP03_COMBINED.search(line)
        if not m:
            continue
        cam = m.group(1)
        if cam not in CAMS:
            continue

        single_err = float(m.group(2))
        multi_err = float(m.group(3))
        single_rot = float(m.group(5))
        multi_rot = float(m.group(6))

        if which == "single":
            rows[cam] = {"t_cm": single_err, "r_deg": single_rot}
        else:
            rows[cam] = {"t_cm": multi_err, "r_deg": multi_rot}

    return rows

def parse_ap03_dedicated(vdir, which):
    if which == "multi":
        p = find_one(vdir / "AP03", ["*AP03_FINAL_MULTI_ARUCO_RESULT.txt"])
    else:
        p = find_one(vdir / "AP03", ["*AP03_FINAL_SINGLE_REF14_RESULT.txt"])

    if p:
        text = read(p)
        rows = parse_pipe_rows(text)
        if rows:
            return rows, p, "VARIANT_SPECIFIC"

    p = find_one(vdir / "AP03", ["*AP03_FINAL_RESULT.txt"])
    rows = parse_ap03_combined(p, which)
    if rows:
        return rows, p, "VARIANT_SPECIFIC"

    return {}, p, "MISSING"

def status(rows):
    if len(rows) == 4:
        return "OK"
    if len(rows) == 0:
        return "MISSING"
    return f"PARTIAL_{len(rows)}/4"

def make_row(variant, approach, method, rows, src, quality):
    out = {
        "variant": variant,
        "approach": approach,
        "method": method,
        "source_quality": quality,
        "status": status(rows),
        "mean_t_cm": mean([rows[c]["t_cm"] for c in CAMS if c in rows]),
        "mean_r_deg": mean([rows[c]["r_deg"] for c in CAMS if c in rows]),
        "registered_static": f"{len(rows)}/4",
    }

    for cam in CAMS:
        short = cam.replace("cam_edge_", "cam")
        if cam in rows:
            out[f"{short}_t_cm"] = rows[cam]["t_cm"]
            out[f"{short}_r_deg"] = rows[cam]["r_deg"]
        else:
            out[f"{short}_t_cm"] = ""
            out[f"{short}_r_deg"] = ""

    out["_source"] = str(src) if src else ""
    return out

def write_table(rows):
    cols = [k for k in rows[0].keys() if not k.startswith("_")]
    text_rows = [{c: fmt(r.get(c, "")) for c in cols} for r in rows]
    widths = {c: max(len(c), *(len(str(r[c])) for r in text_rows)) for c in cols}

    with OUT_TXT.open("w", errors="replace") as f:
        f.write("RES CLEAN ABLATION CAMERA GT TABLE\n")
        f.write("==================================\n\n")
        f.write("Meaning: GT-vs-estimated static camera errors per resolution and approach.\n")
        f.write("Only source_quality=VARIANT_SPECIFIC rows should be interpreted as true resolution-ablation effects.\n")
        f.write("AP01 is marked REUSED_GLOBAL_OR_CONSTANT if the archived AP01 source is identical/global.\n")
        f.write("No cam3-rooted fallback values are used.\n\n")

        f.write(" | ".join(c.ljust(widths[c]) for c in cols) + "\n")
        f.write("-+-".join("-" * widths[c] for c in cols) + "\n")

        last = None
        for r in text_rows:
            if last is not None and r["variant"] != last:
                f.write("\n")
            f.write(" | ".join(str(r[c]).ljust(widths[c]) for c in cols) + "\n")
            last = r["variant"]

    with OUT_CSV.open("w", newline="", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    with OUT_SOURCES.open("w", errors="replace") as f:
        f.write("SOURCES USED\n")
        f.write("============\n\n")
        for r in rows:
            f.write(f"{r['variant']} | {r['approach']} | {r['source_quality']} | {r['_source']}\n")

def main():
    rows = []

    for variant in VARIANTS:
        vdir = COMPACT / variant

        ap01, src, q = parse_ap01(vdir)
        rows.append(make_row(variant, "AP01", "marker_direct_relay_ref14_origin_gt_eval", ap01, src, q))

        ap02, src, q = parse_ap02(vdir)
        rows.append(make_row(variant, "AP02", "ref_marker_graph_ba_gt_aligned_full_map", ap02, src, q))

        ap03m, src, q = parse_ap03_dedicated(vdir, "multi")
        rows.append(make_row(variant, "AP03_MULTI", "targetless_colmap_multi_aruco_sim3", ap03m, src, q))

        ap03s, src, q = parse_ap03_dedicated(vdir, "single")
        rows.append(make_row(variant, "AP03_SINGLE_REF14", "targetless_colmap_single_ref14_sim3", ap03s, src, q))

    write_table(rows)

    print("[OK] wrote:")
    print(" ", OUT_TXT)
    print(" ", OUT_CSV)
    print(" ", OUT_SOURCES)

if __name__ == "__main__":
    main()
