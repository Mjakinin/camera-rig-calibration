#!/usr/bin/env python3
from pathlib import Path
import csv
import os
import re

STUDY_ROOT = Path(os.environ.get("STUDY_ROOT", "results/bus_real_data/ablation/moving_cam/res"))
FINAL = STUDY_ROOT / "final_results"

STATUS_CSV = FINAL / "res_pipeline_status_summary.csv"
if not STATUS_CSV.exists():
    # FOV later may use fov naming.
    candidates = list(FINAL.glob("*pipeline_status_summary.csv"))
    STATUS_CSV = candidates[0] if candidates else STATUS_CSV

OUT = FINAL / "COMPACT_CAMERA_GT_SUMMARY.txt"

VARIANT_REPORTS = sorted(FINAL.glob("*_FINAL_RESULT.txt"))

CAM_ROW_RE = re.compile(r"^\s*cam_edge_\d+\s*\|")
CAM_HEADER_RE = re.compile(r"^\s*camera\s*\|", re.IGNORECASE)
MEAN_RE = re.compile(r"(mean translation error|mean_translation_error|mean rotation error|mean_rotation_error|registered static cameras|registered moving frames)", re.IGNORECASE)
IMPORTANT_RE = re.compile(r"(Static camera GT evaluation|camera GT|FINAL METHOD|METHOD COMPARISON|AP02|AP03|AP01)", re.IGNORECASE)

def read_status_rows():
    if not STATUS_CSV.exists():
        return []
    with STATUS_CSV.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def variant_from_file(p):
    name = p.name
    return name.replace("_FINAL_RESULT.txt", "")

def extract_compact_blocks(p):
    lines = p.read_text(errors="replace").splitlines()
    blocks = []
    current_approach = None
    recent_heading = ""

    for i, line in enumerate(lines):
        if "AP01 —" in line or "AP01 -" in line:
            current_approach = "AP01"
        elif "AP02 —" in line or "AP02 -" in line:
            current_approach = "AP02"
        elif "AP03 —" in line or "AP03 -" in line:
            current_approach = "AP03"

        if IMPORTANT_RE.search(line):
            recent_heading = line.strip()

        keep = False
        reason = ""

        if MEAN_RE.search(line):
            keep = True
            reason = "metric"
        elif CAM_HEADER_RE.search(line):
            keep = True
            reason = "camera_table_header"
        elif CAM_ROW_RE.search(line):
            keep = True
            reason = "camera_row"
        elif "registered static cameras" in line.lower():
            keep = True
            reason = "registered_static"
        elif "static registered:" in line.lower() or "static missing:" in line.lower():
            keep = True
            reason = "colmap_static"

        if keep:
            blocks.append({
                "approach": current_approach or "UNKNOWN",
                "line_no": i + 1,
                "heading": recent_heading,
                "reason": reason,
                "text": line.rstrip(),
            })

    # Deduplicate exact same approach+text while preserving order.
    seen = set()
    unique = []
    for b in blocks:
        key = (b["approach"], b["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(b)

    return unique

def main():
    status_rows = read_status_rows()

    with OUT.open("w", errors="replace") as f:
        f.write("COMPACT CAMERA GT SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Study root: {STUDY_ROOT}\n")
        f.write(f"Final folder: {FINAL}\n\n")

        f.write("1. PIPELINE STATUS / MEAN GT ERRORS\n")
        f.write("-" * 100 + "\n")

        if status_rows:
            cols = [
                "variant", "approach", "status", "error_hits",
                "mean_t_cm", "mean_r_deg",
                "registered_static", "registered_moving",
                "ref14_obs", "corner_fit_cm",
            ]
            widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in status_rows)) for c in cols}
            f.write(" | ".join(c.ljust(widths[c]) for c in cols) + "\n")
            f.write("-+-".join("-" * widths[c] for c in cols) + "\n")
            for r in status_rows:
                f.write(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + "\n")
        else:
            f.write("[WARN] No pipeline status CSV found.\n")

        f.write("\n\n2. CAMERA-WISE GT / REGISTRATION SNIPPETS\n")
        f.write("-" * 100 + "\n")
        f.write("This section extracts camera table rows and key GT/registration metrics from each per-variant final report.\n\n")

        for report in VARIANT_REPORTS:
            variant = variant_from_file(report)
            if variant == "ALL_RES_FINAL_RESULTS":
                continue

            f.write("\n")
            f.write("=" * 100 + "\n")
            f.write(f"VARIANT: {variant}\n")
            f.write("=" * 100 + "\n")

            blocks = extract_compact_blocks(report)
            if not blocks:
                f.write("[WARN] No compact camera/GT rows extracted.\n")
                continue

            last_approach = None
            last_heading = None

            for b in blocks:
                if b["approach"] != last_approach:
                    f.write(f"\n[{b['approach']}]\n")
                    last_approach = b["approach"]
                    last_heading = None

                if b["heading"] and b["heading"] != last_heading:
                    f.write(f"\n# {b['heading']}\n")
                    last_heading = b["heading"]

                f.write(f"L{b['line_no']}: {b['text']}\n")

        f.write("\n\n3. HOW TO READ THIS FILE\n")
        f.write("-" * 100 + "\n")
        f.write(
            "- mean_t_cm / mean_r_deg are the compact GT error indicators.\n"
            "- registered_static shows how many static cameras were successfully estimated.\n"
            "- AP03 PARTIAL means COLMAP/registration did not recover all static cameras or had a warning.\n"
            "- Camera rows show per-camera estimated-vs-GT information when present in the source reports.\n"
            "- The large per-variant FINAL_RESULT files remain the full archive; this file is the readable summary.\n"
        )

    print("[OK] wrote:", OUT)

if __name__ == "__main__":
    main()
