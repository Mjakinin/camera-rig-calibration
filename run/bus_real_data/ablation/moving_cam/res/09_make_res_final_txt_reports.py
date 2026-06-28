#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import csv
import re

BASE = Path("results/bus_real_data/ablation/moving_cam/res")
OUT = BASE / "99_summary" / "final_txt_by_variant"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "res_320x180_extreme",
    "res_640x360",
    "res_960x540",
    "res_1280x720_baseline",
    "res_1920x1080",
]

APPROACHES = {
    "AP01": BASE / "02_ap01_results",
    "AP02": BASE / "03_ap02_results",
    "AP03": BASE / "04_ap03_results",
}

TEXT_EXTS = {".txt", ".md", ".csv", ".log"}

MAX_CHARS_PER_FILE = 180_000
MAX_CSV_LINES = 220
MAX_ERROR_HITS = 80

INCLUDE_FILE_RE = re.compile(
    r"(README|SUMMARY|REPORT|FINAL|COMPARISON|RESULT|EVAL|evaluation|"
    r"graph_connectivity|colmap_model_summary|registered_images|"
    r"ap02_final|ap03_colmap|method_comparison|camera|extrinsics)",
    re.IGNORECASE,
)

INCLUDE_DIR_RE = re.compile(
    r"(04_moving_camera_colmap_trajectory|05_direct|06_moving|07_final|"
    r"05_graph_initialization|07_graph_ba|08_final_results|"
    r"03_reconstruction_inspection|06_triangulated_ref_aruco_registration|"
    r"90_approach_comparison_ref_aruco)",
    re.IGNORECASE,
)

EXCLUDE_RE = re.compile(
    r"(debug_images|__pycache__|\.ap01_compact_cache|database\.db|"
    r"spawn_.*\.sh|reprojection_errors_by_observation|"
    r"points3D\.txt|images\.txt|cameras\.txt|\.png|\.jpg|\.jpeg|\.bin)$",
    re.IGNORECASE,
)

ERROR_RE = re.compile(
    r"(Traceback|RuntimeError|\[ERROR\]|\[FAIL\]|Missing static cameras|"
    r"Missing AP03 static camera|No good initial image pair)",
    re.IGNORECASE,
)

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(BASE))
    except Exception:
        return str(p)

def read_text(p: Path) -> str:
    try:
        s = p.read_text(errors="replace")
    except Exception as e:
        return f"[COULD NOT READ: {e}]"

    if p.suffix.lower() == ".csv":
        lines = s.splitlines()
        if len(lines) > MAX_CSV_LINES:
            return "\n".join(lines[:MAX_CSV_LINES]) + f"\n\n[TRUNCATED CSV: {len(lines) - MAX_CSV_LINES} more lines omitted]"
        return s

    if len(s) > MAX_CHARS_PER_FILE:
        return s[:MAX_CHARS_PER_FILE] + f"\n\n[TRUNCATED FILE: {len(s) - MAX_CHARS_PER_FILE} more chars omitted]"
    return s

def should_include(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() not in TEXT_EXTS:
        return False
    path_s = str(p)
    if EXCLUDE_RE.search(path_s):
        return False
    if INCLUDE_FILE_RE.search(p.name):
        return True
    if INCLUDE_DIR_RE.search(path_s):
        return True
    return False

def collect_files(root: Path):
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if should_include(p)]
    return sorted(set(files), key=lambda x: str(x))

def scan_errors(root: Path):
    hits = []
    if not root.exists():
        return hits

    for p in sorted(root.rglob("*"), key=lambda x: str(x)):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
            continue
        if EXCLUDE_RE.search(str(p)):
            continue

        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            if ERROR_RE.search(line):
                hits.append((p, i, line.strip()))
                if len(hits) >= MAX_ERROR_HITS:
                    return hits
    return hits

def load_status_rows():
    csv_path = BASE / "99_summary" / "res_pipeline_status_summary.csv"
    rows = {}
    if not csv_path.exists():
        return rows
    with csv_path.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[(r.get("variant", ""), r.get("approach", ""))] = r
    return rows

def read_shared_observation_summary(variant: str) -> str:
    p = BASE / "01_shared_observations" / variant / "SHARED_ARUCO_DETECTION_SUMMARY.txt"
    if p.exists():
        return read_text(p)

    # Fallback: global summary table
    p = BASE / "99_summary" / "moving_cam_res_shared_observations_summary.txt"
    if p.exists():
        return read_text(p)

    return "[MISSING] shared ArUco observation summary not found."

def raw_dataset_info(variant: str) -> str:
    raw = BASE / "00_prepared_datasets" / variant / "raw_images"
    cap = BASE / "00_captures" / variant / "images"
    info = []

    info.append(f"prepared_raw_images_exists: {raw.exists()}")
    info.append(f"capture_images_exists: {cap.exists()}")

    moving_count = len(list((raw / "moving").glob("frame_*.png"))) if raw.exists() else 0
    static_count = len(list((raw / "static").glob("*.png"))) if raw.exists() else 0
    cap_count = len(list(cap.glob("frame_*.png"))) if cap.exists() else 0

    info.append(f"prepared_moving_frames: {moving_count}")
    info.append(f"prepared_static_images: {static_count}")
    info.append(f"captured_moving_frames: {cap_count}")
    info.append(f"moving_camera_info_exists: {(raw / 'camera_info' / 'moving_calib_camera.json').exists()}")

    return "\n".join(info)

def write_section(f, title: str):
    f.write("\n\n")
    f.write("=" * 100 + "\n")
    f.write(title + "\n")
    f.write("=" * 100 + "\n\n")

def write_file_block(f, p: Path):
    f.write("\n")
    f.write("-" * 100 + "\n")
    f.write(f"FILE: {rel(p)}\n")
    f.write("-" * 100 + "\n")
    f.write(read_text(p).rstrip() + "\n")

def write_variant_report(variant: str, status_rows: dict) -> Path:
    out_path = OUT / f"{variant}_FINAL_RESULT.txt"

    with out_path.open("w", errors="replace") as f:
        f.write(f"MOVING CAMERA RESOLUTION FINAL RESULT — {variant}\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Experiment root: {BASE}\n")
        f.write(f"Variant: {variant}\n")

        write_section(f, "0. DATASET / RAW IMAGE STATUS")
        f.write(raw_dataset_info(variant) + "\n")

        write_section(f, "1. SHARED ARUCO OBSERVATIONS")
        f.write(read_shared_observation_summary(variant).rstrip() + "\n")

        write_section(f, "2. PIPELINE STATUS TABLE")
        for approach in ["AP01", "AP02", "AP03"]:
            row = status_rows.get((variant, approach), {})
            if not row:
                f.write(f"{approach}: status row missing\n")
                continue

            keys = [
                "status",
                "error_hits",
                "mean_t_cm",
                "mean_r_deg",
                "registered_static",
                "registered_moving",
                "ref14_obs",
                "scale",
                "corner_fit_cm",
                "candidate_files",
                "compact_files_copied",
            ]
            f.write(f"\n{approach}\n")
            f.write("-" * 40 + "\n")
            for k in keys:
                f.write(f"{k}: {row.get(k, '')}\n")

        for approach, root_base in APPROACHES.items():
            variant_root = root_base / variant

            write_section(f, f"3. {approach} — DIAGNOSTIC WARNINGS / ERRORS")
            hits = scan_errors(variant_root)
            if not hits:
                f.write("[OK] No obvious error markers found in included text/log files.\n")
            else:
                for p, line_no, line in hits:
                    f.write(f"{rel(p)}:{line_no}: {line}\n")

            write_section(f, f"4. {approach} — FINAL / REPORT / SUMMARY FILES")
            files = collect_files(variant_root)
            if not files:
                f.write(f"[MISSING] No final/report/summary text files found under: {rel(variant_root)}\n")
            else:
                f.write("Included files:\n")
                for p in files:
                    f.write(f"- {rel(p)}\n")
                for p in files:
                    write_file_block(f, p)

        write_section(f, "5. INTERPRETATION NOTES")
        f.write(
            "This TXT file is the human-facing consolidated result for one resolution variant.\n"
            "The original AP01/AP02/AP03 step folders are intentionally kept as reproducibility artifacts.\n"
            "Do not interpret debug overlay scripts containing literal '[ERROR]' echo statements as executed pipeline failures.\n"
            "For AP03, partial COLMAP registration, missing static cameras, high Ref14 reprojection error, or RuntimeError in a combined report should be interpreted as a method-specific failure/instability for that variant.\n"
        )

    return out_path

def main():
    status_rows = load_status_rows()

    written = []
    for variant in VARIANTS:
        written.append(write_variant_report(variant, status_rows))

    master = OUT / "ALL_RES_FINAL_RESULTS.txt"
    with master.open("w", errors="replace") as f:
        f.write("MOVING CAMERA RESOLUTION — ALL FINAL RESULTS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("Variant reports included:\n")
        for p in written:
            f.write(f"- {p}\n")

        for p in written:
            f.write("\n\n\n")
            f.write("#" * 120 + "\n")
            f.write(f"BEGIN {p.name}\n")
            f.write("#" * 120 + "\n\n")
            f.write(p.read_text(errors="replace"))

    print("[OK] wrote per-variant final TXT reports:")
    for p in written:
        print(" ", p)
    print("[OK] wrote master report:")
    print(" ", master)

if __name__ == "__main__":
    main()
