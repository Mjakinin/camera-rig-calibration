#!/usr/bin/env python3
from pathlib import Path
import csv
import re
import shutil

BASE = Path("results/bus_real_data/ablation/moving_cam/res")
OUT = BASE / "99_summary"
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

TEXT_SUFFIXES = {".txt", ".md", ".log", ".csv"}

KEY_NAME_RE = re.compile(
    r"(FINAL|SUMMARY|REPORT|COMPARISON|ap03|AP03|evaluation|eval)",
    re.IGNORECASE,
)

ERROR_RE = re.compile(
    r"(Traceback|RuntimeError|\[ERROR\]|\[FAIL\]|Missing static cameras|Missing AP03 static camera|No good initial image pair)",
    re.IGNORECASE,
)

METRIC_PATTERNS = {
    "mean_t_cm": [
        r"mean_translation_error_cm[:\s]+([0-9.+\-eE]+)",
        r"mean translation error[:\s]+([0-9.+\-eE]+)\s*cm",
    ],
    "mean_r_deg": [
        r"mean_rotation_error_deg[:\s]+([0-9.+\-eE]+)",
        r"mean rotation error[:\s]+([0-9.+\-eE]+)\s*deg",
    ],
    "registered_static": [
        r"registered static cameras[:\s]+([0-9]+)\s*/\s*([0-9]+)",
    ],
    "registered_moving": [
        r"registered moving frames[:\s]+([0-9]+)",
    ],
    "ref14_obs": [
        r"Ref14 corner observations[:\s]+([0-9]+)",
        r"ref14_corner_observations[:\s]+([0-9]+)",
    ],
    "scale": [
        r"scale_colmap_to_metric[:\s]+([0-9.+\-eE]+)",
        r"estimated COLMAP-to-meter scale[:\s]+([0-9.+\-eE]+)",
    ],
    "corner_fit_cm": [
        r"corner_fit_mean_error_cm[:\s]+([0-9.+\-eE]+)",
        r"corner fit mean error[:\s]+([0-9.+\-eE]+)\s*cm",
    ],
}

def read_text_safe(p: Path, max_chars=2_000_000) -> str:
    try:
        s = p.read_text(errors="ignore")
        return s[:max_chars]
    except Exception:
        return ""

def find_candidate_files(root: Path):
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in TEXT_SUFFIXES and KEY_NAME_RE.search(p.name):
            out.append(p)
    return sorted(out)

def scan_errors(root: Path):
    hits = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in TEXT_SUFFIXES:
            text = read_text_safe(p, 500_000)
            for m in ERROR_RE.finditer(text):
                hits.append((p, m.group(0)))
                break
    return hits

def extract_metrics(files):
    blob = "\n".join(read_text_safe(p, 500_000) for p in files)
    metrics = {}
    for key, patterns in METRIC_PATTERNS.items():
        metrics[key] = ""
        for pat in patterns:
            m = re.search(pat, blob, re.IGNORECASE)
            if m:
                if key == "registered_static":
                    metrics[key] = f"{m.group(1)}/{m.group(2)}"
                else:
                    metrics[key] = m.group(1)
                break
    return metrics

def status_for(approach, root, errors, metrics):
    if not root.exists():
        return "MISSING"
    if approach == "AP03" and (
        errors or
        metrics.get("registered_static", "").startswith("0/") or
        "3/4" in metrics.get("registered_static", "")
    ):
        return "PARTIAL"
    if errors:
        return "WARN"
    return "OK"

def copy_compact_files(variant, approach, files):
    export_root = OUT / "compact_final_outputs" / variant / approach
    export_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for p in files:
        # avoid copying huge logs unless they look important
        if p.suffix == ".log" and "pipeline" not in p.name.lower():
            continue
        rel_name = "__".join(p.relative_to(BASE).parts)
        dst = export_root / rel_name
        try:
            shutil.copy2(p, dst)
            copied.append(dst)
        except Exception:
            pass
    return copied

rows = []

for variant in VARIANTS:
    for approach, approach_root in APPROACHES.items():
        root = approach_root / variant
        files = find_candidate_files(root)
        errors = scan_errors(root)
        metrics = extract_metrics(files)
        status = status_for(approach, root, errors, metrics)
        copied = copy_compact_files(variant, approach, files)

        rows.append({
            "variant": variant,
            "approach": approach,
            "status": status,
            "error_hits": len(errors),
            "mean_t_cm": metrics.get("mean_t_cm", ""),
            "mean_r_deg": metrics.get("mean_r_deg", ""),
            "registered_static": metrics.get("registered_static", ""),
            "registered_moving": metrics.get("registered_moving", ""),
            "ref14_obs": metrics.get("ref14_obs", ""),
            "scale": metrics.get("scale", ""),
            "corner_fit_cm": metrics.get("corner_fit_cm", ""),
            "candidate_files": len(files),
            "compact_files_copied": len(copied),
        })

csv_path = OUT / "res_pipeline_status_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

md_path = OUT / "res_pipeline_status_summary.md"
with md_path.open("w") as f:
    f.write("# Moving Camera Resolution — Pipeline Status Summary\n\n")
    f.write("This file summarizes AP01/AP02/AP03 outputs for the moving-camera resolution ablation.\n\n")
    f.write("## Status interpretation\n\n")
    f.write("- `OK`: no obvious error markers found in final/report files.\n")
    f.write("- `WARN`: output exists but warning/error markers were found.\n")
    f.write("- `PARTIAL`: AP03 produced output, but COLMAP/registration was incomplete for at least one camera or post-report checks failed.\n")
    f.write("- `MISSING`: output folder missing.\n\n")

    f.write("| variant | approach | status | errors | mean_t_cm | mean_r_deg | registered_static | registered_moving | ref14_obs | scale | corner_fit_cm |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['variant']} | {r['approach']} | {r['status']} | {r['error_hits']} | "
            f"{r['mean_t_cm']} | {r['mean_r_deg']} | {r['registered_static']} | "
            f"{r['registered_moving']} | {r['ref14_obs']} | {r['scale']} | {r['corner_fit_cm']} |\n"
        )

    f.write("\n## Compact exports\n\n")
    f.write("Selected final/report/comparison files were copied to:\n\n")
    f.write("```text\n")
    f.write(str(OUT / "compact_final_outputs") + "\n")
    f.write("```\n\n")

print("[OK] wrote:")
print(" ", csv_path)
print(" ", md_path)
print(" ", OUT / "compact_final_outputs")
