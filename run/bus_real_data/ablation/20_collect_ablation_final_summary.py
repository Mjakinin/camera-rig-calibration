#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("usage: python3 20_collect_ablation_final_summary.py <ablation_root> <parameter_name>")
    sys.exit(1)

ROOT = Path(sys.argv[1])
PARAM = sys.argv[2]
OUT = ROOT / "ABLATION_SUMMARY"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["AP01", "AP02", "AP03"]

def parse_status(p: Path):
    d = {}
    if not p.exists():
        return d
    for line in p.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d

def parse_comparison_txt(p: Path):
    d = {m: {} for m in METHODS}
    if not p.exists():
        return d

    for line in p.read_text(errors="replace").splitlines():
        s = line.strip()
        method = next((m for m in METHODS if s.startswith(m)), None)
        if not method:
            continue

        parts = [x.strip() for x in s.split("|")]
        if len(parts) >= 2:
            d[method]["table_status"] = parts[1]

        vals = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*cm\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*deg", s)
        if vals:
            d[method]["mean_t_cm"] = vals[0][0]
            d[method]["mean_r_deg"] = vals[0][1]
            if len(vals) >= 2:
                d[method]["worst_t_cm"] = vals[-1][0]
                d[method]["worst_r_deg"] = vals[-1][1]

        pair = re.search(r"worst\s+([A-Za-z0-9_\-]+)", s)
        if pair:
            d[method]["worst_pair"] = pair.group(1)

    return d

def as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    rows = []

    for var_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "ABLATION_SUMMARY"):
        final = var_dir / "FINAL_RESULTS"
        status = parse_status(final / "RUN_STATUS.txt")
        comp = parse_comparison_txt(final / "BASELINE_FINAL_CLEAN_COMPARISON.txt")

        row = {
            "variant": var_dir.name,
            "parameter": PARAM,
            "final_results_exists": str(final.exists()),
            "AP01_STATUS": status.get("AP01_STATUS", ""),
            "AP02_STATUS": status.get("AP02_STATUS", ""),
            "AP03_STATUS": status.get("AP03_STATUS", ""),
            "PAIRWISE_STATUS": status.get("PAIRWISE_STATUS", ""),
            "SECONDARY_STATUS": status.get("SECONDARY_STATUS", ""),
        }

        for m in METHODS:
            row[f"{m}_mean_t_cm"] = comp[m].get("mean_t_cm", "")
            row[f"{m}_mean_r_deg"] = comp[m].get("mean_r_deg", "")
            row[f"{m}_worst_pair"] = comp[m].get("worst_pair", "")
            row[f"{m}_worst_t_cm"] = comp[m].get("worst_t_cm", "")
            row[f"{m}_worst_r_deg"] = comp[m].get("worst_r_deg", "")

        rows.append(row)

    baseline = next((r for r in rows if "baseline" in r["variant"]), None)
    if baseline:
        for row in rows:
            for m in METHODS:
                bt = as_float(baseline.get(f"{m}_mean_t_cm"))
                br = as_float(baseline.get(f"{m}_mean_r_deg"))
                rt = as_float(row.get(f"{m}_mean_t_cm"))
                rr = as_float(row.get(f"{m}_mean_r_deg"))
                row[f"{m}_delta_t_cm_vs_baseline"] = "" if bt is None or rt is None else f"{rt - bt:.6f}"
                row[f"{m}_delta_r_deg_vs_baseline"] = "" if br is None or rr is None else f"{rr - br:.6f}"

    if not rows:
        raise SystemExit(f"[ERROR] no variants found under {ROOT}")

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    csv_path = OUT / "ABLATION_PARAMETER_EFFECT_SUMMARY.csv"
    txt_path = OUT / "ABLATION_PARAMETER_EFFECT_SUMMARY.txt"

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append(f"ABLATION PARAMETER EFFECT SUMMARY — {PARAM}")
    lines.append("=" * 80)
    lines.append(f"root: {ROOT}")
    lines.append("")

    for row in rows:
        lines.append(f"VARIANT: {row['variant']}")
        lines.append("-" * 80)
        lines.append(f"status: AP01={row['AP01_STATUS']} AP02={row['AP02_STATUS']} AP03={row['AP03_STATUS']} PAIRWISE={row['PAIRWISE_STATUS']} SECONDARY={row['SECONDARY_STATUS']}")
        for m in METHODS:
            lines.append(
                f"{m}: mean {row.get(f'{m}_mean_t_cm','')} cm / {row.get(f'{m}_mean_r_deg','')} deg"
                f" | worst {row.get(f'{m}_worst_pair','')} {row.get(f'{m}_worst_t_cm','')} cm / {row.get(f'{m}_worst_r_deg','')} deg"
                f" | delta_vs_baseline {row.get(f'{m}_delta_t_cm_vs_baseline','')} cm / {row.get(f'{m}_delta_r_deg_vs_baseline','')} deg"
            )
        lines.append("")

    txt_path.write_text("\n".join(lines) + "\n")

    print("[OK] wrote:")
    print(csv_path)
    print(txt_path)

if __name__ == "__main__":
    main()
