#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1])
variants = sorted([
    p for p in root.iterdir()
    if p.is_dir()
    and not p.name.startswith(("00_", "01_", "02_", "03_", "99_"))
    and p.name != "ABLATION_SUMMARY"
])

for v in variants:
    final = v / "FINAL_RESULTS"
    if not final.exists():
        continue

    pair_sum = final / "BASELINE_FINAL_PAIRWISE_SUMMARY.csv"
    pair_det = final / "BASELINE_FINAL_PAIRWISE_DETAIL.csv"
    sec_sum = final / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv"
    sec_det = final / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv"

    statuses = {"AP01": "FAILED", "AP02": "FAILED", "AP03": "FAILED"}

    if pair_sum.exists():
        with pair_sum.open(newline="") as f:
            for row in csv.DictReader(f):
                m = row.get("method", "")
                if m in statuses:
                    statuses[m] = row.get("status", "FAILED") or "FAILED"

    pairwise_status = "OK" if pair_sum.exists() and pair_det.exists() else "FAILED"
    secondary_status = "OK" if sec_sum.exists() and sec_det.exists() else "FAILED"

    txt = "\n".join([
        f"variant={v.name}",
        f"AP01_STATUS={statuses['AP01']}",
        f"AP02_STATUS={statuses['AP02']}",
        f"AP03_STATUS={statuses['AP03']}",
        f"PAIRWISE_STATUS={pairwise_status}",
        f"SECONDARY_STATUS={secondary_status}",
        "",
    ])
    (final / "RUN_STATUS.txt").write_text(txt)
    print("[OK]", v.name, statuses, "PAIRWISE", pairwise_status, "SECONDARY", secondary_status)
