#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

CAMS = ["cam0", "cam1", "cam3", "cam5"]

def fmt(x):
    if x is None or str(x).strip() == "":
        return "-"
    try:
        return f"{float(x):.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def md_table(rows, cols):
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    lines = []
    lines.append(" | ".join(c.ljust(widths[c]) for c in cols))
    lines.append("-+-".join("-" * widths[c] for c in cols))
    last = None
    for r in rows:
        if last is not None and r.get("resolution") != last:
            lines.append("")
        lines.append(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
        last = r.get("resolution")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())

    rows_out = []
    for r in spec["rows"]:
        row = {
            "resolution": r["resolution"],
            "method": r["method"],
            "status": r.get("status", "OK"),
            "mean_t_cm": fmt(r.get("mean_t_cm")),
            "mean_r_deg": fmt(r.get("mean_r_deg")),
            "cam0_t": fmt(r.get("cam0_t")),
            "cam0_r": fmt(r.get("cam0_r")),
            "cam1_t": fmt(r.get("cam1_t")),
            "cam1_r": fmt(r.get("cam1_r")),
            "cam3_t": fmt(r.get("cam3_t")),
            "cam3_r": fmt(r.get("cam3_r")),
            "cam5_t": fmt(r.get("cam5_t")),
            "cam5_r": fmt(r.get("cam5_r")),
            "note": r.get("note", ""),
        }
        rows_out.append(row)

    cols = [
        "resolution", "method", "status",
        "mean_t_cm", "mean_r_deg",
        "cam0_t", "cam0_r",
        "cam1_t", "cam1_r",
        "cam3_t", "cam3_r",
        "cam5_t", "cam5_r",
        "note",
    ]

    lines = []
    lines.append(spec["title"])
    lines.append("=" * len(spec["title"]))
    lines.append("")
    lines.append("Scope:")
    lines.append(spec["scope"])
    lines.append("Translation error in cm. Rotation error in degrees.")
    lines.append("")
    lines.append("Final source policy:")
    for item in spec["source_policy"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Final comparison table:")
    lines.append("")
    lines.append(md_table(rows_out, cols))
    lines.append("")

    if spec.get("interpretation"):
        lines.append("Compact interpretation:")
        for item in spec["interpretation"]:
            lines.append(f"- {item}")
        lines.append("")

    if spec.get("workflow"):
        lines.append("Reusable finalization workflow:")
        for i, item in enumerate(spec["workflow"], 1):
            lines.append(f"{i}. {item}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
