#!/usr/bin/env python3
import csv
from pathlib import Path


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fields):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def format_table(rows, headers, keys):
    data = [headers]
    for r in rows:
        data.append([str(r.get(k, "")) for k in keys])
    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = []
    out.append(" | ".join(data[0][i].ljust(widths[i]) for i in range(len(headers))))
    out.append("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in data[1:]:
        out.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)
