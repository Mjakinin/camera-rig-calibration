#!/usr/bin/env python3
# AUTO_IMPORT_COMMON_START
from pathlib import Path as _CalibLabPath
import sys as _CalibLabSys
for _p in _CalibLabPath(__file__).resolve().parents:
    if _p.name == "calib_lab":
        if str(_p) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_p))
        _common_scripts = _p / "common" / "scripts"
        if _common_scripts.exists() and str(_common_scripts) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_common_scripts))
        break
# AUTO_IMPORT_COMMON_END

import csv
import argparse
from pathlib import Path


def read_summary(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["scenario"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--a_name", default="A")
    parser.add_argument("--b", required=True)
    parser.add_argument("--b_name", default="B")
    args = parser.parse_args()

    a = read_summary(args.a)
    b = read_summary(args.b)

    scenarios = sorted(set(a.keys()) | set(b.keys()))

    print(
        "scenario, "
        f"{args.a_name}_class, {args.a_name}_err_cm, {args.a_name}_rot, "
        f"{args.b_name}_class, {args.b_name}_err_cm, {args.b_name}_rot, "
        "change"
    )

    for s in scenarios:
        ra = a.get(s, {})
        rb = b.get(s, {})

        ca = ra.get("pose_class", "missing")
        cb = rb.get("pose_class", "missing")

        ea = ra.get("baseline_error_cm", "")
        eb = rb.get("baseline_error_cm", "")
        rota = ra.get("rotation_error_deg", "")
        rotb = rb.get("rotation_error_deg", "")

        if ca != cb:
            change = f"{ca}->{cb}"
        else:
            change = "same"

        print(f"{s}, {ca}, {ea}, {rota}, {cb}, {eb}, {rotb}, {change}")

    def counts(data):
        c = {}
        for r in data.values():
            cls = r["pose_class"]
            c[cls] = c.get(cls, 0) + 1
        return c

    print("\nCounts:")
    print(args.a_name, counts(a))
    print(args.b_name, counts(b))


if __name__ == "__main__":
    main()
