#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_csv", required=True)
    args = parser.parse_args()

    p = Path(args.trajectory_csv)
    if not p.exists():
        raise FileNotFoundError(p)

    groups = defaultdict(list)

    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("tag", "")
            err = float(row["pos_error_m"])
            groups[tag].append(err)

    print("tag,count,mean_cm,median_cm,max_cm,rmse_cm")
    for tag, vals in sorted(groups.items()):
        vals_sorted = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        median = vals_sorted[n // 2]
        max_err = max(vals)
        rmse = math.sqrt(sum(v*v for v in vals) / n)
        print(f"{tag},{n},{mean*100:.2f},{median*100:.2f},{max_err*100:.2f},{rmse*100:.2f}")


if __name__ == "__main__":
    main()
