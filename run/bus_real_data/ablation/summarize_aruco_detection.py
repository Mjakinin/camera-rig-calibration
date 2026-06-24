#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


def parse_ids(s):
    if pd.isna(s) or str(s).strip() == "":
        return set()
    return set(int(x) for x in str(s).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    original_ids = set()
    if "original" in set(df["dataset"]):
        for s in df[df["dataset"] == "original"]["ids"]:
            original_ids |= parse_ids(s)

    rows = []

    for dataset, group in df.groupby("dataset"):
        all_ids = set()
        for s in group["ids"]:
            all_ids |= parse_ids(s)

        num_images = len(group)
        detected_images = int((group["marker_count"] > 0).sum())
        detection_rate = detected_images / num_images if num_images else 0.0
        avg_marker_count = float(group["marker_count"].mean()) if num_images else 0.0
        total_markers = int(group["marker_count"].sum())

        lost_ids = sorted(original_ids - all_ids)

        rows.append({
            "dataset": dataset,
            "num_images": num_images,
            "detected_images": detected_images,
            "detection_rate_percent": round(detection_rate * 100, 2),
            "avg_marker_count": round(avg_marker_count, 3),
            "total_detected_markers": total_markers,
            "num_unique_ids": len(all_ids),
            "all_ids": " ".join(map(str, sorted(all_ids))),
            "lost_ids_vs_original": " ".join(map(str, lost_ids)),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("dataset")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(out.to_string(index=False))
    print()
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
