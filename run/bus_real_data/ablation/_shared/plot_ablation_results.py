#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import matplotlib.pyplot as plt


def read_summary_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "dataset": row["dataset"],
                "num_images": int(row["num_images"]),
                "detected_images": int(row["detected_images"]),
                "detection_rate_percent": float(row["detection_rate_percent"]),
                "avg_marker_count": float(row["avg_marker_count"]),
                "total_detected_markers": int(row["total_detected_markers"]),
                "num_unique_ids": int(row["num_unique_ids"]),
                "lost_ids_vs_original": row.get("lost_ids_vs_original", ""),
            })
    return rows


def dataset_order(name):
    order = {
        "original": 0,

        "blur_7": 10,
        "blur_15": 11,
        "blur_25": 12,
        "motion_7": 10,
        "motion_15": 11,
        "motion_25": 12,

        "gaussian_7": 20,
        "gaussian_15": 21,
        "gaussian_25": 22,

        "low_light_pp": 30,
        "side_light_pp": 31,
        "glare_pp": 32,

        "ev_minus2": 40,
        "ev_minus1": 41,
        "ev_plus1": 42,
        "ev_plus2": 43,
        "flicker": 44,
    }
    return order.get(name, 999)


def pretty_label(name):
    labels = {
        "original": "Original",

        "blur_7": "Motion 7",
        "blur_15": "Motion 15",
        "blur_25": "Motion 25",
        "motion_7": "Motion 7",
        "motion_15": "Motion 15",
        "motion_25": "Motion 25",

        "gaussian_7": "Gaussian 7",
        "gaussian_15": "Gaussian 15",
        "gaussian_25": "Gaussian 25",

        "low_light_pp": "Low light",
        "side_light_pp": "Side light",
        "glare_pp": "Glare",

        "ev_minus2": "EV -2",
        "ev_minus1": "EV -1",
        "ev_plus1": "EV +1",
        "ev_plus2": "EV +2",
        "flicker": "Flicker",
    }
    return labels.get(name, name)


def plot_metric(rows, metric, ylabel, title, output_path):
    rows = sorted(rows, key=lambda r: dataset_order(r["dataset"]))

    labels = [pretty_label(r["dataset"]) for r in rows]
    values = [r[metric] for r in rows]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    for bar, value in zip(bars, values):
        height = bar.get_height()
        if isinstance(value, float):
            text = f"{value:.2f}"
        else:
            text = str(value)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Wrote: {output_path}")


def plot_summary(csv_path, output_dir):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    rows = read_summary_csv(csv_path)
    stem = csv_path.stem.replace("aruco_detection_", "").replace("_summary", "")

    plot_metric(
        rows,
        "detection_rate_percent",
        "Detection rate [%]",
        f"ArUco detection rate - {stem}",
        output_dir / f"{stem}_detection_rate.png",
    )

    plot_metric(
        rows,
        "avg_marker_count",
        "Average markers per image",
        f"Average detected ArUco markers - {stem}",
        output_dir / f"{stem}_avg_marker_count.png",
    )

    plot_metric(
        rows,
        "total_detected_markers",
        "Total detected markers",
        f"Total detected ArUco markers - {stem}",
        output_dir / f"{stem}_total_markers.png",
    )

    plot_metric(
        rows,
        "num_unique_ids",
        "Unique marker IDs",
        f"Unique ArUco IDs - {stem}",
        output_dir / f"{stem}_unique_ids.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more summary CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/bus_real_data/ablation/plots",
    )
    args = parser.parse_args()

    for csv_path in args.csv:
        plot_summary(csv_path, args.output_dir)


if __name__ == "__main__":
    main()
