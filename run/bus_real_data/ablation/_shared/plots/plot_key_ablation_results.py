#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib.pyplot as plt


SUMMARY_FILES = [
    "results/bus_real_data/ablation/aruco_detection_motion_vs_gaussian_summary.csv",
    "results/bus_real_data/ablation/aruco_detection_lighting_pp_summary.csv",
    "results/bus_real_data/ablation/aruco_detection_exposure_summary.csv",
]

OUT_DIR = Path("results/bus_real_data/ablation/plots_key")
OUT_DIR.mkdir(parents=True, exist_ok=True)


LABELS = {
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


ORDER = {
    "original": 0,
    "blur_7": 10,
    "motion_7": 10,
    "blur_15": 11,
    "motion_15": 11,
    "blur_25": 12,
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


def read_all_rows():
    rows = {}

    for file in SUMMARY_FILES:
        path = Path(file)
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row["dataset"]

                # original can appear in multiple summary files; keep first
                if dataset in rows:
                    continue

                rows[dataset] = {
                    "dataset": dataset,
                    "detected_images": int(row["detected_images"]),
                    "detection_rate_percent": float(row["detection_rate_percent"]),
                    "avg_marker_count": float(row["avg_marker_count"]),
                    "total_detected_markers": int(row["total_detected_markers"]),
                    "num_unique_ids": int(row["num_unique_ids"]),
                }

    if "original" not in rows:
        raise SystemExit("No original row found in summary CSVs.")

    return rows


def select_main_rows(rows):
    # Nur Varianten, die wirklich die Story zeigen.
    wanted = [
        "original",
        "motion_7", "motion_15", "motion_25",
        "blur_7", "blur_15", "blur_25",
        "gaussian_7", "gaussian_15", "gaussian_25",
        "low_light_pp", "glare_pp",
        "ev_minus2", "ev_plus2",
    ]

    selected = []
    seen_labels = set()

    for key in wanted:
        if key not in rows:
            continue

        label = LABELS.get(key, key)

        # Falls blur_7 und motion_7 doppelt existieren, nur einmal zeigen.
        if label in seen_labels:
            continue

        selected.append(rows[key])
        seen_labels.add(label)

    return sorted(selected, key=lambda r: ORDER.get(r["dataset"], 999))


def plot_bar(rows, values, title, ylabel, output_path, ylim=None):
    labels = [LABELS.get(r["dataset"], r["dataset"]) for r in rows]

    plt.figure(figsize=(11, 5))
    bars = plt.bar(labels, values)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")

    if ylim:
        plt.ylim(*ylim)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote: {output_path}")


def main():
    rows = read_all_rows()
    original_total = rows["original"]["total_detected_markers"]
    original_ids = rows["original"]["num_unique_ids"]

    main_rows = select_main_rows(rows)

    # Plot 1: wichtigster Plot
    marker_retention = [
        100.0 * r["total_detected_markers"] / original_total
        for r in main_rows
    ]

    plot_bar(
        main_rows,
        marker_retention,
        "Ablation impact on ArUco marker detections",
        "Detected markers retained vs. original [%]",
        OUT_DIR / "01_marker_retention_overview.png",
        ylim=(0, max(marker_retention) * 1.15),
    )

    # Plot 2: ID-Verlust zeigen
    id_retention = [
        100.0 * r["num_unique_ids"] / original_ids
        for r in main_rows
    ]

    plot_bar(
        main_rows,
        id_retention,
        "Ablation impact on unique ArUco IDs",
        "Unique IDs retained vs. original [%]",
        OUT_DIR / "02_unique_id_retention_overview.png",
        ylim=(0, 110),
    )

    # Plot 3: Exposure separat, weil dort kaum Unterschied ist
    exposure_keys = ["original", "ev_minus2", "ev_minus1", "ev_plus1", "ev_plus2", "flicker"]
    exposure_rows = [rows[k] for k in exposure_keys if k in rows]

    exposure_values = [r["detection_rate_percent"] for r in exposure_rows]

    plot_bar(
        exposure_rows,
        exposure_values,
        "Exposure ablation: detection rate remains stable",
        "Detection rate [%]",
        OUT_DIR / "03_exposure_detection_rate.png",
        ylim=(0, 100),
    )


if __name__ == "__main__":
    main()
