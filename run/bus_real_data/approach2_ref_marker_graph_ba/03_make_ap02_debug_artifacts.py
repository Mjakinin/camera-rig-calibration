#!/usr/bin/env python3

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ap02_common import AP02_ROOT, ensure_dir, read_csv, write_csv


OBS_CSV = AP02_ROOT / "02_aruco_observations" / "ap02_all_aruco_observations.csv"


def main():
    rows = read_csv(OBS_CSV)
    out = ensure_dir(AP02_ROOT / "03_debug_artifacts")

    observers = sorted({r["observer_id"] for r in rows})
    markers = sorted({int(float(r["marker_id"])) for r in rows})

    counts = defaultdict(int)
    for r in rows:
        counts[(r["observer_id"], int(float(r["marker_id"])))] += 1

    matrix_rows = []
    for obs in observers:
        row = {"observer_id": obs}
        for marker in markers:
            row[f"marker_{marker}"] = counts[(obs, marker)]
        matrix_rows.append(row)

    fields = ["observer_id"] + [f"marker_{m}" for m in markers]
    write_csv(out / "ap02_observer_marker_visibility_matrix.csv", matrix_rows, fields)

    if observers and markers:
        mat = np.array([[counts[(obs, m)] for m in markers] for obs in observers], dtype=float)
        fig, ax = plt.subplots(figsize=(max(8, len(markers) * 0.55), max(4, len(observers) * 0.35)))
        ax.imshow(mat, aspect="auto")
        ax.set_xticks(range(len(markers)))
        ax.set_xticklabels([str(m) for m in markers], rotation=90)
        ax.set_yticks(range(len(observers)))
        ax.set_yticklabels(observers)
        ax.set_xlabel("marker id")
        ax.set_ylabel("observer")
        ax.set_title("AP02 observer-marker visibility")
        fig.tight_layout()
        fig.savefig(out / "ap02_observer_marker_visibility_matrix.png", dpi=180)
        plt.close(fig)

    marker_counts = Counter(int(float(r["marker_id"])) for r in rows)
    marker_rows = [
        {"marker_id": marker, "observation_count": count}
        for marker, count in sorted(marker_counts.items())
    ]
    write_csv(out / "ap02_marker_observation_counts.csv", marker_rows, ["marker_id", "observation_count"])

    if marker_rows:
        x = np.arange(len(marker_rows))
        labels = [str(r["marker_id"]) for r in marker_rows]
        values = [r["observation_count"] for r in marker_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("marker id")
        ax.set_ylabel("observation count")
        ax.set_title("AP02 marker observation counts")
        fig.tight_layout()
        fig.savefig(out / "ap02_marker_observation_counts.png", dpi=180)
        plt.close(fig)

    report = [
        "AP02 debug artifact report",
        "==========================",
        "",
        f"Input observations: {len(rows)}",
        f"Observers: {len(observers)}",
        f"Markers: {len(markers)}",
        "",
        "Generated:",
        "- ap02_observer_marker_visibility_matrix.csv",
        "- ap02_observer_marker_visibility_matrix.png",
        "- ap02_marker_observation_counts.csv",
        "- ap02_marker_observation_counts.png",
        "",
    ]
    (out / "ap02_debug_artifacts_report.txt").write_text("\n".join(report))
    print("[OK] wrote debug artifacts:", out)


if __name__ == "__main__":
    main()
