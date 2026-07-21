#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing CSV: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: object, digits: int = 6) -> str:
    parsed = number(value)
    return "NA" if parsed is None else f"{parsed:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--observations-root", required=True)
    parser.add_argument("--ref-marker-id", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    observations_root = Path(args.observations_root).resolve()
    final = root / "99_FINAL_RESULTS"
    final.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Primary: GT-free camera-to-camera distances.
    # ------------------------------------------------------------------
    pairwise_path = final / "REAL_DATA_PAIRWISE_DISTANCES.csv"
    pairwise = read_csv(pairwise_path)

    primary_lines = [
        "REAL VEHICLE — PRIMARY CAMERA-TO-CAMERA RESULTS",
        "=" * 120,
        "",
        "Evaluation type: GT-free camera-rig geometry.",
        (
            "No simulation pose or marker-map position is used by the "
            "real calibration methods."
        ),
        (
            "Nominal simulation distances are context only and are not "
            "real-world accuracy measurements."
        ),
        "",
        (
            f"{'Camera pair':31}"
            f"{'AP01 [m]':>14}"
            f"{'AP02 [m]':>14}"
            f"{'AP03 [m]':>14}"
            f"{'Method spread [cm]':>22}"
        ),
        "-" * 120,
    ]

    for row in pairwise:
        pair = f"{row['camera_a']} - {row['camera_b']}"
        values = [
            number(row.get("ap01_distance_m")),
            number(row.get("ap02_distance_m")),
            number(row.get("ap03_distance_m")),
        ]
        finite = [value for value in values if value is not None]
        spread_cm = (
            100.0 * (max(finite) - min(finite))
            if len(finite) >= 2
            else None
        )

        primary_lines.append(
            f"{pair:31}"
            f"{fmt(row.get('ap01_distance_m')):>14}"
            f"{fmt(row.get('ap02_distance_m')):>14}"
            f"{fmt(row.get('ap03_distance_m')):>14}"
            f"{fmt(spread_cm, 3):>22}"
        )

    measured_rows = [
        row
        for row in pairwise
        if number(row.get("measured_real_reference_m")) is not None
    ]

    primary_lines += [
        "",
        "INDEPENDENT REAL MEASUREMENTS",
        "-" * 120,
    ]

    if measured_rows:
        for row in measured_rows:
            pair = f"{row['camera_a']} - {row['camera_b']}"
            primary_lines.append(
                f"{pair}: measured="
                f"{fmt(row.get('measured_real_reference_m'))} m"
            )
    else:
        primary_lines.append(
            "No independent physical measurements are currently entered."
        )

    primary_path = final / "REAL_PRIMARY_CAM_TO_CAM.txt"
    primary_path.write_text(
        "\n".join(primary_lines) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Secondary: AP02 GT-free marker map in reference-marker coordinates.
    # ------------------------------------------------------------------
    marker_source = (
        root
        / "03_ap02_real"
        / "07_graph_ba"
        / "with_moving"
        / "optimized_marker_poses_ref_marker.csv"
    )
    marker_rows = read_csv(marker_source)

    output_rows: list[dict[str, object]] = []

    for row in marker_rows:
        marker_id = int(float(row["entity_id"]))
        output_rows.append({
            "marker_id": marker_id,
            "reference_marker_id": args.ref_marker_id,
            "coordinate_frame": (
                f"marker_{args.ref_marker_id}_reference_frame"
            ),
            **row,
        })

    output_rows.sort(key=lambda row: int(row["marker_id"]))

    map_csv = final / "REAL_SECONDARY_AP02_MARKER_MAP_REF3.csv"
    write_csv(map_csv, output_rows)

    observation_rows = read_csv(
        observations_root / "shared_all_aruco_observations.csv"
    )

    observed_ids = sorted({
        int(float(row["marker_id"]))
        for row in observation_rows
        if str(row.get("pnp_success", "")).lower()
        in {"true", "1", "yes"}
    })

    optimized_ids = [
        int(row["marker_id"])
        for row in output_rows
    ]

    missing_ids = sorted(set(observed_ids) - set(optimized_ids))

    secondary_lines = [
        "REAL VEHICLE — SECONDARY AP02 MARKER MAP",
        "=" * 120,
        "",
        f"Reference marker: {args.ref_marker_id}",
        (
            f"Coordinate frame: marker_{args.ref_marker_id}_reference_frame"
        ),
        "Ground truth used: false",
        (
            "This is an internally estimated metric marker map. "
            "No simulation marker positions are used or compared."
        ),
        "",
        f"Observed marker IDs:  {observed_ids}",
        f"Optimized marker IDs: {optimized_ids}",
        f"Observed but missing from optimized map: {missing_ids}",
        "",
        (
            f"{'Marker':>8}"
            f"{'x [m]':>14}"
            f"{'y [m]':>14}"
            f"{'z [m]':>14}"
            f"{'rvec x':>14}"
            f"{'rvec y':>14}"
            f"{'rvec z':>14}"
        ),
        "-" * 120,
    ]

    for row in output_rows:
        secondary_lines.append(
            f"{int(row['marker_id']):>8}"
            f"{fmt(row.get('x_m')):>14}"
            f"{fmt(row.get('y_m')):>14}"
            f"{fmt(row.get('z_m')):>14}"
            f"{fmt(row.get('rvec_x')):>14}"
            f"{fmt(row.get('rvec_y')):>14}"
            f"{fmt(row.get('rvec_z')):>14}"
        )

    map_txt = final / "REAL_SECONDARY_AP02_MARKER_MAP_REF3.txt"
    map_txt.write_text(
        "\n".join(secondary_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "primary_report": str(primary_path),
        "secondary_report": str(map_txt),
        "secondary_csv": str(map_csv),
        "reference_marker_id": args.ref_marker_id,
        "ground_truth_used": False,
        "simulation_marker_map_used": False,
    }

    (
        final / "REAL_PRIMARY_SECONDARY_MANIFEST.json"
    ).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[OK] wrote:")
    print(primary_path)
    print(map_txt)
    print(map_csv)


if __name__ == "__main__":
    main()
