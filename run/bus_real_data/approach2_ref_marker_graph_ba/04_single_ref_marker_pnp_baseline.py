#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from ap02_common import (
    AP02_ROOT,
    DEFAULT_REF_MARKER_ID,
    ensure_dir,
    read_csv,
    write_csv,
    make_T,
    T_from_detection_row,
    make_observer_known_from_marker,
    pose_row,
    pose_fields,
)


OBS_CSV = AP02_ROOT / "02_aruco_observations" / "ap02_static_aruco_observations.csv"


def is_success(row):
    return str(row.get("pnp_success", "")).strip().lower() in ["true", "1", "yes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-marker-id", type=int, default=DEFAULT_REF_MARKER_ID)
    ap.add_argument("--out", default=str(AP02_ROOT / "04_single_ref_marker_pnp"))
    args = ap.parse_args()

    out = ensure_dir(Path(args.out))

    rows = read_csv(OBS_CSV)
    T_ref_marker = make_T(np.eye(3), np.zeros(3))

    pose_rows = []
    seen_ref = []
    all_static_cams = sorted({r["observer_id"] for r in rows if r.get("observer_type") == "static"})

    for r in rows:
        if not is_success(r):
            continue
        if int(float(r["marker_id"])) != args.ref_marker_id:
            continue

        T_cam_marker = T_from_detection_row(r)
        if T_cam_marker is None:
            continue

        T_ref_cam = make_observer_known_from_marker(T_ref_marker, T_cam_marker)

        pose_rows.append(
            pose_row(
                entity_type="static_camera",
                entity_id=r["observer_id"],
                T=T_ref_cam,
                source=f"single_ref_marker_{args.ref_marker_id}_pnp",
            )
        )
        seen_ref.append(r["observer_id"])

    seen_ref = sorted(set(seen_ref))
    not_seen = sorted(set(all_static_cams) - set(seen_ref))

    write_csv(out / "single_ref_marker_static_camera_poses_ref_marker.csv", pose_rows, pose_fields())

    report = [
        "AP02 single reference-marker PnP baseline",
        "=========================================",
        "",
        f"Reference marker id: {args.ref_marker_id}",
        "",
        "Goal:",
        "Estimate T_ref_marker_cam directly for every static camera that sees the reference marker.",
        "",
        f"Static cameras in AP02 observations: {all_static_cams}",
        f"Static cameras seeing reference marker: {seen_ref}",
        f"Static cameras NOT seeing reference marker: {not_seen}",
        "",
        f"Estimated static camera poses: {len(pose_rows)}",
        "",
        "Interpretation:",
        "- If a static camera is not listed, it cannot be calibrated by the naive single-reference-marker method.",
        "- Missing cameras motivate the marker-map graph method.",
        "- This AP02 script reads only AP02 observations, not AP01 result internals.",
        "",
    ]

    (out / "single_ref_marker_report.txt").write_text("\n".join(report) + "\n")

    print("[OK] wrote", out)
    print("[OK] reference marker:", args.ref_marker_id)
    print("[OK] direct static cameras:", seen_ref)
    print("[OK] missing static cameras:", not_seen)


if __name__ == "__main__":
    main()
