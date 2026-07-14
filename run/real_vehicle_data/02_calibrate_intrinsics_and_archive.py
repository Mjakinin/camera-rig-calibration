#!/usr/bin/env python3
"""Run the existing intrinsic calibration and archive a readable result per video.

The numerical calibration remains implemented in
``02_calibrate_intrinsics_from_video.py``.  This wrapper preserves that tested
pipeline and adds a stable, per-video archive under
``results/real_vehicle_data/INTRINSIC_RESULTS``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENGINE = HERE / "02_calibrate_intrinsics_from_video.py"
DEFAULT_ARCHIVE_ROOT = Path("results/real_vehicle_data/INTRINSIC_RESULTS")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "intrinsic_calibration"


def matrix_lines(flat: list[float]) -> list[str]:
    if len(flat) != 9:
        return [str(flat)]
    return [
        f"[{flat[0]: .10f} {flat[1]: .10f} {flat[2]: .10f}]",
        f"[{flat[3]: .10f} {flat[4]: .10f} {flat[5]: .10f}]",
        f"[{flat[6]: .10f} {flat[7]: .10f} {flat[8]: .10f}]",
    ]


def readable_report(label: str, data: dict, json_path: Path) -> str:
    comparison = data.get("model_comparison", {})
    removed = data.get("removed_outlier_views", [])
    K = [float(value) for value in data.get("K", data.get("k", []))]
    D = [float(value) for value in data.get("D", data.get("d", []))]

    lines = [
        f"INTRINSIC CALIBRATION RESULT — {label}",
        "=" * 88,
        "",
        "SOURCE",
        "------",
        f"Video: {data.get('source_video', '-')}",
        f"Resolution: {data.get('width', '-')}x{data.get('height', '-')}",
        f"Source FPS: {data.get('source_fps', '-')}",
        "",
        "CHECKERBOARD AND VIEW SELECTION",
        "-------------------------------",
        (
            "Inner corners: "
            f"{data.get('checkerboard_inner_corners', {}).get('columns', '-')}x"
            f"{data.get('checkerboard_inner_corners', {}).get('rows', '-')}"
        ),
        f"Original frames scanned: {data.get('scanned_original_frames', '-')}",
        (
            "Successful checkerboard detections: "
            f"{data.get('successful_checkerboard_detections', '-')}"
        ),
        f"Final calibration views: {data.get('selected_calibration_views', '-')}",
        f"Removed outlier views: {len(removed)}",
        "",
        "MODEL AND QUALITY",
        "-----------------",
        f"Selected distortion model: {data.get('distortion_model', '-')}",
        (
            "Standard holdout median RMSE [px]: "
            f"{comparison.get('standard_holdout_median_rmse_px', '-')}"
        ),
        (
            "Rational holdout median RMSE [px]: "
            f"{comparison.get('rational_holdout_median_rmse_px', '-')}"
        ),
        f"OpenCV calibration RMS [px]: {data.get('opencv_calibration_rms_px', '-')}",
        (
            "Median per-view reprojection RMSE [px]: "
            f"{data.get('median_view_reprojection_rmse_px', '-')}"
        ),
        (
            "Maximum retained per-view RMSE [px]: "
            f"{data.get('maximum_view_reprojection_rmse_px', '-')}"
        ),
        "",
        "CAMERA MATRIX K",
        "---------------",
        *matrix_lines(K),
        "",
        "SCALAR INTRINSICS",
        "-----------------",
        f"fx [px]: {data.get('fx', '-')}",
        f"fy [px]: {data.get('fy', '-')}",
        f"cx [px]: {data.get('cx', '-')}",
        f"cy [px]: {data.get('cy', '-')}",
        "",
        "DISTORTION D",
        "------------",
        "[" + ", ".join(f"{value:.12g}" for value in D) + "]",
        "",
        "OUTLIER VIEWS",
        "-------------",
    ]

    if removed:
        for row in removed:
            lines.append(
                f"- frame {row.get('frame_index', '-')}: "
                f"RMSE={row.get('reprojection_rmse_px', '-')} px"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "FILES",
            "-----",
            f"Archived CameraInfo JSON: {json_path}",
            "",
            "Interpretation:",
            "- K contains focal lengths and principal point in pixels.",
            "- D contains lens-distortion coefficients for the selected model.",
            "- The result is valid for this physical camera, resolution and capture mode.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate intrinsics from one video and archive one readable TXT "
            "and CameraInfo JSON per source video."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--max-views", type=int, default=80)
    parser.add_argument("--minimum-frame-gap", type=int, default=5)
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument(
        "--result-name",
        default="",
        help="Optional stable label, for example iphone_05x_4k or iphone_1x_4k.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video).resolve()
    out = Path(args.out)
    archive_root = Path(args.archive_root)

    if not ENGINE.is_file():
        raise RuntimeError(f"Missing intrinsic calibration engine: {ENGINE}")

    command = [
        sys.executable,
        str(ENGINE),
        "--video",
        str(video),
        "--out",
        str(out),
        "--cols",
        str(args.cols),
        "--rows",
        str(args.rows),
        "--max-views",
        str(args.max_views),
        "--minimum-frame-gap",
        str(args.minimum_frame_gap),
    ]

    print("[CMD]", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    source_json = out / "moving_calib_camera.json"
    if not source_json.is_file():
        raise RuntimeError(f"Calibration did not create {source_json}")

    data = json.loads(source_json.read_text(encoding="utf-8"))
    label = args.result_name.strip() or video.stem
    base = safe_name(
        f"{label}_{data.get('width', 'unknown')}x{data.get('height', 'unknown')}"
    )

    archive_root.mkdir(parents=True, exist_ok=True)
    archived_json = archive_root / f"{base}_moving_calib_camera.json"
    archived_report = archive_root / f"{base}_INTRINSICS_REPORT.txt"

    shutil.copy2(source_json, archived_json)
    archived_report.write_text(
        readable_report(label, data, archived_json),
        encoding="utf-8",
    )

    print("[OK] intrinsic calibration archive updated")
    print("[OK]", archived_report)
    print("[OK]", archived_json)


if __name__ == "__main__":
    main()
