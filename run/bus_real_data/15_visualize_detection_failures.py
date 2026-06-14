#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


EXPECTED_IDS_DEFAULT = list(range(14))


def parse_expected_ids(text):
    if text is None or text.strip() == "":
        return EXPECTED_IDS_DEFAULT
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_frame_number(path):
    return int(path.stem.split("_")[1])


def row_corners(row):
    pts = np.array([
        [float(row["corner0_u"]), float(row["corner0_v"])],
        [float(row["corner1_u"]), float(row["corner1_v"])],
        [float(row["corner2_u"]), float(row["corner2_v"])],
        [float(row["corner3_u"]), float(row["corner3_v"])],
    ], dtype=np.float32)
    return pts


def load_detections(csv_path):
    csv_path = Path(csv_path)
    by_frame = {}
    by_marker = {}
    all_ids = set()

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open() as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            marker_id = int(row["marker_id"])

            by_frame.setdefault(frame, []).append(row)
            by_marker.setdefault(marker_id, []).append(row)
            all_ids.add(marker_id)

    return by_frame, by_marker, all_ids


def ids_in_frame(rows):
    return sorted({int(r["marker_id"]) for r in rows})


def draw_label(img, text, x, y, color, scale=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = max(0, min(int(x), img.shape[1] - tw - 4))
    y = max(th + 4, min(int(y), img.shape[0] - 4))

    cv2.rectangle(
        img,
        (x, y - th - baseline - 4),
        (x + tw + 4, y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img,
        text,
        (x + 2, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_marker_polygon(img, row, color, label_prefix):
    marker_id = int(row["marker_id"])
    pts = row_corners(row).astype(np.int32)

    cv2.polylines(
        img,
        [pts],
        isClosed=True,
        color=color,
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    center = pts.mean(axis=0)
    draw_label(
        img,
        f"{label_prefix}{marker_id}",
        int(center[0]),
        int(center[1]),
        color,
        scale=0.6,
    )


def annotate_frame(
    image_path,
    frame,
    degraded_rows,
    baseline_rows,
    expected_ids,
    global_missing_ids,
    out_path,
):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] could not read {image_path}")
        return False

    degraded_ids = set(ids_in_frame(degraded_rows))
    baseline_ids = set(ids_in_frame(baseline_rows))

    missed_vs_baseline = sorted(baseline_ids - degraded_ids)

    # Draw baseline-visible but missed markers first in red.
    for row in baseline_rows:
        marker_id = int(row["marker_id"])
        if marker_id in missed_vs_baseline:
            draw_marker_polygon(
                img,
                row,
                color=(0, 0, 255),
                label_prefix="missed ",
            )

    # Draw detected degraded markers in green.
    for row in degraded_rows:
        draw_marker_polygon(
            img,
            row,
            color=(0, 255, 0),
            label_prefix="id ",
        )

    # Header overlay
    cv2.rectangle(img, (0, 0), (img.shape[1], 105), (0, 0, 0), -1)

    draw_label(
        img,
        f"frame {frame:04d}",
        12,
        28,
        color=(255, 255, 255),
        scale=0.7,
    )
    draw_label(
        img,
        f"detected in degraded: {sorted(degraded_ids)}",
        12,
        58,
        color=(0, 255, 0),
        scale=0.55,
    )
    draw_label(
        img,
        f"missed vs baseline in this frame: {missed_vs_baseline}",
        12,
        84,
        color=(0, 0, 255),
        scale=0.55,
    )

    # Footer overlay
    cv2.rectangle(
        img,
        (0, img.shape[0] - 38),
        (img.shape[1], img.shape[0]),
        (0, 0, 0),
        -1,
    )
    draw_label(
        img,
        f"global missing IDs in this sequence: {global_missing_ids}",
        12,
        img.shape[0] - 12,
        color=(0, 200, 255),
        scale=0.55,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def make_contact_sheet(image_paths, out_path, thumb_w=420, cols=3):
    images = []

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_w / float(w)
        thumb_h = int(h * scale)
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        images.append(thumb)

    if not images:
        print("[WARN] no images for contact sheet")
        return

    thumb_h = images[0].shape[0]
    rows = int(np.ceil(len(images) / cols))

    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y0 = r * thumb_h
        x0 = c * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def choose_frames(
    image_frames,
    degraded_by_frame,
    baseline_by_frame,
    global_missing_ids,
    max_frames,
):
    selected = set()

    # Include frames where degraded sequence detects nothing but baseline did.
    for frame in image_frames:
        degraded_ids = set(ids_in_frame(degraded_by_frame.get(frame, [])))
        baseline_ids = set(ids_in_frame(baseline_by_frame.get(frame, [])))

        if baseline_ids and not degraded_ids:
            selected.add(frame)

    # Include frames where globally missing IDs were visible in baseline.
    for frame in image_frames:
        baseline_ids = set(ids_in_frame(baseline_by_frame.get(frame, [])))
        if baseline_ids.intersection(global_missing_ids):
            selected.add(frame)

    # Add evenly spaced frames for context.
    step = max(1, len(image_frames) // 12)
    for frame in image_frames[::step]:
        selected.add(frame)

    selected = sorted(selected)

    if len(selected) > max_frames:
        # Keep a representative subset.
        idxs = np.linspace(0, len(selected) - 1, max_frames).round().astype(int)
        selected = [selected[i] for i in idxs]

    return selected


def write_missed_csv(
    out_csv,
    image_frames,
    degraded_by_frame,
    baseline_by_frame,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "frame",
            "baseline_ids",
            "degraded_ids",
            "missed_vs_baseline",
            "extra_in_degraded",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for frame in image_frames:
            baseline_ids = set(ids_in_frame(baseline_by_frame.get(frame, [])))
            degraded_ids = set(ids_in_frame(degraded_by_frame.get(frame, [])))

            missed = sorted(baseline_ids - degraded_ids)
            extra = sorted(degraded_ids - baseline_ids)

            w.writerow({
                "frame": frame,
                "baseline_ids": ";".join(map(str, sorted(baseline_ids))),
                "degraded_ids": ";".join(map(str, sorted(degraded_ids))),
                "missed_vs_baseline": ";".join(map(str, missed)),
                "extra_in_degraded": ";".join(map(str, extra)),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sequence",
        required=True,
        help="Degraded sequence folder, e.g. results/.../03_moving_camera_sequence_motion_blur_15",
    )
    ap.add_argument(
        "--baseline-sequence",
        default="results/bus_real_data/03_moving_camera_sequence",
        help="Baseline sequence folder for comparison",
    )
    ap.add_argument(
        "--expected-ids",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output visualization folder. Default: <sequence>/visualization",
    )
    ap.add_argument(
        "--max-sheet-frames",
        type=int,
        default=30,
    )
    ap.add_argument(
        "--all-frames",
        action="store_true",
        help="Write annotated images for all frames, not only selected frames",
    )
    args = ap.parse_args()

    seq = Path(args.sequence)
    baseline_seq = Path(args.baseline_sequence)

    img_dir = seq / "images"
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    out_dir = Path(args.out) if args.out else seq / "visualization"
    annotated_dir = out_dir / "annotated_frames"

    expected_ids = parse_expected_ids(args.expected_ids)

    degraded_csv = seq / "moving_detections.csv"
    baseline_csv = baseline_seq / "moving_detections.csv"

    degraded_by_frame, degraded_by_marker, degraded_global_ids = load_detections(degraded_csv)
    baseline_by_frame, baseline_by_marker, baseline_global_ids = load_detections(baseline_csv)

    global_missing_ids = sorted(set(expected_ids) - degraded_global_ids)

    image_paths = sorted(img_dir.glob("frame_*.png"))
    image_frames = [parse_frame_number(p) for p in image_paths]
    image_by_frame = {parse_frame_number(p): p for p in image_paths}

    selected_frames = choose_frames(
        image_frames,
        degraded_by_frame,
        baseline_by_frame,
        set(global_missing_ids),
        args.max_sheet_frames,
    )

    frames_to_write = image_frames if args.all_frames else selected_frames

    written = []
    for frame in frames_to_write:
        img_path = image_by_frame[frame]
        out_path = annotated_dir / f"frame_{frame:04d}_annotated.png"

        ok = annotate_frame(
            img_path,
            frame,
            degraded_by_frame.get(frame, []),
            baseline_by_frame.get(frame, []),
            expected_ids,
            global_missing_ids,
            out_path,
        )
        if ok and frame in selected_frames:
            written.append(out_path)

    make_contact_sheet(
        written,
        out_dir / "contact_sheet_detection_failures.png",
    )

    write_missed_csv(
        out_dir / "missed_vs_baseline_by_frame.csv",
        image_frames,
        degraded_by_frame,
        baseline_by_frame,
    )

    missing_txt = out_dir / "missing_ids_summary.txt"
    with missing_txt.open("w") as f:
        f.write("Detection failure visualization summary\n")
        f.write("======================================\n\n")
        f.write(f"Sequence: {seq}\n")
        f.write(f"Baseline sequence: {baseline_seq}\n\n")
        f.write(f"Expected IDs: {expected_ids}\n")
        f.write(f"Detected IDs in degraded sequence: {sorted(degraded_global_ids)}\n")
        f.write(f"Global missing IDs in degraded sequence: {global_missing_ids}\n\n")

        f.write("Baseline-visible frames for globally missing IDs:\n")
        for marker_id in global_missing_ids:
            frames = sorted({
                int(r["frame"])
                for r in baseline_by_marker.get(marker_id, [])
            })
            f.write(f"  marker {marker_id}: {frames}\n")

        f.write("\nLegend:\n")
        f.write("  green polygon: detected in degraded sequence\n")
        f.write("  red polygon: detected in baseline at this frame, but missed in degraded sequence\n")

    print("[OK] visualization folder:", out_dir)
    print("[OK] contact sheet:", out_dir / "contact_sheet_detection_failures.png")
    print("[OK] annotated frames:", annotated_dir)
    print("[OK] missed CSV:", out_dir / "missed_vs_baseline_by_frame.csv")
    print("[OK] missing summary:", missing_txt)
    print()
    print("Detected IDs:", sorted(degraded_global_ids))
    print("Global missing IDs:", global_missing_ids)


if __name__ == "__main__":
    main()
