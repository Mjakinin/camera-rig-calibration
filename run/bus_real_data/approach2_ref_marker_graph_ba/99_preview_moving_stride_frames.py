#!/usr/bin/env python3

import csv
import re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


OBS_CSV = Path("results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/ap02_all_aruco_observations.csv")
OUT = Path("results/bus_real_data/02_ref_marker_graph_ba/99_moving_stride_preview")
OUT.mkdir(parents=True, exist_ok=True)

STRIDES = [2, 3, 5, 6, 10, 20]

THUMB_W = 320
THUMB_H = 180
COLS = 4


def is_success(row):
    return str(row.get("pnp_success", "")).strip().lower() in ["true", "1", "yes"]


def frame_number_from_text(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else 10**9


def frame_key(row):
    # Prefer observer_id because AP02 moving frames are usually moving_frame_XXXX.
    for k in ["observer_id", "frame_id", "image_path", "image_file", "filename", "path"]:
        if row.get(k):
            return row[k]
    return "unknown"


def sort_key(frame_id):
    return frame_number_from_text(frame_id)


def image_path_from_row(row):
    candidates = []
    for k in ["image_path", "image_file", "filename", "path", "source_image", "image"]:
        v = row.get(k, "")
        if v:
            candidates.append(Path(v))

    frame_id = frame_key(row)
    n = frame_number_from_text(frame_id)
    if n < 10**9:
        names = [
            f"frame_{n:04d}.png",
            f"frame_{n:05d}.png",
            f"{n:04d}.png",
            f"{n:05d}.png",
        ]
        roots = [
            Path("results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images/moving/images"),
            Path("results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images/moving"),
            Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/.ap01_compat_cache/moving_observations/images"),
            Path("results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/images"),
        ]
        for root in roots:
            for name in names:
                candidates.append(root / name)

    for p in candidates:
        if p.exists():
            return p

    return None


def get_corner(row, idx):
    # Supports corner0_u/corner0_v and also c0_u/c0_v style if present.
    keys = [
        (f"corner{idx}_u", f"corner{idx}_v"),
        (f"corner_{idx}_u", f"corner_{idx}_v"),
        (f"c{idx}_u", f"c{idx}_v"),
        (f"u{idx}", f"v{idx}"),
    ]
    for ku, kv in keys:
        if ku in row and kv in row:
            try:
                return float(row[ku]), float(row[kv])
            except Exception:
                pass
    return None


def draw_frame(rows):
    img_path = image_path_from_row(rows[0])
    if img_path is None:
        canvas = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        cv2.putText(canvas, "IMAGE MISSING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return canvas, None

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        canvas = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        cv2.putText(canvas, "READ FAILED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return canvas, img_path

    h, w = img.shape[:2]

    for row in rows:
        pts = []
        for i in range(4):
            c = get_corner(row, i)
            if c is not None:
                pts.append(c)

        if len(pts) == 4:
            p = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [p], True, (0, 255, 0), 3)
            cx = int(sum(x for x, y in pts) / 4)
            cy = int(sum(y for x, y in pts) / 4)
            cv2.putText(img, str(row.get("marker_id", "?")), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    thumb = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    return thumb, img_path


def make_contact_sheet(selected_frames, grouped, stride):
    thumbs = []
    stats_rows = []

    for frame_id in selected_frames:
        rows = grouped[frame_id]
        thumb, img_path = draw_frame(rows)

        marker_ids = sorted({str(r.get("marker_id", "")) for r in rows})
        label1 = f"{frame_id} | markers={len(marker_ids)}"
        label2 = ",".join(marker_ids[:8]) + ("..." if len(marker_ids) > 8 else "")

        cv2.rectangle(thumb, (0, 0), (THUMB_W, 42), (0, 0, 0), -1)
        cv2.putText(thumb, label1[:42], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(thumb, label2[:42], (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        thumbs.append(thumb)

        stats_rows.append({
            "stride": stride,
            "frame_id": frame_id,
            "image_path": str(img_path) if img_path else "",
            "num_marker_observations": len(rows),
            "unique_marker_ids": ";".join(marker_ids),
        })

    rows_n = int(np.ceil(len(thumbs) / COLS))
    sheet = np.zeros((rows_n * THUMB_H, COLS * THUMB_W, 3), dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        r = idx // COLS
        c = idx % COLS
        sheet[r*THUMB_H:(r+1)*THUMB_H, c*THUMB_W:(c+1)*THUMB_W] = thumb

    img_out = OUT / f"moving_stride_{stride}_preview.jpg"
    cv2.imwrite(str(img_out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    csv_out = OUT / f"moving_stride_{stride}_selected_frames.csv"
    with csv_out.open("w", newline="") as fp:
        fields = ["stride", "frame_id", "image_path", "num_marker_observations", "unique_marker_ids"]
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(stats_rows)

    return img_out, csv_out, stats_rows


def main():
    if not OBS_CSV.exists():
        raise SystemExit(f"Missing: {OBS_CSV}")

    all_rows = list(csv.DictReader(OBS_CSV.open()))
    moving_rows = [
        r for r in all_rows
        if r.get("observer_type") == "moving" and is_success(r)
    ]

    grouped = defaultdict(list)
    for r in moving_rows:
        grouped[frame_key(r)].append(r)

    frames = sorted(grouped.keys(), key=sort_key)

    print(f"[INFO] AP02 moving frames with successful ArUco/PnP observations: {len(frames)}")
    print(f"[INFO] AP02 moving observations: {len(moving_rows)}")
    print(f"[INFO] output: {OUT}")

    summary_rows = []

    html = [
        "<html><body>",
        "<h1>AP02 moving-frame stride preview</h1>",
        f"<p>Source CSV: {OBS_CSV}</p>",
        f"<p>Frames with successful moving ArUco observations: {len(frames)}</p>",
    ]

    for stride in STRIDES:
        selected = frames[::stride]
        img_out, csv_out, stats_rows = make_contact_sheet(selected, grouped, stride)

        marker_union = set()
        marker_counts = []
        for r in stats_rows:
            ids = [x for x in r["unique_marker_ids"].split(";") if x != ""]
            marker_union.update(ids)
            marker_counts.append(r["num_marker_observations"])

        mean_markers = sum(marker_counts) / len(marker_counts) if marker_counts else 0.0
        min_markers = min(marker_counts) if marker_counts else 0
        max_markers = max(marker_counts) if marker_counts else 0

        summary_rows.append({
            "stride": stride,
            "selected_frames": len(selected),
            "mean_marker_observations_per_frame": f"{mean_markers:.3f}",
            "min_marker_observations_per_frame": min_markers,
            "max_marker_observations_per_frame": max_markers,
            "unique_markers_seen": len(marker_union),
            "preview_image": str(img_out),
            "selected_frames_csv": str(csv_out),
        })

        print()
        print(f"stride={stride}")
        print(f"  selected_frames={len(selected)}")
        print(f"  unique_markers_seen={len(marker_union)}")
        print(f"  marker_obs_per_frame mean/min/max={mean_markers:.2f}/{min_markers}/{max_markers}")
        print(f"  preview={img_out}")
        print(f"  csv={csv_out}")

        html += [
            f"<h2>stride={stride}</h2>",
            f"<p>selected frames: {len(selected)} | unique markers: {len(marker_union)} | marker obs/frame mean: {mean_markers:.2f}</p>",
            f'<p><a href="{csv_out.name}">selected frames csv</a></p>',
            f'<img src="{img_out.name}" style="max-width: 100%; border: 1px solid #aaa;">',
        ]

    summary_csv = OUT / "moving_stride_preview_summary.csv"
    with summary_csv.open("w", newline="") as fp:
        fields = [
            "stride",
            "selected_frames",
            "mean_marker_observations_per_frame",
            "min_marker_observations_per_frame",
            "max_marker_observations_per_frame",
            "unique_markers_seen",
            "preview_image",
            "selected_frames_csv",
        ]
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    html += ["</body></html>"]
    html_out = OUT / "index.html"
    html_out.write_text("\n".join(html))

    print()
    print("[OK] wrote summary:", summary_csv)
    print("[OK] wrote html:", html_out)


if __name__ == "__main__":
    main()
