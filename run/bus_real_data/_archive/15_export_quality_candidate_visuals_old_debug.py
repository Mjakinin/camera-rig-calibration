#!/usr/bin/env python3

import argparse
import csv
import html
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(".").resolve()

DEFAULT_CANDIDATE_DIR = Path("results/bus_real_data/05_quality_chain_candidate_selection")
DEFAULT_OUT_DIR = DEFAULT_CANDIDATE_DIR / "visual_check"


def resolve_path(p):
    if p is None or str(p).strip() == "":
        return None

    path = Path(str(p).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def safe_copy(src, dst):
    if src is None or not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_image(path, fallback_text):
    if path is None or not path.exists():
        img = np.full((360, 640, 3), 245, dtype=np.uint8)
        cv2.putText(
            img,
            "MISSING",
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            fallback_text[:60],
            (40, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return img

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        img = np.full((360, 640, 3), 245, dtype=np.uint8)
        cv2.putText(
            img,
            "COULD NOT READ IMAGE",
            (40, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return img


def resize_keep_aspect(img, target_w=640, target_h=360):
    h, w = img.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    return canvas


def draw_header(img, lines):
    header_h = 95
    h, w = img.shape[:2]

    canvas = np.full((h + header_h, w, 3), 255, dtype=np.uint8)
    canvas[header_h:, :] = img

    y = 28
    for i, line in enumerate(lines):
        font_scale = 0.65 if i == 0 else 0.52
        thickness = 2 if i == 0 else 1
        cv2.putText(
            canvas,
            line[:110],
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y += 28

    return canvas


def make_contact_sheet(row, out_path):
    root_img = read_image(resolve_path(row.get("root_image")), "root_image")
    root_dbg = read_image(resolve_path(row.get("root_debug_image")), "root_debug_image")
    tgt_img = read_image(resolve_path(row.get("target_image")), "target_image")
    tgt_dbg = read_image(resolve_path(row.get("target_debug_image")), "target_debug_image")

    root_img = resize_keep_aspect(root_img)
    root_dbg = resize_keep_aspect(root_dbg)
    tgt_img = resize_keep_aspect(tgt_img)
    tgt_dbg = resize_keep_aspect(tgt_dbg)

    top_left = draw_header(root_img, [
        "ROOT normal image",
        f"{row.get('root_camera')} marker {row.get('root_marker')} frame {row.get('root_frame')}",
        f"ids: {row.get('root_frame_ids')} | obsQ {row.get('root_moving_obs_quality')} | frameQ {row.get('root_frame_quality')}",
    ])

    top_right = draw_header(tgt_img, [
        "TARGET normal image",
        f"{row.get('target_camera')} marker {row.get('target_marker')} frame {row.get('target_frame')}",
        f"ids: {row.get('target_frame_ids')} | obsQ {row.get('target_moving_obs_quality')} | frameQ {row.get('target_frame_quality')}",
    ])

    bottom_left = draw_header(root_dbg, [
        "ROOT debug image",
        f"COLMAP reg: {row.get('root_colmap_registered')} | obs3D: {row.get('root_colmap_obs_3d')}",
        f"staticQ {row.get('root_static_quality')} | colmapQ {row.get('root_colmap_quality')}",
    ])

    bottom_right = draw_header(tgt_dbg, [
        "TARGET debug image",
        f"COLMAP reg: {row.get('target_colmap_registered')} | obs3D: {row.get('target_colmap_obs_3d')}",
        f"staticQ {row.get('target_static_quality')} | colmapQ {row.get('target_colmap_quality')}",
    ])

    row1 = np.hstack([top_left, top_right])
    row2 = np.hstack([bottom_left, bottom_right])

    title_h = 100
    full = np.full((title_h + row1.shape[0] + row2.shape[0], row1.shape[1], 3), 255, dtype=np.uint8)

    title_lines = [
        f"QUALITY CHAIN CANDIDATE | score={row.get('chain_quality_score')} | gap={row.get('frame_gap_signed')}",
        f"{row.get('root_camera')} -> moving frame {row.get('root_frame')} -> moving frame {row.get('target_frame')} -> {row.get('target_camera')}",
        f"root marker {row.get('root_marker')} | target marker {row.get('target_marker')}",
    ]

    y = 30
    for i, line in enumerate(title_lines):
        cv2.putText(
            full,
            line[:150],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75 if i == 0 else 0.62,
            (0, 0, 0),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )
        y += 30

    full[title_h:title_h + row1.shape[0], :] = row1
    full[title_h + row1.shape[0]:, :] = row2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full)


def export_candidate_csv(csv_path, out_root, top_n):
    rows = []

    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("[WARN] no rows:", csv_path)
        return []

    target_name = csv_path.stem.replace("07_relay_candidates_", "").replace("_colmap_registered_quality", "")
    target_out = out_root / target_name
    target_out.mkdir(parents=True, exist_ok=True)

    exported = []

    for idx, row in enumerate(rows[:top_n], start=1):
        cand_name = (
            f"rank_{idx:03d}"
            f"__score_{row.get('chain_quality_score', 'NA')}"
            f"__rootF_{int(row.get('root_frame')):04d}_m{int(row.get('root_marker')):02d}"
            f"__targetF_{int(row.get('target_frame')):04d}_m{int(row.get('target_marker')):02d}"
        )

        cand_dir = target_out / cand_name
        cand_dir.mkdir(parents=True, exist_ok=True)

        files_to_copy = [
            ("root_image", "root_image.png"),
            ("root_debug_image", "root_debug.png"),
            ("target_image", "target_image.png"),
            ("target_debug_image", "target_debug.png"),
        ]

        for key, filename in files_to_copy:
            src = resolve_path(row.get(key))
            safe_copy(src, cand_dir / filename)

        with (cand_dir / "candidate_info.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        with (cand_dir / "candidate_info.txt").open("w") as f:
            f.write("Quality chain candidate\n")
            f.write("=======================\n\n")
            for k, v in row.items():
                f.write(f"{k}: {v}\n")

        contact_path = cand_dir / "contact_sheet.png"
        make_contact_sheet(row, contact_path)

        exported.append({
            "rank": idx,
            "target_name": target_name,
            "candidate_dir": str(cand_dir),
            "contact_sheet": str(contact_path),
            "score": row.get("chain_quality_score", ""),
            "root_frame": row.get("root_frame", ""),
            "root_marker": row.get("root_marker", ""),
            "target_frame": row.get("target_frame", ""),
            "target_marker": row.get("target_marker", ""),
            "frame_gap_signed": row.get("frame_gap_signed", ""),
            "root_frame_ids": row.get("root_frame_ids", ""),
            "target_frame_ids": row.get("target_frame_ids", ""),
        })

    print(f"[OK] exported top {min(top_n, len(rows))}: {csv_path.name} -> {target_out}")
    return exported


def write_index_html(out_root, exported):
    html_path = out_root / "index.html"

    parts = []
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<title>Quality Chain Candidate Visual Check</title>")
    parts.append("""
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
h1, h2 { margin-top: 28px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
th, td { border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; }
th { background: #eee; }
img { max-width: 900px; border: 1px solid #aaa; margin: 8px 0 24px 0; }
.small { color: #555; font-size: 12px; }
</style>
""")
    parts.append("</head><body>")
    parts.append("<h1>Quality Chain Candidate Visual Check</h1>")
    parts.append("<p>Diese Auswahl nutzt keine GT-Errors. Die Kandidaten sind nach real messbaren Qualitätsmetriken sortiert.</p>")

    by_target = {}
    for e in exported:
        by_target.setdefault(e["target_name"], []).append(e)

    for target, rows in by_target.items():
        parts.append(f"<h2>{html.escape(target)}</h2>")
        parts.append("<table>")
        parts.append(
            "<tr>"
            "<th>rank</th><th>score</th><th>root frame</th><th>root marker</th>"
            "<th>target frame</th><th>target marker</th><th>gap</th>"
            "<th>root ids</th><th>target ids</th><th>folder</th>"
            "</tr>"
        )

        for e in rows:
            rel_dir = Path(e["candidate_dir"]).relative_to(out_root)
            rel_img = Path(e["contact_sheet"]).relative_to(out_root)

            parts.append(
                "<tr>"
                f"<td>{e['rank']}</td>"
                f"<td>{html.escape(str(e['score']))}</td>"
                f"<td>{html.escape(str(e['root_frame']))}</td>"
                f"<td>{html.escape(str(e['root_marker']))}</td>"
                f"<td>{html.escape(str(e['target_frame']))}</td>"
                f"<td>{html.escape(str(e['target_marker']))}</td>"
                f"<td>{html.escape(str(e['frame_gap_signed']))}</td>"
                f"<td>{html.escape(str(e['root_frame_ids']))}</td>"
                f"<td>{html.escape(str(e['target_frame_ids']))}</td>"
                f"<td><a href='{html.escape(str(rel_dir))}/candidate_info.txt'>info</a></td>"
                "</tr>"
            )

            parts.append(
                f"<tr><td colspan='10'>"
                f"<a href='{html.escape(str(rel_img))}'><img src='{html.escape(str(rel_img))}'></a>"
                f"</td></tr>"
            )

        parts.append("</table>")

    parts.append("</body></html>")

    html_path.write_text("\n".join(parts), encoding="utf-8")
    print("[OK] wrote:", html_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    candidate_dir = Path(args.candidate_dir)
    out_root = Path(args.out)

    if not candidate_dir.exists():
        raise RuntimeError(f"Candidate dir not found: {candidate_dir}")

    csvs = sorted(candidate_dir.glob("07_relay_candidates_*_colmap_registered_quality.csv"))

    if not csvs:
        raise RuntimeError(
            "No colmap-registered candidate CSVs found. "
            "Run 14b_select_chain_candidates_by_quality.py first."
        )

    if out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    all_exported = []

    for csv_path in csvs:
        all_exported.extend(export_candidate_csv(csv_path, out_root, args.top_n))

    summary_csv = out_root / "exported_candidates_summary.csv"
    if all_exported:
        with summary_csv.open("w", newline="") as f:
            fields = list(all_exported[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_exported)

    write_index_html(out_root, all_exported)

    print()
    print("[DONE] visual export finished")
    print("Output folder:", out_root)
    print("Open this HTML:")
    print(out_root / "index.html")


if __name__ == "__main__":
    main()
