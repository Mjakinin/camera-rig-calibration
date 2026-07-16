#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "results/bus_real_data/quality_check/full_approach_benchmark"
EVALUATOR_PATH = REPO / "run/bus_real_data/evaluation/10_eval_pairwise_static_camera_extrinsics.py"
POSE_REL = Path("AP03/results/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("pairwise_eval", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(case_id: str, rows: list[dict], source: Path) -> dict:
    ok = [row for row in rows if row.get("status") == "OK"]
    if not ok:
        return {
            "case_id": case_id,
            "status": "FAILED",
            "pair_count": 0,
            "translation_error_m_mean": "",
            "translation_error_m_median": "",
            "translation_error_m_max": "",
            "rotation_error_deg_mean": "",
            "rotation_error_deg_median": "",
            "rotation_error_deg_max": "",
            "source": str(source),
        }
    ts_m = [float(row["translation_error_cm"]) * 0.01 for row in ok]
    rs_deg = [float(row["rotation_error_deg"]) for row in ok]
    return {
        "case_id": case_id,
        "status": "OK",
        "pair_count": len(ok),
        "translation_error_m_mean": mean(ts_m),
        "translation_error_m_median": median(ts_m),
        "translation_error_m_max": max(ts_m),
        "rotation_error_deg_mean": mean(rs_deg),
        "rotation_error_deg_median": median(rs_deg),
        "rotation_error_deg_max": max(rs_deg),
        "source": str(source),
    }


def discover_cases(root: Path, selected: set[str]):
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected and case_dir.name not in selected:
            continue
        pose_source = case_dir / POSE_REL
        if pose_source.is_file():
            yield case_dir.name, pose_source


def main():
    parser = argparse.ArgumentParser(description="Evaluate AP03 benchmark snapshots against GT using pairwise static-camera extrinsics.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case", action="append", dest="cases", help="Optional benchmark case ID; repeatable.")
    args = parser.parse_args()

    evaluator = load_evaluator()
    gt, _ = evaluator.load_gt_camera_map()
    selected = set(args.cases or [])
    summaries = []

    for case_id, pose_source in discover_cases(args.root, selected):
        out_dir = pose_source.parent
        detail_path = out_dir / "AP03_GT_PAIRWISE_POSE_ERRORS.csv"
        summary_path = out_dir / "AP03_GT_PAIRWISE_POSE_SUMMARY.csv"

        try:
            poses, meta = evaluator.load_pose_csv_camera_map(pose_source, "AP03")
            rows = evaluator.eval_method_pairwise(
                "AP03", poses, gt, meta.get("source", ""),
                "GT used only for post-hoc evaluation.",
            )
            summary = summarize(case_id, rows, pose_source)
        except Exception as exc:
            rows = evaluator.failed_rows("AP03", str(exc))
            summary = summarize(case_id, rows, pose_source)
            summary["error"] = str(exc)

        detail_fields = list(rows[0].keys()) if rows else []
        write_csv(detail_path, rows, detail_fields)
        write_csv(summary_path, [summary], list(summary.keys()))
        summaries.append(summary)
        print(f"{case_id}: {summary['status']} T={summary['translation_error_m_mean']} R={summary['rotation_error_deg_mean']}")

    aggregate = args.root / "ap03_pose_benchmark_summary.csv"
    if summaries:
        fields = []
        for row in summaries:
            for key in row:
                if key not in fields:
                    fields.append(key)
        write_csv(aggregate, summaries, fields)
    else:
        print(f"No AP03 pose snapshots found under: {args.root}")
    print(f"Wrote: {aggregate}")


if __name__ == "__main__":
    main()
