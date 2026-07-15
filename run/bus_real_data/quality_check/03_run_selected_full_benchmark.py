#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CASE_ROOT = REPO / "results/bus_real_data/quality_check/benchmark_cases"
OUT_ROOT = REPO / "results/bus_real_data/quality_check/full_approach_benchmark"
SHARED_OBS = REPO / "results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations"

APPROACHES = {
    "AP01": {
        "result_root": REPO / "results/bus_real_data/01_marker_direct_relay_multimarker_multichain",
        "command": ["bash", "run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh"],
        "env": {"RUN_SHARED_BASELINE": "0"},
    },
    "AP02": {
        "result_root": REPO / "results/bus_real_data/02_ref_marker_graph_ba",
        "command": [
            "bash",
            "run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh",
            "--skip-shared-baseline",
        ],
        "env": {},
    },
    "AP03": {
        "result_root": REPO / "results/bus_real_data/03_targetless_colmap_aruco_scale",
        "command": [
            "bash",
            "run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh",
            "--skip-prepare",
            "--skip-colmap",
            "--skip-inspect",
        ],
        "env": {},
    },
}

DEFAULT_CASES = [
    "baseline",
    "area_750",
    "area_1000",
    "area_2000",
    "distance_4m",
    "distance_5m",
    "distance_6m",
    "reprojection_0.2px",
    "reprojection_0.3px",
    "reprojection_0.5px",
    "subsample_0.75_seed_1",
    "subsample_0.50_seed_1",
    "subsample_0.25_seed_1",
    "corner_noise_1px",
    "corner_noise_2px",
    "corner_noise_5px",
    "outliers_0.01_seed_1",
    "outliers_0.05_seed_1",
    "outliers_0.10_seed_1",
    "outliers_0.20_seed_1",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_csv(case_id: str) -> Path:
    candidates = [
        CASE_ROOT / case_id / "AP02/aruco_observations.csv",
        CASE_ROOT / case_id / "AP01/aruco_observations.csv",
        CASE_ROOT / case_id / "AP03/aruco_observations.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No observations found for benchmark case: {case_id}")


def stage_shared_observations(source: Path) -> dict[str, int]:
    rows = read_rows(source)
    static_rows = [row for row in rows if row.get("observer_type") == "static"]
    moving_rows = [row for row in rows if row.get("observer_type") == "moving"]
    if not static_rows or not moving_rows:
        raise RuntimeError(
            f"Case {source} must contain both static and moving observations "
            f"(static={len(static_rows)}, moving={len(moving_rows)})."
        )
    write_rows(SHARED_OBS / "shared_all_aruco_observations.csv", rows)
    write_rows(SHARED_OBS / "shared_static_aruco_observations.csv", static_rows)
    write_rows(SHARED_OBS / "shared_moving_aruco_observations.csv", moving_rows)
    return {"all": len(rows), "static": len(static_rows), "moving": len(moving_rows)}


def copy_tree_if_present(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    if source.exists():
        shutil.copytree(source, destination, symlinks=True)


def run_command(command: list[str], env_extra: dict[str, str], log_path: Path) -> tuple[int, float]:
    env = os.environ.copy()
    env.update(env_extra)
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected AP01/AP02/AP03 benchmark cases safely.")
    parser.add_argument("--case", action="append", dest="cases", help="Benchmark case ID; repeatable.")
    parser.add_argument("--approach", action="append", choices=sorted(APPROACHES), dest="approaches")
    parser.add_argument("--execute", action="store_true", help="Actually execute pipelines. Without this flag, print the plan only.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    cases = args.cases or DEFAULT_CASES
    approaches = args.approaches or list(APPROACHES)
    for case_id in cases:
        case_csv(case_id)

    print(f"Cases: {len(cases)}")
    print(f"Approaches: {', '.join(approaches)}")
    print(f"Runs: {len(cases) * len(approaches)}")
    print(f"Output: {args.out_root}")
    if not args.execute:
        print("Dry run only. Add --execute to start the pipelines.")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="quality-benchmark-backup-") as tmp:
        backup = Path(tmp)
        shared_backup = backup / "shared_observations"
        copy_tree_if_present(SHARED_OBS, shared_backup)
        result_backups: dict[str, Path] = {}
        for approach in approaches:
            root = APPROACHES[approach]["result_root"]
            target = backup / f"result_{approach}"
            copy_tree_if_present(root, target)
            result_backups[approach] = target

        try:
            for case_id in cases:
                counts = stage_shared_observations(case_csv(case_id))
                for approach in approaches:
                    spec = APPROACHES[approach]
                    destination = args.out_root / case_id / approach
                    destination.mkdir(parents=True, exist_ok=True)
                    log_path = destination / "pipeline.log"
                    print(f"[{case_id}] {approach} ...", flush=True)
                    code, runtime = run_command(spec["command"], spec["env"], log_path)
                    copy_tree_if_present(spec["result_root"], destination / "results")
                    status = "success" if code == 0 else "failed"
                    record = {
                        "case_id": case_id,
                        "approach": approach,
                        "status": status,
                        "return_code": code,
                        "runtime_seconds": runtime,
                        "input_observations": counts["all"],
                        "static_observations": counts["static"],
                        "moving_observations": counts["moving"],
                        "result_dir": str(destination / "results"),
                        "log_path": str(log_path),
                    }
                    run_rows.append(record)
                    (destination / "run_metadata.json").write_text(
                        json.dumps(record, indent=2) + "\n", encoding="utf-8"
                    )
                    if code != 0 and not args.continue_on_error:
                        raise RuntimeError(f"{approach} failed for {case_id}; see {log_path}")
        finally:
            copy_tree_if_present(shared_backup, SHARED_OBS)
            for approach in approaches:
                copy_tree_if_present(result_backups[approach], APPROACHES[approach]["result_root"])

    summary = args.out_root / "pipeline_run_summary.csv"
    if run_rows:
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
    print(f"Wrote: {summary}")
    print("Canonical shared observations and approach result directories were restored.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
