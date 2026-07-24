from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict, deque
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from .common import read_csv


def _reachability(
    rows: list[dict[str, str]], reference_marker: int, *, combined: bool
) -> tuple[set[str], set[int]]:
    marker_observers: dict[int, set[str]] = defaultdict(set)
    observer_markers: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if not combined and row.get("observer_type") != "static":
            continue
        marker = int(float(row["marker_id"]))
        observer = str(row["observer_id"])
        marker_observers[marker].add(observer)
        observer_markers[observer].add(marker)
    observers: set[str] = set()
    markers = {reference_marker}
    queue: deque[tuple[str, str | int]] = deque(
        [("marker", reference_marker)]
    )
    while queue:
        kind, value = queue.popleft()
        if kind == "marker":
            for observer in sorted(marker_observers.get(int(value), set())):
                if observer not in observers:
                    observers.add(observer)
                    queue.append(("observer", observer))
        else:
            for marker in sorted(observer_markers.get(str(value), set())):
                if marker not in markers:
                    markers.add(marker)
                    queue.append(("marker", marker))
    return observers, markers


def run(
    *,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    reference_marker_id: int,
) -> StageResult:
    stage_root = output_root / "02_aruco_observations"
    source = observations_root / "shared_all_aruco_observations.csv"

    def action() -> dict[str, Path | int]:
        stage_root.mkdir(parents=True, exist_ok=True)
        mapping = {
            "shared_static_aruco_observations.csv":
                "ap02_static_aruco_observations.csv",
            "shared_moving_aruco_observations.csv":
                "ap02_moving_aruco_observations.csv",
            "shared_all_aruco_observations.csv":
                "ap02_all_aruco_observations.csv",
        }
        for source_name, destination_name in mapping.items():
            shutil.copy2(
                observations_root / source_name,
                stage_root / destination_name,
            )
        rows = read_csv(source)
        static_observers, static_markers = _reachability(
            rows, reference_marker_id, combined=False
        )
        combined_observers, combined_markers = _reachability(
            rows, reference_marker_id, combined=True
        )
        static_cameras = sorted(set(camera_ids).intersection(static_observers))
        combined_cameras = sorted(
            set(camera_ids).intersection(combined_observers)
        )
        summary = stage_root / "graph_summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "reference_marker_id": reference_marker_id,
                    "static_only": {
                        "reached_cameras": static_cameras,
                        "reached_camera_count": len(static_cameras),
                        "reached_marker_count": len(static_markers),
                    },
                    "combined": {
                        "reached_cameras": combined_cameras,
                        "reached_camera_count": len(combined_cameras),
                        "reached_marker_count": len(combined_markers),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "observations": stage_root
            / "ap02_all_aruco_observations.csv",
            "graph_summary": summary,
            "accepted_observations": len(rows),
        }

    return run_stage(
        "ap02.build_graph",
        stage_root,
        action,
        inputs={"quality_accepted_observations": observations_root},
        parameters={
            "reference_marker_id": reference_marker_id,
            "weighted_graph": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--ref-marker-id", type=int, required=True)
    args = parser.parse_args()
    run(
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=tuple(
            item.strip() for item in args.cameras.split(",") if item.strip()
        ),
        reference_marker_id=args.ref_marker_id,
    )


if __name__ == "__main__":
    main()
