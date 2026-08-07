from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from camera_rig_calibration.ap02_graph import (
    graph_components,
    rows_for_component,
)
from camera_rig_calibration.pipeline import StageResult, run_stage

from .common import read_csv, write_csv
from .frame_selection import (
    AP02FrameSelection,
    legacy_frame_number,
    select_ap02_frames,
    write_ap02_frame_selection,
)
from .initialize import main_observation_score
from .frozen_observations import resolve_ap02_observation_input


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
    dataset_root: Path | None = None,
    output_root: Path,
    camera_ids: tuple[str, ...],
    reference_marker_id: int,
    reference_marker_maximum_frames: int | None,
    top_per_marker: int | None,
    top_per_marker_pair: int | None,
    maximum_total_frames: int | None,
    graph_observation_policy: str = (
        "wizard_graph_preserving_preselection_v1"
    ),
    method_contract: dict[str, object] | None = None,
    method_contract_sha256: str | None = None,
    historical_reproduction: bool = False,
) -> StageResult:
    stage_root = output_root / "02_aruco_observations"
    contract_name = str((method_contract or {}).get("name", "baseline_v1"))
    contract_sha256 = str(method_contract_sha256 or "")
    source, frozen = resolve_ap02_observation_input(
        observations_root=observations_root,
        dataset=dataset_root,
        historical_reproduction=historical_reproduction,
        method_contract_name=contract_name,
        method_contract_sha256=contract_sha256,
        reference_marker_policy=str(
            (method_contract or {}).get("reference_marker_policy", "baseline")
        ),
        reference_marker_id=reference_marker_id,
        root_pose_policy=str(
            (method_contract or {}).get(
                "root_pose_policy", "reference_marker_identity_v1"
            )
        ),
    )
    frame_selection_limits = {
        "reference_marker_maximum_frames": (
            reference_marker_maximum_frames
        ),
        "top_per_marker": top_per_marker,
        "top_per_marker_pair": top_per_marker_pair,
        "maximum_total_frames": maximum_total_frames,
    }

    def action() -> dict[str, Path | int]:
        stage_root.mkdir(parents=True, exist_ok=True)
        if frozen is not None:
            provenance_path = (
                stage_root / "HISTORICAL_REPRODUCTION_PROVENANCE.json"
            )
            provenance_path.write_text(
                json.dumps(frozen.provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        accepted_rows = read_csv(source)
        if graph_observation_policy == (
            "legacy_quality_valid_all_observations_v1"
        ):
            rows = [
                row for row in accepted_rows if main_observation_score(row) > 0.0
            ]
            moving_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                if row.get("observer_type") == "moving":
                    moving_by_frame[str(row["observer_id"])].append(row)
            frame_ids = tuple(
                sorted(moving_by_frame, key=legacy_frame_number)
            )
            frame_selection = AP02FrameSelection(
                selected_rows=tuple(rows),
                selected_frame_ids=frame_ids,
                diagnostics=tuple(
                    {
                        "frame_id": frame_id,
                        "selected": True,
                        "selection_reasons": ["legacy_quality_valid_graph"],
                        "marker_ids": sorted(
                            int(float(row["marker_id"]))
                            for row in moving_by_frame[frame_id]
                        ),
                        "marker_count": len(moving_by_frame[frame_id]),
                        "selection_score": sum(
                            main_observation_score(row)
                            for row in moving_by_frame[frame_id]
                        ),
                        "score_area": "",
                        "score_reprojection": "",
                        "score_border": "",
                        "score_distance": "",
                        "tie_breaker": "legacy input order",
                    }
                    for frame_id in frame_ids
                ),
                summary={
                    "schema_version": 1,
                    "strategy": graph_observation_policy,
                    "input_observations": len(accepted_rows),
                    "quality_valid_observations": len(rows),
                    "selected_moving_frames": len(frame_ids),
                    "selected_frame_ids": list(frame_ids),
                    "limits": frame_selection_limits,
                },
            )
        elif graph_observation_policy == (
            "wizard_graph_preserving_preselection_v1"
        ):
            frame_selection = select_ap02_frames(
                accepted_rows,
                camera_ids=camera_ids,
                reference_marker_id=reference_marker_id,
                **frame_selection_limits,
            )
            rows = list(frame_selection.selected_rows)
        else:
            raise ValueError(
                f"Unknown AP02 graph observation policy: {graph_observation_policy}"
            )
        write_ap02_frame_selection(frame_selection, stage_root)
        write_csv(
            stage_root / "ap02_all_aruco_observations.csv",
            rows,
            list(rows[0]) if rows else [],
        )
        write_csv(
            stage_root / "ap02_static_aruco_observations.csv",
            [
                row
                for row in rows
                if row.get("observer_type") == "static"
            ],
            list(rows[0]) if rows else [],
        )
        write_csv(
            stage_root / "ap02_moving_aruco_observations.csv",
            [
                row
                for row in rows
                if row.get("observer_type") == "moving"
            ],
            list(rows[0]) if rows else [],
        )
        components = graph_components(rows, camera_ids)
        primary_component = next(
            (
                component
                for component in components
                if reference_marker_id in component.marker_ids
            ),
            None,
        )
        fields = list(rows[0]) if rows else []
        components_root = stage_root / "components"
        for component in components:
            component_root = components_root / component.component_id
            component_rows = rows_for_component(rows, component)
            write_csv(
                component_root / "ap02_all_aruco_observations.csv",
                component_rows,
                fields,
            )
            write_csv(
                component_root / "ap02_static_aruco_observations.csv",
                [
                    row
                    for row in component_rows
                    if row.get("observer_type") == "static"
                ],
                fields,
            )
            write_csv(
                component_root / "ap02_moving_aruco_observations.csv",
                [
                    row
                    for row in component_rows
                    if row.get("observer_type") == "moving"
                ],
                fields,
            )
        component_manifest = stage_root / "component_manifest.json"
        component_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "reference_marker_id": reference_marker_id,
                    "primary_component_id": (
                        primary_component.component_id
                        if primary_component is not None
                        else None
                    ),
                    "expected_static_cameras": list(camera_ids),
                    "components": [
                        component.model_dump() for component in components
                    ],
                    "not_observable_between_components": (
                        len(components) > 1
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
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
                    "component_manifest": str(component_manifest),
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
            "component_manifest": component_manifest,
            "accepted_observations": len(rows),
            "selected_moving_frames": len(
                frame_selection.selected_frame_ids
            ),
            **(
                {"historical_reproduction_provenance": provenance_path}
                if frozen is not None
                else {}
            ),
        }

    return run_stage(
        "ap02.build_graph",
        stage_root,
        action,
        inputs={"quality_accepted_observations": source},
        parameters={
            "reference_marker_id": reference_marker_id,
            "weighted_graph": False,
            "frame_selection": frame_selection_limits,
            "graph_observation_policy": graph_observation_policy,
            "method_contract": method_contract,
            "method_contract_sha256": method_contract_sha256,
            "historical_reproduction": historical_reproduction,
            "observation_input_mode": (
                "frozen_historical_ap02_reproduction"
                if frozen is not None
                else "fresh_dataset_derived_observations"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--ref-marker-id", type=int, required=True)
    parser.add_argument("--reference-marker-maximum-frames", type=int)
    parser.add_argument("--top-per-marker", type=int)
    parser.add_argument("--top-per-marker-pair", type=int)
    parser.add_argument("--maximum-total-frames", type=int)
    parser.add_argument(
        "--graph-observation-policy",
        default="legacy_quality_valid_all_observations_v1",
    )
    parser.add_argument("--method-contract-sha256")
    parser.add_argument("--historical-reproduction", action="store_true")
    args = parser.parse_args()
    run(
        observations_root=args.observations_root.resolve(),
        dataset_root=(args.dataset.resolve() if args.dataset else None),
        output_root=args.out.resolve(),
        camera_ids=tuple(
            item.strip() for item in args.cameras.split(",") if item.strip()
        ),
        reference_marker_id=args.ref_marker_id,
        reference_marker_maximum_frames=(
            args.reference_marker_maximum_frames
        ),
        top_per_marker=args.top_per_marker,
        top_per_marker_pair=args.top_per_marker_pair,
        maximum_total_frames=args.maximum_total_frames,
        graph_observation_policy=args.graph_observation_policy,
        method_contract={
            "name": "baseline_v1",
            "reference_marker_policy": "baseline",
            "root_pose_policy": "reference_marker_identity_v1",
        },
        method_contract_sha256=args.method_contract_sha256,
        historical_reproduction=args.historical_reproduction,
    )


if __name__ == "__main__":
    main()
