from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, encode_candidate, parser, prepared_observations
from .contracts import AP01MethodContract, resolve_ap01_method_contract


def construct_candidates(
    *,
    static_rows: list[dict],
    moving_rows: list[dict],
    poses: dict[int, object],
    scale: float,
    camera_ids: tuple[str, ...],
    root_camera: str,
    contract: AP01MethodContract,
) -> tuple[list[dict], list[dict]]:
    """Pure contract-driven AP01 candidate construction boundary."""

    static = core.best_static_by_camera_marker(static_rows)
    registered_frames = set(poses)
    moving = core.moving_by_contract(
        moving_rows, registered_frames, contract
    )
    selected_ids = {id(row) for rows in moving.values() for row in rows}
    selection_rows: list[dict] = []
    registered_by_marker: dict[int, list[dict]] = {}
    for row in moving_rows:
        if int(row["_frame"]) in registered_frames:
            registered_by_marker.setdefault(int(row["_marker"]), []).append(row)
    for marker_id, registered in sorted(registered_by_marker.items()):
        ranked = sorted(
            registered,
            key=lambda row: (-float(row["_quality"]), int(row["_frame"])),
        )
        for rank, row in enumerate(ranked, 1):
            selection_rows.append(
                {
                    "marker_id": marker_id,
                    "frame_id": int(row["_frame"]),
                    "quality_rank": rank,
                    "selection_score": float(row["_quality"]),
                    "selected": id(row) in selected_ids,
                    "registered_observations_for_marker": len(ranked),
                    "selected_observations_for_marker": len(moving.get(marker_id, [])),
                    "top_moving_per_marker": contract.relay_input_limit,
                    "support_policy": contract.moving_support_policy,
                    "tie_breaker": contract.tie_break_policy,
                }
            )

    direct_targets = contract.direct_targets(camera_ids, root_camera)
    relay_targets = contract.relay_targets(camera_ids, root_camera)
    records: list[dict] = []

    def add_direct(target: str) -> None:
        records.extend(core.direct_candidates(root_camera, target, static))

    def add_relay(target: str) -> None:
        records.extend(
            core.relay_candidates(
                root_camera, target, static, moving, poses, scale
            )
        )

    if contract.candidate_construction_order == "all_direct_then_all_relay":
        for target in direct_targets:
            add_direct(target)
        for target in relay_targets:
            add_relay(target)
    elif contract.candidate_construction_order == "per_target_direct_then_relay":
        direct_set = set(direct_targets)
        relay_set = set(relay_targets)
        for target in camera_ids:
            if target == root_camera:
                continue
            if target in direct_set:
                add_direct(target)
            if target in relay_set:
                add_relay(target)
    else:
        raise ValueError(
            "Unknown AP01 candidate order: "
            f"{contract.candidate_construction_order}"
        )
    return records, selection_rows


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
    top_moving_per_marker: int | None,
    method_contract: str = "baseline_v1",
    direct_target_camera: str = "cam_edge_1",
) -> StageResult:
    stage_root = output_root / "03_candidates"

    class Arguments:
        pass

    arguments = Arguments()
    arguments.dataset = dataset
    arguments.observations_root = observations_root
    arguments.out = output_root
    arguments.root_camera = root_camera
    arguments.moving_camera_id = moving_camera_id
    arguments.method_contract = method_contract
    arguments.direct_target_camera = direct_target_camera
    arguments.top_moving_per_marker = top_moving_per_marker
    contract = resolve_ap01_method_contract(
        method_contract,
        direct_target_camera=direct_target_camera,
        top_moving_per_marker=top_moving_per_marker,
    )

    def action() -> dict[str, Path | int]:
        static_rows, moving_rows, poses = prepared_observations(arguments)
        scale = float(
            (output_root / "02_metric_scale/metric_scale.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        raw_records, selection_rows = construct_candidates(
            static_rows=static_rows,
            moving_rows=moving_rows,
            poses=poses,
            scale=scale,
            camera_ids=camera_ids,
            root_camera=root_camera,
            contract=contract,
        )
        stage_root.mkdir(parents=True, exist_ok=True)
        core.write_csv(
            stage_root / "AP01_RELAY_SELECTION.csv",
            selection_rows,
        )
        (stage_root / "AP01_RELAY_SELECTION.json").write_text(
            json.dumps(selection_rows, indent=2) + "\n",
            encoding="utf-8",
        )
        records = [encode_candidate(item) for item in raw_records]
        stage_root.mkdir(parents=True, exist_ok=True)
        path = stage_root / "transform_candidates.json"
        path.write_text(
            json.dumps(records, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return {"candidates": path, "candidate_count": len(records)}

    return run_stage(
        "ap01.build_candidates",
        stage_root,
        action,
        inputs={
            "observations": observations_root,
            "colmap": output_root / "01_moving_colmap",
            "scale": output_root / "02_metric_scale/metric_scale.txt",
        },
        parameters={
            "method_contract": contract.fingerprint_payload(),
            "method_contract_sha256": contract.scientific_fingerprint(),
        },
    )


def main() -> None:
    args = parser(__doc__ or "Build AP01 transform candidates").parse_args()
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=cameras(args),
        root_camera=args.root_camera,
        moving_camera_id=args.moving_camera_id,
        top_moving_per_marker=args.top_moving_per_marker,
        method_contract=args.method_contract,
        direct_target_camera=args.direct_target_camera,
    )


if __name__ == "__main__":
    main()
