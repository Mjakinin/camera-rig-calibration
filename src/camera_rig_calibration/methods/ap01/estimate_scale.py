from __future__ import annotations

import json
import shutil
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, parser, prepared_observations
from .contracts import resolve_ap01_method_contract
from .frozen_intermediate import validate_frozen_intermediate


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
    scale_top_per_marker: int | None,
    method_contract: str = "baseline_v1",
    direct_target_camera: str = "cam_edge_1",
    top_moving_per_marker: int | None = 8,
) -> StageResult:
    stage_root = output_root / "02_metric_scale"

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
    arguments.scale_top_per_marker = scale_top_per_marker
    contract = resolve_ap01_method_contract(
        method_contract,
        direct_target_camera=direct_target_camera,
        top_moving_per_marker=top_moving_per_marker,
        scale_top_per_marker=scale_top_per_marker,
    )

    def action() -> dict[str, Path | float | int]:
        stage_root.mkdir(parents=True, exist_ok=True)
        scale_file = stage_root / "metric_scale.txt"
        pairs: list[dict] | None
        if contract.scale_execution_policy == "frozen_historical_sfm_gauge_scale":
            frozen = validate_frozen_intermediate(
                dataset=dataset,
                moving_camera_id=moving_camera_id,
                contract=contract,
            )
            shutil.copy2(frozen.metric_scale, scale_file)
            scale = float(scale_file.read_text(encoding="utf-8").strip())
            statistics = {
                "scale_m_per_colmap_unit": scale,
                "raw_pairs": 1869,
                "used_pairs": 1617,
                "sfm_mode": "frozen_historical_reproduction",
                "scale_mode": "frozen_historical_sfm_gauge_scale",
                "ground_truth_used": False,
                "source_manifest": str(frozen.manifest),
                "source_metric_scale_sha256": (
                    contract.scale_frozen_metric_sha256
                ),
                "method_contract": contract.fingerprint_payload(),
                "method_contract_sha256": contract.scientific_fingerprint(),
            }
            pairs = None
        elif contract.scale_execution_policy == "fresh_metric_scale_estimation":
            _, moving_rows, colmap_poses = prepared_observations(arguments)
            scale, statistics, pairs = core.robust_scale(
                moving_rows,
                colmap_poses,
                maximum_observations_per_marker=scale_top_per_marker,
                contract=contract,
            )
            scale_file.write_text(f"{scale:.12g}\n", encoding="utf-8")
        else:
            raise ValueError(
                "Unknown AP01 scale execution policy: "
                f"{contract.scale_execution_policy}"
            )
        diagnostics = stage_root / "SCALE_DIAGNOSTICS.json"
        diagnostics.write_text(
            json.dumps(statistics, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs: dict[str, Path | float | int] = {
            "metric_scale": scale_file,
            "diagnostics": diagnostics,
            "used_observation_pairs": int(statistics["used_pairs"]),
        }
        if pairs is not None:
            pairs_file = stage_root / "scale_pairs.csv"
            core.write_csv(pairs_file, pairs)
            outputs["pairs"] = pairs_file
        return outputs

    return run_stage(
        "ap01.estimate_scale",
        stage_root,
        action,
        inputs={
            "observations": observations_root,
            "colmap": output_root / "01_moving_colmap",
        },
        parameters={
            "selection": contract.scale_observation_construction_policy,
            "scale_mode": contract.scale_execution_policy,
            "scale_top_per_marker": (
                contract.scale_observation_limit_per_marker
            ),
            "method_contract": contract.fingerprint_payload(),
            "method_contract_sha256": contract.scientific_fingerprint(),
        },
    )


def main() -> None:
    args = parser(__doc__ or "Estimate AP01 metric scale").parse_args()
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=cameras(args),
        root_camera=args.root_camera,
        moving_camera_id=args.moving_camera_id,
        scale_top_per_marker=args.scale_top_per_marker,
        method_contract=args.method_contract,
        direct_target_camera=args.direct_target_camera,
        top_moving_per_marker=args.top_moving_per_marker,
    )


if __name__ == "__main__":
    main()
