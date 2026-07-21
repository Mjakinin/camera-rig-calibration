# Bus Real-Data Camera-Rig Calibration

Canonical execution, evaluation, ablation and reporting root for the bus
real-data camera-rig calibration experiments.

## Project structure

```text
run/bus_real_data/
├── _shared/
│   ├── baseline/
│   ├── common/
│   └── tools/
├── approach1_marker_direct_relay/
├── approach2_ref_marker_graph_ba/
├── approach3_targetless_colmap_aruco_scale/
├── ablation/
│   ├── _shared/
│   ├── moving_cam/
│   │   ├── fov/
│   │   ├── motion_blur/
│   │   └── res/
│   └── world/
│       └── lighting/
├── evaluation/
├── reporting/
│   └── visualization/
├── README.md
└── run_all_and_refresh.sh
```

## Responsibilities

- `_shared/`: method-independent preprocessing and reusable utilities.
- `approach1_marker_direct_relay/`: AP01.
- `approach2_ref_marker_graph_ba/`: AP02.
- `approach3_targetless_colmap_aruco_scale/`: AP03.
- `ablation/`: controlled dataset variants and common orchestration.
- `evaluation/`: numerical metrics.
- `reporting/`: canonical human-readable and machine-readable reports.

Evaluation and reporting remain separate.

## Canonical results

```text
results/bus_real_data/
├── 00_shared_baseline/
├── 01_marker_direct_relay_multimarker_multichain/
├── 02_ref_marker_graph_ba/
├── 03_targetless_colmap_aruco_scale/
├── 99_FINAL_RESULTS_FOR_REPORT/
└── ablation/
    ├── moving_cam/
    └── world/
```

Each of the four principal roots contains one generated
`RESULTS_SUMMARY.txt`. Detailed comparison data belongs in
`99_FINAL_RESULTS_FOR_REPORT/`.

## Metrics

Primary: pairwise static camera-to-camera extrinsic error against ground truth,
without global map alignment.

Secondary: available static-camera map after best-fit rigid SE(3) alignment,
without scale optimization.

AP02 FOV40 has complete numeric coverage but invalid global geometry and must
remain classified as `INVALID_FULL_COVERAGE`.

## Reporting

Validation only:

```bash
bash run/bus_real_data/reporting/run_refresh_final_results.sh --reuse-baseline
```

Promotion:

```bash
bash run/bus_real_data/reporting/run_refresh_final_results.sh   --reuse-baseline --promote
```

Full execution:

```bash
bash run/bus_real_data/run_all_and_refresh.sh --run-methods
```

A clean full run requires Python, COLMAP and the project ROS/Gazebo runtime.
