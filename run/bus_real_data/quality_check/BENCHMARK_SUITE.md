# Cross-Approach Quality Benchmark Suite

This suite creates reproducible marker-observation test cases for AP01, AP02 and AP03.

## Implemented experiment families

- marker-area threshold sweep
- maximum-distance threshold sweep
- maximum PnP reprojection-RMSE sweep
- random observation subsampling with repeated seeded trials
- leave-one-marker-out
- reference-marker sensitivity inputs
- Gaussian marker-corner noise injection
- sparse gross corner-outlier injection
- retention and reference-component graph metrics
- identical input CSVs for AP01, AP02 and AP03

## Generate all cases

```bash
python run/bus_real_data/quality_check/02_generate_quality_benchmark_cases.py
```

Outputs:

```text
results/bus_real_data/quality_check/benchmark_cases/
  benchmark_case_summary.csv
  benchmark_manifest.json
  <case_id>/AP01/aruco_observations.csv
  <case_id>/AP02/aruco_observations.csv
  <case_id>/AP03/aruco_observations.csv
```

The matrix is configured in `benchmark_matrix.json`. The random seed is fixed for reproducibility.

## Required approach adapter contract

Each full approach runner should accept:

```text
--observations-csv <case>/<approach>/aruco_observations.csv
--out-root <case>/<approach>/results
```

The approach must write a machine-readable `metrics.json` containing as many of these fields as applicable:

```json
{
  "approach": "AP02",
  "success": true,
  "failure_reason": "",
  "translation_error_m": 0.0,
  "rotation_error_deg": 0.0,
  "reprojection_rmse_px": 0.0,
  "scale_error_percent": 0.0,
  "cycle_translation_error_m": 0.0,
  "cycle_rotation_error_deg": 0.0,
  "runtime_seconds": 0.0,
  "peak_memory_mb": 0.0
}
```

## Interpretation priorities

1. Reject profiles that disconnect required cameras or markers.
2. Compare success rate across repeated trials, not only successful-run error.
3. Prefer median and p90 translation/rotation errors over a single run.
4. For AP02, report pre-BA and post-BA metrics separately.
5. For AP03, report scale median, MAD, inlier count and scale error.
6. Report worst-case performance across route, lighting, blur, resolution, density and FOV datasets.

## Current integration boundary

Observation-case generation is complete. Existing AP01/AP02/AP03 scripts still need a consistent `--observations-csv` override before the suite can launch every full calibration automatically. Until those adapters are added, copy or point each case CSV to the input expected by the selected pipeline and keep results under the corresponding case directory.

Do not commit generated case directories unless they are intentionally selected report artifacts.
