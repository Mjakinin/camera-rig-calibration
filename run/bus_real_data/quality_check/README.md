# Cross-Approach Marker Observation Quality Check

This directory defines one shared set of marker-observation quality profiles for
Approach 1, Approach 2 and Approach 3.

The purpose is to prevent an unfair comparison where each approach receives a
different subset of ArUco observations.

## Step 1: update the branch

```bash
git switch quality-check
git pull origin quality-check
```

## Step 2: prepare identical filtered inputs

```bash
python run/bus_real_data/quality_check/01_prepare_cross_approach_quality_inputs.py
```

The default source is:

```text
results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/ap02_all_aruco_observations.csv
```

The generated files are written under:

```text
results/bus_real_data/quality_check/cross_approach_inputs/<profile>/<approach>/aruco_observations.csv
```

For a given profile, the AP01, AP02 and AP03 CSV files contain the same accepted
rows. The approach name is retained in the directory structure so each pipeline
can consume its own isolated input path without mutating another run.

The main comparison file is:

```text
results/bus_real_data/quality_check/cross_approach_inputs/cross_approach_quality_input_summary.csv
```

## Included profiles

The profile definitions live in `quality_profiles.json`.

They include:

- baseline
- marker-area thresholds at 500, 1000 and 2000 px²
- maximum distances of 3, 4 and 5 m
- maximum reprojection RMSE values of 0.2, 0.3 and 0.5 px
- loose, medium and strict combined filters

Run only selected profiles with repeated `--profile` arguments:

```bash
python run/bus_real_data/quality_check/01_prepare_cross_approach_quality_inputs.py \
  --profile baseline \
  --profile combined_medium \
  --profile combined_strict
```

## Interpretation

The preparation summary measures input retention and reference-marker graph
connectivity. It does not by itself determine which calibration approach is
best.

After each generated CSV has been consumed by AP01, AP02 and AP03, append the
approach-specific final metrics to one result table with at least:

```text
approach
filter_profile
accepted_observations
retention_fraction
connected_markers
connected_observers
translation_error_m
rotation_error_deg
reprojection_rmse_px
success
```

The preferred profile is not necessarily the strictest one. It should reduce
final calibration error while preserving the marker/camera graph and maintaining
a high pipeline success rate.

## Current adapter status

This commit provides the shared profiles, identical filtered inputs and the
cross-approach input summary. The three existing pipelines currently have
approach-specific input wiring. Use the generated `aruco_observations.csv` as
the observation source when running each pipeline variant; do not overwrite the
baseline source file.
