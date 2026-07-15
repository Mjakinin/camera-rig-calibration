# Marker Observation Quality Ablation

This experiment separates the marker-observation quality criteria that were previously combined inside `observation_score()`.

## Tested variants

- `baseline_success`: every successful PnP observation
- `marker_area_only`: minimum projected marker area
- `distance_only`: maximum camera-to-marker distance
- `reprojection_only`: maximum PnP reprojection RMSE
- `marker_area_distance`
- `marker_area_reprojection`
- `distance_reprojection`
- `all_hard_filters`
- `weighted_score`: existing combined observation score

## Run

From the repository root:

```bash
python run/bus_real_data/approach2_ref_marker_graph_ba/10_run_observation_quality_ablation.py
```

Explicit thresholds:

```bash
python run/bus_real_data/approach2_ref_marker_graph_ba/10_run_observation_quality_ablation.py \
  --min-area-px2 64 \
  --max-distance-m 12 \
  --max-reprojection-rmse-px 4 \
  --min-weighted-score 0.01
```

A subset can be run with:

```bash
python run/bus_real_data/approach2_ref_marker_graph_ba/10_run_observation_quality_ablation.py \
  --variants baseline_success marker_area_only distance_only reprojection_only all_hard_filters
```

## Outputs

The default output directory is:

```text
results/bus_real_data/02_ref_marker_graph_ba/10_observation_quality_ablation/
```

Top-level files:

- `quality_ablation_summary.csv`: comparison across all variants
- `quality_ablation_manifest.json`: thresholds, variants and rejection counts

Each variant contains:

- `accepted_observations.csv`: observations retained by that filter
- `observation_decisions.csv`: every observation with pass/fail flags
- `summary.json`: retention, quality distributions and graph connectivity

## Primary evaluation columns

- `retention_fraction_of_successful`
- `unique_markers`
- `unique_static_observers`
- `unique_moving_observers`
- `reference_component_markers`
- `reference_component_observers`
- `reference_component_edges`
- median and percentile values for marker area, distance and reprojection RMSE

A useful filter should remove low-quality observations without disconnecting markers or observers from the reference-marker component. This observation-level ablation should be followed by rerunning graph initialization and bundle adjustment for the strongest candidate variants, then comparing final translation, rotation and BA reprojection errors.
