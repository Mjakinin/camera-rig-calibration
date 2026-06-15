# Ablation Study Experiments

This folder contains curated outputs for the bus interior camera calibration ablation study.

Generated intermediate outputs under `results/` are ignored and not committed.
Only compact reports, CSV files, final summaries, and visualization artifacts are stored here.

## Experiment categories

### Detection-only ablations

These experiments evaluate marker detection robustness without running the full COLMAP / relay / final extrinsics pipeline.

- Gaussian blur:
  - `ablation_blur_k3`
  - `ablation_blur_k5`
  - `ablation_blur_k9`

- Motion blur:
  - `ablation_motion_blur_7`
  - `ablation_motion_blur_11`
  - `ablation_motion_blur_15`

- Lighting:
  - `light_ablation_detection/`

### Full-pipeline ablations

These experiments run detection, COLMAP, ArUco metric scale estimation, moving-camera relay evaluation, and final extrinsics export.

- Moving-camera motion blur:
  - `full_pipeline_motion_blur_7`
  - `full_pipeline_motion_blur_11`
  - `full_pipeline_motion_blur_15`

- Moving-camera lighting:
  - `full_pipeline_brightness_50`
  - `full_pipeline_brightness_200`
  - `full_pipeline_gamma_0_5`

- Combined static + moving degradation:
  - `full_pipeline_combined_motion_blur_15`

### Visualizations

Visualization folders contain contact sheets and annotated frames showing detected and missed marker IDs.

- `visualization_brightness_50`
- `visualization_brightness_200`
- `visualization_gamma_0_5`

## Key result

The ablation study shows that the calibration pipeline is most sensitive to directional motion blur.
Gaussian blur and lighting changes preserve ArUco marker detection, but motion blur removes critical marker anchors and can break moving-camera relay chains.

Lighting changes do not break ArUco detection, but linear brightness scaling can reduce COLMAP registration quality.
Gamma adjustment preserves both marker detection and COLMAP reconstruction in the tested setup.

The combined static-and-moving motion blur experiment represents a full-system stress test and causes a relay-calibration breakdown.
