# Frozen Simulation Ablation Evidence

Status: **FROZEN FOR FINAL EVALUATION**

This document fixes the historical simulation ablation study that is reused for the final evaluation. The study is not to be recomputed merely to reproduce already available controlled simulation results.

## 1. Immutable historical source

Historical source commit:

```text
8f9dcea1e8b3189b3c195db2cafe65d5b0e5756b
```

All numerical ablation results referenced below are interpreted as results of that historical `main` implementation and must not be described as newly recomputed by the current `final/rigcal-v4.3` implementation.

## 2. Frozen ablation families and variants

The final simulation sensitivity study contains **25 controlled variants**.

### Moving-camera frame density

Canonical report:

```text
results/bus_real_data/ablation/moving_cam/density/ABLATION_SUMMARY/FULL_ABLATION_REPORT.txt
```

Frozen variants:

```text
density_stride_1_100pct
density_stride_2_50pct
density_stride_4_25pct
density_stride_8_12p5pct
density_stride_16_6p25pct
density_stride_8_offset4
density_route2_125pct_recaptured
```

### Moving-camera field of view

Canonical report:

```text
results/bus_real_data/ablation/moving_cam/fov/ABLATION_SUMMARY/FULL_ABLATION_REPORT.txt
```

Frozen variants:

```text
fov_40deg
fov_69deg_baseline
fov_100deg
fov_140deg_extreme
```

### Moving-camera resolution

Canonical report:

```text
results/bus_real_data/ablation/moving_cam/res/ABLATION_SUMMARY/FULL_ABLATION_REPORT.txt
```

Frozen variants:

```text
moving_res_160x90_extreme_pixel
moving_res_320x180_low
moving_res_1280x720_baseline
moving_res_2560x1440_upscaled
```

### Moving-camera motion blur

Canonical report:

```text
results/bus_real_data/ablation/moving_cam/motion_blur/ABLATION_SUMMARY/FULL_ABLATION_REPORT.txt
```

Frozen variants:

```text
moving_blur_k00_baseline
moving_blur_k09_mild
moving_blur_k21_strong
moving_blur_k41_extreme
```

### Scene lighting

Canonical report:

```text
results/bus_real_data/ablation/world/lighting/ABLATION_SUMMARY/FULL_ABLATION_REPORT.txt
```

Frozen variants:

```text
ceiling_dark_extreme
ceiling_low
ceiling_normal
ceiling_bright
```

### Route / viewpoint

Canonical report:

```text
results/bus_real_data/ablation/world/route/ABLATION_SUMMARY/FULL_ABLATION_REPORT.txt
```

Frozen variants:

```text
route1
route2
```

## 3. Authoritative metrics retained from the historical study

For every variant, the canonical `FULL_ABLATION_REPORT.txt` is the authoritative source for the following result fields when available:

- method completion / camera coverage status,
- all six unordered static-camera pair results,
- mean / median / maximum pairwise translation error [cm],
- mean / median / maximum pairwise rotation error [deg],
- mean baseline-length error [cm],
- mean baseline-direction error [deg],
- worst camera pair,
- secondary Ref14/world-aligned static-camera translation and rotation errors.

These accuracy and robustness metrics are the primary quantities used for final simulation-ablation plots and tables.

A representative canonical baseline (`moving_res_1280x720_baseline`, identical baseline condition also used by the no-blur and full-density studies) reports:

```text
AP01: mean pair translation error = 27.5020568734 cm
      mean pair rotation error    = 1.9184099823 deg

AP02: mean pair translation error = 5.4695312742 cm
      mean pair rotation error    = 0.3332054569 deg

AP03: mean pair translation error = 2.9243535263 cm
      mean pair rotation error    = 0.0702646127 deg
```

This baseline summary is included only as a convenient frozen headline. Detailed and variant-specific values remain authoritative in the corresponding historical full reports.

## 4. Runtime treatment

The historical clean-ablation runner measured per-method wall-clock time through:

```text
AP01_RUNTIME_SECONDS
AP02_RUNTIME_SECONDS
AP03_RUNTIME_SECONDS
```

However, later packaging/reconciliation did not preserve those runtime fields consistently in every final `RUN_STATUS.txt`. Therefore historical runtime is **secondary/non-authoritative** for the final ablation study. Do not invent missing runtimes and do not rerun the full simulation study solely to recover them.

Final simulation-ablation claims should focus on calibration accuracy, coverage and failure behavior. Runtime can be reported separately from current controlled runs where it is reliably recorded.

## 5. Parity evidence supporting reuse with the final implementation

The current `final/rigcal-v4.3` branch contains explicit historical-Main parity evidence under:

```text
parity/main_route2_v1/
```

### AP01

`parity/main_route2_v1/FINAL_PARITY_REPORT.txt` documents exact parity through observations, candidate construction, selection and final pose for the locked Main Route-2 evidence. The final-pose classification is `EXACT`.

### AP02

`parity/main_route2_v1/ap02/AP02_FINAL_POSE_PARITY.json` documents:

```text
classification = NUMERICALLY_EQUIVALENT_WITHIN_TOLERANCE
historical final mean reprojection = 0.404579 px
current final mean reprojection    = 0.404579 px
historical nfev = 57
current nfev    = 57
historical max_nfev = 80
```

The maximum recorded translation difference is on the order of `5.6e-8 m` and the maximum rotation difference about `1.2e-6 deg`.

### AP03

`parity/main_route2_v1/ap03/AP03_BASELINE_V1_VALIDATION.txt` documents different COLMAP registration counts in the current environment but equivalent final gauge-invariant extrinsics. The reported maximum gauge-invariant pair difference is approximately:

```text
translation = 0.0166733607 m
rotation    = 0.0692166815 deg
```

The difference is attributed to COLMAP environment/numerical reconstruction rather than a changed prepared-pixel acquisition contract.

## 6. Scientific-use rule

For the final thesis/evaluation:

1. Reuse the frozen historical simulation ablations above instead of rerunning all 25 variants.
2. Label them transparently as historical-main simulation experiments from commit `8f9dcea1...`.
3. Cite the parity evidence when relating them to the final implementation.
4. Do not mix a newly computed baseline into a historical ablation curve unless it is explicitly labelled as an independent validation run.
5. Ground Truth is evaluation evidence only; it must not be used for method selection or calibration.
6. Preserve failure/partial-coverage variants rather than deleting them; failure behavior is part of the sensitivity result.
7. Any future plots/tables should be generated from the frozen canonical reports or their directly derived machine-readable summaries, without changing the underlying method estimates.

## 7. Re-run policy

A full simulation-ablation rerun is required only if one of the following becomes true:

- the historical result artifact is shown to be corrupted or internally inconsistent,
- the evaluated method contract is intentionally changed in a way not covered by parity evidence,
- a new experimental question requires a simulation factor not present in the frozen study.

Otherwise the frozen historical study is the final simulation-ablation evidence set.
