# Real-vehicle calibration pipeline

Canonical dataset: `results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input`

Canonical methods:

- AP01 baseline: direct multimarker, moving-COLMAP relay only where required.
- AP02 baseline: distortion-aware reference-marker graph BA with soft-L1.
- AP03 current: calibrated grouped COLMAP with marker-size-only metric scale.

Canonical entry point:

```bash
bash run/real_vehicle_data/run_full_real_pipeline.sh
```

Scripts `11`–`13` and the robust-refinement runner were development candidates and
are intentionally not part of the final pipeline.
