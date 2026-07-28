# rigcal pipeline

`run/rigcal.py` is only a thin launcher. All capture, preparation, ArUco,
calibration, evaluation and reporting code lives in
`src/camera_rig_calibration/`.

The terminal uses this logical order:

```text
1 Capture or import
  → 2 Normalize, extract frames and resolve intrinsics
  → 3 Validate the dataset
  → 4 Detect ArUco markers and write debug images
  → 5 Check observation quality and select references
  → 6 Run one calibration method and its internal substages
  → 7 Evaluate every primary method on a common anchor
  → 8 Build the cross-method comparison
  → 9 Atomically publish the experiment and summary
```

Numbers describe execution order only. Each persistent experiment uses
descriptive folders such as `raw_images`, `observations`, `diagnostics`,
`evaluations`, `logs` and `provenance`.
