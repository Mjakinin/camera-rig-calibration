# Evaluation

Computes numerical metrics. It does not format the final report tree.

Primary metric:

- pairwise static camera-to-camera extrinsic error,
- no global alignment,
- six pairs for four cameras,
- translation, rotation and explicit coverage.

Secondary metric:

- best-fit rigid SE(3) camera-map alignment,
- no scale optimization,
- per-camera and aggregate residuals.

AP02 diagnostics may additionally evaluate Ref14 gauge, GT-aligned full maps,
held-out behavior, parameter stability and validity independently of coverage.
