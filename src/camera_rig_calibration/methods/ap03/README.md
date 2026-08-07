# AP03 — reconstructed Main SfM/Multi-camera baseline

AP03 defaults to the immutable `baseline_v1` contract reconstructed from Main.
It registers the four static images and every sorted moving image together,
with one calibrated COLMAP camera per static camera and one shared camera for
the moving sequence. Intrinsics remain fixed. Main left SIFT feature limits
unset, used exhaustive CPU matching with eight mapper matches, and selected
the sparse model by registered-static count, total registrations, then point
count.

The primary Multi result re-detects markers 0–14 in registered images, rejects
detections below 100 px², triangulates marker corners, and obtains metric scale
from four edges plus two diagonals. Only COLMAP translations are scaled, so the
reference remains the native COLMAP gauge. Ground Truth is forbidden during
reconstruction and calibration.

Execution order:

1. `prepare_colmap.py` — create grouped COLMAP input
2. `reconstruct_stage.py` / `reconstruct.py` — run the shared reconstruction
3. `inspect_stage.py` / `inspect.py` — select and validate the sparse model
4. `estimate_scale.py` / `scale_core.py` — estimate Single and Multi scale
5. `report.py` — normalize both results, with Multi as primary

The COLMAP reconstruction is shared; Single is a negligible diagnostic and
Multi is the primary baseline. `wizard_explicit_limits_v1` and
`wizard_filtered_observations_v1` preserve newer Wizard behavior as explicit
advanced settings. Future scientific changes require a new contract name such
as `baseline_v2`.
