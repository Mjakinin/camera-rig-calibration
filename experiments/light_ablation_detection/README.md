# Light Degradation Detection Ablation

This experiment evaluates lighting sensitivity for ArUco marker detection
in the moving relay camera and the static bus cameras.

Moving-camera conditions:
- brightness_50: darkened moving-camera sequence
- brightness_200: brightened moving-camera sequence
- gamma_0_5: nonlinear gamma change

Static-camera conditions:
- static_brightness_50
- static_brightness_200
- static_gamma_0_5

Purpose:
This detection-level ablation separates lighting sensitivity from motion-blur
sensitivity. The key outputs are the remaining marker IDs, missing marker IDs,
and annotated contact sheets showing detected and missed markers.

Interpretation:
- If all marker IDs remain detected, the pipeline is detection-robust for that condition.
- If target markers disappear, the corresponding relay chain can fail.
- If root markers disappear in cam_edge_3, the full relay calibration becomes unstable.
