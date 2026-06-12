# 01 Station ID Detection

This folder summarizes the ArUco ID detection and station suitability.

The Gazebo world contains all 8 ArUco stations at the same time:
- Front: F1, F2, F3, F4
- Rear: R1, R2, R3
- Floor/general: G

Each station has a unique marker ID range.
Several boards can be visible in one image.
For each station, the estimator filters only the IDs belonging to that station.

Files:
- station_suitability.csv
- station_suitability.md
- station_visibility_matrix.csv
- station_visibility_matrix.md
- debug_images/front_static_representative_detection.png
- debug_images/rear_static_representative_detection.png

Valid front anchor candidates:
- F3
- F4

Valid rear anchor candidates:
- R1
- R3

Why only four evaluated front-rear pairs:
F3/F4 are valid front anchors and R1/R3 are valid rear anchors.
Therefore:
- F3_R1
- F3_R3
- F4_R1
- F4_R3

Not selected:
- F1/F2: moving-camera observation is not reliable enough
- R2: rear_static sees it, but moving-camera observation is not reliable enough
- G: floor/general limit case, not used as front-rear anchor
