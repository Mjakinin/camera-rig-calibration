
Ablation result standard

========================



Each ablation variant must create:



FINAL_RESULTS/

- FINAL_REPORT.txt

- PRIMARY_PAIRWISE_SUMMARY.csv

- PRIMARY_PAIRWISE_DETAIL.csv

- SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv

- SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv

- DIAGNOSTIC_AP03_COLMAP_RECONSTRUCTION.txt

- DIAGNOSTIC_AP03_MARKER_SIZE_SCALE.txt

- DIAGNOSTIC_AP03_MARKER_SIZE_SCALE.json

- MANIFEST.json



Group summaries must be written to:



ABLATION_SUMMARY/

- ABLATION_PARAMETER_EFFECT_SUMMARY.txt

- ABLATION_PARAMETER_EFFECT_SUMMARY.csv

- ABLATION_PAIRWISE_ALL_VARIANTS.csv



Primary metric:

pairwise static camera-to-camera extrinsic errors.



Secondary metric:

Ref14/world-frame static camera-map vs GT after evaluation-only SE(3) alignment.



Diagnostics:

AP02 full marker-map evaluation, AP03 COLMAP coverage, AP03 marker-size-only scale stability.

