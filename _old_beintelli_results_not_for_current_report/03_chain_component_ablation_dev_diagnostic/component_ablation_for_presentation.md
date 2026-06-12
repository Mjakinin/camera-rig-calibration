# Component Ablation for Presentation

This reduced table avoids misleading mixed-case interpretation.

| Case | Pair | Meaning | Baseline error [cm] | Translation error [cm] | Rotation error [deg] |
|---|---|---|---:|---:|---:|
| 1_all_gt_sanity_check | F3_R1 | GT static cameras + GT board poses + GT moving camera | 0.00 | 0.00 | 0.00 |
| 2_colmap_moving_only_error | F3_R1 | GT camera-board links + COLMAP moving motion | 14.72 | 18.15 | 4.21 |
| 3_final_no_gt_pipeline | F3_R1 | PnP camera-board links + COLMAP moving motion | 5.15 | 7.05 | 0.86 |
| 1_all_gt_sanity_check | F3_R3 | GT static cameras + GT board poses + GT moving camera | -0.00 | 0.00 | 0.00 |
| 2_colmap_moving_only_error | F3_R3 | GT camera-board links + COLMAP moving motion | -8.20 | 22.30 | 8.03 |
| 3_final_no_gt_pipeline | F3_R3 | PnP camera-board links + COLMAP moving motion | 2.07 | 5.03 | 1.24 |
| 1_all_gt_sanity_check | F4_R1 | GT static cameras + GT board poses + GT moving camera | 0.00 | 0.00 | 0.00 |
| 2_colmap_moving_only_error | F4_R1 | GT camera-board links + COLMAP moving motion | 13.43 | 137.25 | 6.52 |
| 3_final_no_gt_pipeline | F4_R1 | PnP camera-board links + COLMAP moving motion | 4.22 | 8.07 | 0.98 |
| 1_all_gt_sanity_check | F4_R3 | GT static cameras + GT board poses + GT moving camera | 0.00 | 0.00 | 0.00 |
| 2_colmap_moving_only_error | F4_R3 | GT camera-board links + COLMAP moving motion | -10.00 | 165.07 | 17.36 |
| 3_final_no_gt_pipeline | F4_R3 | PnP camera-board links + COLMAP moving motion | 1.14 | 7.02 | 1.35 |
