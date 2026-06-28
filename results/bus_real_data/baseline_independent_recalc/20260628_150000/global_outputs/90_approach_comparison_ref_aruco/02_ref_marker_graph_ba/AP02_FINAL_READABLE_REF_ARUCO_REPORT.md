# AP02 Final Ref-ArUco Evaluation

**Approach:** AP02 reference-marker graph BA  
**Selected variant:** `graph_ba_with_moving_sparse`  
**Reference frame:** `aruco_marker_14 / aruco_ref_floor_14`

## Summary

| metric | value |
|---|---:|
| camera count | 4 |
| camera mean translation error | 22.676 cm |
| camera median translation error | 18.330 cm |
| camera mean rotation error | 3.479 deg |
| camera median rotation error | 3.282 deg |
| markers excluding ref | 14 |
| marker mean translation error excluding ref | 17.484 cm |
| marker median translation error excluding ref | 10.816 cm |
| marker mean rotation error excluding ref | 4.748 deg |
| marker median rotation error excluding ref | 3.952 deg |

## Static camera extrinsics vs GT

| camera | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cam_edge_0 | 21.465 | 5.251 | -19.351 | 5.113 | -7.754 | ( +0.394,  -1.683,  +2.010) | ( +0.588,  -1.735,  +2.087) |
| cam_edge_1 | 15.194 | 2.099 | -10.582 | 0.037 | -10.904 | ( -4.080,  +0.475,  +2.023) | ( -3.974,  +0.475,  +2.132) |
| cam_edge_3 | 11.359 | 2.293 | -6.207 | -2.068 | -9.286 | ( -3.289,  -1.584,  +1.961) | ( -3.227,  -1.563,  +2.054) |
| cam_edge_5 | 42.684 | 4.271 | 20.854 | -35.060 | -12.563 | ( +6.067,  -0.002,  +2.013) | ( +5.859,  +0.348,  +2.139) |

## Marker map vs GT

Reference marker 14 is the fixed gauge/reference frame and is excluded from this table. It should not be interpreted as a zero-error estimated marker. Detection quality is evaluated via reprojection/corner residuals.

| marker | id | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marker_000 | 0 | 9.523 | 2.482 | -3.237 | -0.910 | -8.910 | ( -2.582,  -0.029,  +0.691) | ( -2.550,  -0.020,  +0.780) |
| marker_001 | 1 | 7.503 | 2.237 | -4.342 | -1.801 | -5.849 | ( -2.293,  -1.043,  +1.172) | ( -2.250,  -1.025,  +1.230) |
| marker_002 | 2 | 4.196 | 2.655 | -1.717 | -1.301 | -3.601 | ( -1.117,  -1.038,  +0.664) | ( -1.100,  -1.025,  +0.700) |
| marker_003 | 3 | 10.655 | 1.900 | 7.547 | 5.967 | -4.577 | ( +0.175,  +0.770,  +0.354) | ( +0.100,  +0.710,  +0.400) |
| marker_004 | 4 | 6.053 | 13.968 | -2.595 | 4.324 | -3.349 | ( +0.674,  +0.753,  +1.567) | ( +0.700,  +0.710,  +1.600) |
| marker_005 | 5 | 10.977 | 2.881 | 10.132 | -2.201 | 3.605 | ( +2.001,  -1.502,  +0.736) | ( +1.900,  -1.480,  +0.700) |
| marker_006 | 6 | 5.643 | 6.052 | 2.083 | 1.514 | 5.021 | ( +1.991,  +0.015,  +1.020) | ( +1.970,  -0.000,  +0.970) |
| marker_007 | 7 | 5.749 | 4.932 | 1.825 | 1.925 | 5.100 | ( +1.988,  +0.454,  +1.021) | ( +1.970,  +0.435,  +0.970) |
| marker_008 | 8 | 12.130 | 4.513 | 9.671 | -0.403 | 7.309 | ( +2.747,  -1.029,  +1.083) | ( +2.650,  -1.025,  +1.010) |
| marker_009 | 9 | 32.747 | 8.563 | 14.022 | -27.415 | -11.143 | ( +3.160,  -0.294,  +0.909) | ( +3.020,  -0.020,  +1.020) |
| marker_010 | 10 | 30.362 | 3.936 | 8.847 | -28.883 | -3.067 | ( +3.108,  -1.769,  +0.989) | ( +3.020,  -1.480,  +1.020) |
| marker_011 | 11 | 35.910 | 3.968 | 14.126 | -31.998 | -8.128 | ( +4.141,  -0.820,  +1.049) | ( +4.000,  -0.500,  +1.130) |
| marker_012 | 12 | 34.460 | 4.549 | 9.472 | -32.957 | -3.407 | ( +4.135,  -1.810,  +0.986) | ( +4.040,  -1.480,  +1.020) |
| marker_013 | 13 | 38.868 | 3.842 | 16.495 | -32.727 | -12.945 | ( +4.215,  -0.097,  +0.741) | ( +4.050,  +0.230,  +0.870) |
