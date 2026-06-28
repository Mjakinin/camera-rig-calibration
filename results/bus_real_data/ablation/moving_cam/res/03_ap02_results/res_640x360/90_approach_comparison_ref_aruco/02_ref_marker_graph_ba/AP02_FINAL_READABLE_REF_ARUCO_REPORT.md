# AP02 Final Ref-ArUco Evaluation

**Approach:** AP02 reference-marker graph BA  
**Selected variant:** `graph_ba_with_moving_sparse`  
**Reference frame:** `aruco_marker_14 / aruco_ref_floor_14`

## Summary

| metric | value |
|---|---:|
| camera count | 4 |
| camera mean translation error | 28.984 cm |
| camera median translation error | 11.879 cm |
| camera mean rotation error | 2.962 deg |
| camera median rotation error | 1.959 deg |
| markers excluding ref | 14 |
| marker mean translation error excluding ref | 27.057 cm |
| marker median translation error excluding ref | 9.302 cm |
| marker mean rotation error excluding ref | 3.360 deg |
| marker median rotation error excluding ref | 3.196 deg |

## Static camera extrinsics vs GT

| camera | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cam_edge_0 | 9.325 | 1.843 | -0.508 | -0.370 | 9.304 | ( +0.583,  -1.738,  +2.180) | ( +0.588,  -1.735,  +2.087) |
| cam_edge_1 | 14.434 | 1.884 | -13.661 | 2.527 | -3.913 | ( -4.111,  +0.500,  +2.092) | ( -3.974,  +0.475,  +2.132) |
| cam_edge_3 | 7.674 | 2.034 | -7.202 | -1.448 | -2.221 | ( -3.299,  -1.578,  +2.032) | ( -3.227,  -1.563,  +2.054) |
| cam_edge_5 | 84.502 | 6.085 | 21.288 | -80.260 | -15.674 | ( +6.072,  -0.454,  +1.982) | ( +5.859,  +0.348,  +2.139) |

## Marker map vs GT

Reference marker 14 is the fixed gauge/reference frame and is excluded from this table. It should not be interpreted as a zero-error estimated marker. Detection quality is evaluated via reprojection/corner residuals.

| marker | id | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marker_000 | 0 | 6.287 | 3.751 | -4.903 | -1.157 | -3.762 | ( -2.599,  -0.032,  +0.742) | ( -2.550,  -0.020,  +0.780) |
| marker_001 | 1 | 6.550 | 1.529 | -6.218 | -2.029 | -0.349 | ( -2.312,  -1.045,  +1.227) | ( -2.250,  -1.025,  +1.230) |
| marker_002 | 2 | 3.657 | 2.640 | -3.211 | -1.746 | 0.130 | ( -1.132,  -1.042,  +0.701) | ( -1.100,  -1.025,  +0.700) |
| marker_003 | 3 | 2.048 | 1.761 | -1.932 | 0.574 | -0.362 | ( +0.081,  +0.716,  +0.396) | ( +0.100,  +0.710,  +0.400) |
| marker_004 | 4 | 5.062 | 3.923 | -2.059 | 3.370 | 3.168 | ( +0.679,  +0.744,  +1.632) | ( +0.700,  +0.710,  +1.600) |
| marker_005 | 5 | 12.055 | 1.418 | 10.794 | -2.429 | 4.785 | ( +2.008,  -1.504,  +0.748) | ( +1.900,  -1.480,  +0.700) |
| marker_006 | 6 | 6.341 | 2.610 | 3.478 | 2.695 | 4.566 | ( +2.005,  +0.027,  +1.016) | ( +1.970,  -0.000,  +0.970) |
| marker_007 | 7 | 5.445 | 1.296 | 1.320 | 3.209 | 4.196 | ( +1.983,  +0.467,  +1.012) | ( +1.970,  +0.435,  +0.970) |
| marker_008 | 8 | 13.237 | 2.334 | 11.135 | 0.148 | 7.156 | ( +2.761,  -1.024,  +1.082) | ( +2.650,  -1.025,  +1.010) |
| marker_009 | 9 | 57.526 | 4.218 | 11.002 | -54.953 | -12.976 | ( +3.130,  -0.570,  +0.890) | ( +3.020,  -0.020,  +1.020) |
| marker_010 | 10 | 58.469 | 5.103 | -1.760 | -57.411 | -10.932 | ( +3.002,  -2.054,  +0.911) | ( +3.020,  -1.480,  +1.020) |
| marker_011 | 11 | 66.679 | 5.574 | 9.436 | -64.747 | -12.842 | ( +4.094,  -1.147,  +1.002) | ( +4.000,  -0.500,  +1.130) |
| marker_012 | 12 | 67.649 | 5.626 | -0.911 | -66.474 | -12.521 | ( +4.031,  -2.145,  +0.895) | ( +4.040,  -1.480,  +1.020) |
| marker_013 | 13 | 67.791 | 5.257 | 15.189 | -64.385 | -14.818 | ( +4.202,  -0.414,  +0.722) | ( +4.050,  +0.230,  +0.870) |
