# AP02 Final Ref-ArUco Evaluation

**Approach:** AP02 reference-marker graph BA  
**Selected variant:** `graph_ba_with_moving_sparse`  
**Reference frame:** `aruco_marker_14 / aruco_ref_floor_14`

## Summary

| metric | value |
|---|---:|
| camera count | 4 |
| camera mean translation error | 38.587 cm |
| camera median translation error | 13.590 cm |
| camera mean rotation error | 3.678 deg |
| camera median rotation error | 1.970 deg |
| markers excluding ref | 14 |
| marker mean translation error excluding ref | 35.492 cm |
| marker median translation error excluding ref | 13.058 cm |
| marker mean rotation error excluding ref | 4.463 deg |
| marker median rotation error excluding ref | 3.305 deg |

## Static camera extrinsics vs GT

| camera | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cam_edge_0 | 10.541 | 2.032 | -3.189 | 1.218 | 9.972 | ( +0.556,  -1.722,  +2.187) | ( +0.588,  -1.735,  +2.087) |
| cam_edge_1 | 16.640 | 1.749 | -16.338 | 3.119 | -0.473 | ( -4.138,  +0.506,  +2.127) | ( -3.974,  +0.475,  +2.132) |
| cam_edge_3 | 8.551 | 1.908 | -8.213 | -2.374 | -0.194 | ( -3.309,  -1.587,  +2.052) | ( -3.227,  -1.563,  +2.054) |
| cam_edge_5 | 118.616 | 9.025 | -2.170 | -99.445 | 64.619 | ( +5.837,  -0.646,  +2.785) | ( +5.859,  +0.348,  +2.139) |

## Marker map vs GT

Reference marker 14 is the fixed gauge/reference frame and is excluded from this table. It should not be interpreted as a zero-error estimated marker. Detection quality is evaluated via reprojection/corner residuals.

| marker | id | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marker_000 | 0 | 15.760 | 9.256 | -9.619 | 6.933 | -10.383 | ( -2.646,  +0.049,  +0.676) | ( -2.550,  -0.020,  +0.780) |
| marker_001 | 1 | 8.340 | 1.323 | -7.624 | -3.007 | 1.546 | ( -2.326,  -1.055,  +1.245) | ( -2.250,  -1.025,  +1.230) |
| marker_002 | 2 | 4.895 | 2.086 | -4.097 | -2.416 | 1.154 | ( -1.141,  -1.049,  +0.712) | ( -1.100,  -1.025,  +0.700) |
| marker_003 | 3 | 1.957 | 1.664 | -0.615 | 1.735 | -0.663 | ( +0.094,  +0.727,  +0.393) | ( +0.100,  +0.710,  +0.400) |
| marker_004 | 4 | 5.380 | 3.810 | -1.410 | 4.226 | 3.016 | ( +0.686,  +0.752,  +1.630) | ( +0.700,  +0.710,  +1.600) |
| marker_005 | 5 | 11.513 | 1.667 | 9.748 | -3.129 | 5.267 | ( +1.997,  -1.511,  +0.753) | ( +1.900,  -1.480,  +0.700) |
| marker_006 | 6 | 7.176 | 1.229 | 4.761 | 2.703 | 4.640 | ( +2.018,  +0.027,  +1.016) | ( +1.970,  -0.000,  +0.970) |
| marker_007 | 7 | 6.072 | 1.514 | 2.554 | 3.526 | 4.231 | ( +1.996,  +0.470,  +1.012) | ( +1.970,  +0.435,  +0.970) |
| marker_008 | 8 | 14.603 | 2.801 | 12.784 | -0.357 | 7.050 | ( +2.778,  -1.029,  +1.080) | ( +2.650,  -1.025,  +1.010) |
| marker_009 | 9 | 73.768 | 4.403 | -1.090 | -63.744 | 37.110 | ( +3.009,  -0.657,  +1.391) | ( +3.020,  -0.020,  +1.020) |
| marker_010 | 10 | 77.179 | 8.541 | -18.072 | -66.394 | 34.956 | ( +2.839,  -2.144,  +1.370) | ( +3.020,  -1.480,  +1.020) |
| marker_011 | 11 | 89.645 | 7.939 | -5.119 | -76.637 | 46.226 | ( +3.949,  -1.266,  +1.592) | ( +4.000,  -0.500,  +1.130) |
| marker_012 | 12 | 91.051 | 8.784 | -16.672 | -77.801 | 44.263 | ( +3.873,  -2.258,  +1.463) | ( +4.040,  -1.480,  +1.020) |
| marker_013 | 13 | 89.551 | 7.465 | 5.607 | -76.019 | 47.000 | ( +4.106,  -0.530,  +1.340) | ( +4.050,  +0.230,  +0.870) |
