# AP02 Final Ref-ArUco Evaluation

**Approach:** AP02 reference-marker graph BA  
**Selected variant:** `graph_ba_with_moving_sparse`  
**Reference frame:** `aruco_marker_14 / aruco_ref_floor_14`

## Summary

| metric | value |
|---|---:|
| camera count | 4 |
| camera mean translation error | 18.203 cm |
| camera median translation error | 13.410 cm |
| camera mean rotation error | 2.916 deg |
| camera median rotation error | 2.673 deg |
| markers excluding ref | 14 |
| marker mean translation error excluding ref | 14.870 cm |
| marker median translation error excluding ref | 8.736 cm |
| marker mean rotation error excluding ref | 3.404 deg |
| marker median rotation error excluding ref | 3.041 deg |

## Static camera extrinsics vs GT

| camera | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cam_edge_0 | 8.225 | 2.380 | -7.103 | 3.063 | 2.794 | ( +0.517,  -1.704,  +2.115) | ( +0.588,  -1.735,  +2.087) |
| cam_edge_1 | 16.080 | 2.598 | -9.968 | 8.022 | -9.740 | ( -4.074,  +0.555,  +2.034) | ( -3.974,  +0.475,  +2.132) |
| cam_edge_3 | 10.739 | 2.748 | -7.521 | 4.412 | -6.269 | ( -3.303,  -1.519,  +1.992) | ( -3.227,  -1.563,  +2.054) |
| cam_edge_5 | 37.769 | 3.939 | 22.756 | -29.234 | -7.351 | ( +6.086,  +0.056,  +2.065) | ( +5.859,  +0.348,  +2.139) |

## Marker map vs GT

Reference marker 14 is the fixed gauge/reference frame and is excluded from this table. It should not be interpreted as a zero-error estimated marker. Detection quality is evaluated via reprojection/corner residuals.

| marker | id | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marker_000 | 0 | 8.259 | 2.752 | -2.479 | 2.900 | -7.326 | ( -2.575,  +0.009,  +0.707) | ( -2.550,  -0.020,  +0.780) |
| marker_001 | 1 | 7.015 | 2.312 | -5.570 | 2.337 | -3.568 | ( -2.306,  -1.002,  +1.194) | ( -2.250,  -1.025,  +1.230) |
| marker_002 | 2 | 3.716 | 1.946 | -3.306 | 0.813 | -1.491 | ( -1.133,  -1.017,  +0.685) | ( -1.100,  -1.025,  +0.700) |
| marker_003 | 3 | 5.054 | 1.570 | 3.047 | 2.732 | -2.965 | ( +0.130,  +0.737,  +0.370) | ( +0.100,  +0.710,  +0.400) |
| marker_004 | 4 | 3.940 | 6.939 | -0.809 | 3.823 | -0.501 | ( +0.692,  +0.748,  +1.595) | ( +0.700,  +0.710,  +1.600) |
| marker_005 | 5 | 9.213 | 1.941 | 6.518 | -3.316 | 5.602 | ( +1.965,  -1.513,  +0.756) | ( +1.900,  -1.480,  +0.700) |
| marker_006 | 6 | 5.026 | 2.669 | 0.845 | 0.153 | 4.952 | ( +1.978,  +0.002,  +1.020) | ( +1.970,  -0.000,  +0.970) |
| marker_007 | 7 | 4.465 | 2.694 | 1.216 | 0.587 | 4.256 | ( +1.982,  +0.441,  +1.013) | ( +1.970,  +0.435,  +0.970) |
| marker_008 | 8 | 10.368 | 5.705 | 5.734 | -2.139 | 8.370 | ( +2.707,  -1.046,  +1.094) | ( +2.650,  -1.025,  +1.010) |
| marker_009 | 9 | 29.336 | 4.833 | 18.685 | -21.694 | -6.391 | ( +3.207,  -0.237,  +0.956) | ( +3.020,  -0.020,  +1.020) |
| marker_010 | 10 | 26.210 | 3.645 | 12.639 | -22.956 | 0.469 | ( +3.146,  -1.710,  +1.025) | ( +3.020,  -1.480,  +1.020) |
| marker_011 | 11 | 31.508 | 3.579 | 17.487 | -25.955 | -3.644 | ( +4.175,  -0.760,  +1.094) | ( +4.000,  -0.500,  +1.130) |
| marker_012 | 12 | 29.619 | 3.736 | 12.830 | -26.694 | 0.375 | ( +4.168,  -1.747,  +1.024) | ( +4.040,  -1.480,  +1.020) |
| marker_013 | 13 | 34.453 | 3.330 | 20.390 | -26.750 | -7.466 | ( +4.254,  -0.037,  +0.795) | ( +4.050,  +0.230,  +0.870) |
