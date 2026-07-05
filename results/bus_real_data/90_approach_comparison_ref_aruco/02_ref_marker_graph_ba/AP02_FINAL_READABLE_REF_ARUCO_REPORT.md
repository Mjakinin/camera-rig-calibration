# AP02 Final Ref-ArUco Evaluation

**Approach:** AP02 reference-marker graph BA  
**Selected variant:** `graph_ba_with_moving_sparse`  
**Reference frame:** `aruco_marker_14 / aruco_ref_floor_14`

## Summary

| metric | value |
|---|---:|
| camera count | 4 |
| camera mean translation error | 8.655 cm |
| camera median translation error | 7.511 cm |
| camera mean rotation error | 0.409 deg |
| camera median rotation error | 0.323 deg |
| markers excluding ref | 14 |
| marker mean translation error excluding ref | 6.368 cm |
| marker median translation error excluding ref | 5.533 cm |
| marker mean rotation error excluding ref | 0.886 deg |
| marker median rotation error excluding ref | 0.717 deg |

## Static camera extrinsics vs GT

| camera | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cam_edge_0 | 5.955 | 0.759 | 3.637 | -4.363 | 1.788 | ( +0.624,  -1.778,  +2.105) | ( +0.588,  -1.735,  +2.087) |
| cam_edge_1 | 8.685 | 0.293 | -8.436 | 0.556 | 1.988 | ( -4.059,  +0.481,  +2.151) | ( -3.974,  +0.475,  +2.132) |
| cam_edge_3 | 6.337 | 0.352 | -5.858 | -1.576 | 1.830 | ( -3.286,  -1.579,  +2.073) | ( -3.227,  -1.563,  +2.054) |
| cam_edge_5 | 13.643 | 0.234 | 13.610 | -0.066 | -0.947 | ( +5.995,  +0.348,  +2.129) | ( +5.859,  +0.348,  +2.139) |

## Marker map vs GT

Reference marker 14 is the fixed gauge/reference frame and is excluded from this table. It should not be interpreted as a zero-error estimated marker. Detection quality is evaluated via reprojection/corner residuals.

| marker | id | t err [cm] | r err [deg] | dX [cm] | dY [cm] | dZ [cm] | est xyz [m] | gt xyz [m] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marker_000 | 0 | 5.838 | 3.289 | -5.806 | -0.553 | -0.263 | ( -2.608,  -0.026,  +0.777) | ( -2.550,  -0.020,  +0.780) |
| marker_001 | 1 | 5.751 | 0.783 | -5.069 | -1.629 | 2.173 | ( -2.301,  -1.041,  +1.252) | ( -2.250,  -1.025,  +1.230) |
| marker_002 | 2 | 4.169 | 1.079 | -3.549 | -1.584 | 1.508 | ( -1.135,  -1.041,  +0.715) | ( -1.100,  -1.025,  +0.700) |
| marker_003 | 3 | 0.874 | 0.723 | -0.306 | 0.788 | 0.224 | ( +0.097,  +0.718,  +0.402) | ( +0.100,  +0.710,  +0.400) |
| marker_004 | 4 | 2.743 | 0.601 | 0.857 | 0.394 | 2.576 | ( +0.709,  +0.714,  +1.626) | ( +0.700,  +0.710,  +1.600) |
| marker_005 | 5 | 4.362 | 0.584 | 3.329 | -1.942 | 2.043 | ( +1.933,  -1.499,  +0.720) | ( +1.900,  -1.480,  +0.700) |
| marker_006 | 6 | 3.662 | 1.013 | 2.896 | 0.738 | 2.116 | ( +1.999,  +0.007,  +0.991) | ( +1.970,  -0.000,  +0.970) |
| marker_007 | 7 | 3.809 | 0.800 | 2.967 | 1.479 | 1.874 | ( +2.000,  +0.450,  +0.989) | ( +1.970,  +0.435,  +0.970) |
| marker_008 | 8 | 5.316 | 0.702 | 4.413 | -0.886 | 2.828 | ( +2.694,  -1.034,  +1.038) | ( +2.650,  -1.025,  +1.010) |
| marker_009 | 9 | 9.632 | 0.256 | 9.533 | -0.642 | -1.218 | ( +3.115,  -0.026,  +1.008) | ( +3.020,  -0.020,  +1.020) |
| marker_010 | 10 | 10.168 | 0.556 | 9.842 | -2.478 | -0.606 | ( +3.118,  -1.505,  +1.014) | ( +3.020,  -1.480,  +1.020) |
| marker_011 | 11 | 10.984 | 0.711 | 10.884 | -1.197 | -0.873 | ( +4.109,  -0.512,  +1.121) | ( +4.000,  -0.500,  +1.130) |
| marker_012 | 12 | 11.107 | 0.487 | 10.911 | -1.876 | -0.896 | ( +4.149,  -1.499,  +1.011) | ( +4.040,  -1.480,  +1.020) |
| marker_013 | 13 | 10.731 | 0.821 | 10.603 | -0.402 | -1.606 | ( +4.156,  +0.226,  +0.854) | ( +4.050,  +0.230,  +0.870) |
