# bus_real_data

Clean realistic Intellibus camera-rig calibration setup.

Main idea:
- reproduce the real Intellibus camera layout from target_transforms.json
- use 1280x720 simulated color cameras
- use realistic A4 sheets with one ArUco marker per sheet
- marker side length: approximately 0.17 m
- use a moving calibration camera as relay
- use COLMAP as first trajectory backend
- keep RTAB-Map as optional later trajectory backend
- estimate pairwise static camera transforms, initially from cam_edge_0 to other edge cameras

Initial static cameras:
- cam_edge_0
- cam_edge_1
- cam_edge_3
- cam_edge_5

Initial outputs:
- T_edge0_edge1
- T_edge0_edge3
- T_edge0_edge5
