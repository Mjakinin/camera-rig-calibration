# Full Pipeline Ablation: Combined Static + Moving Motion Blur length=15

Branch: ablation-study
Commit: 8a58b56

Degradation:
- Static cameras: motion_blur length=15
- Moving camera: motion_blur length=15

Key result:
The combined degradation causes a full relay-calibration breakdown.

Static detection:
- Remaining degraded static IDs across all cameras: [1, 11]
- Globally missed static IDs: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
- cam_edge_0 detects no markers after degradation.
- cam_edge_1 detects only marker 1.
- cam_edge_3 detects only marker 1.
- cam_edge_5 detects only marker 11.

Moving detection:
- Frames: 131
- Total detections: 91
- Missing moving marker IDs: [2, 4, 6, 7]
- Max consecutive frames without marker: 24

COLMAP:
- COLMAP poses loaded in relay: 87
- ArUco metric scale: 0.557285340105

Relay result:
- cam3_to_cam0 GT_motion: no valid chains
- cam3_to_cam0 COLMAP_motion: no valid chains
- cam3_to_cam5 GT_motion: no valid chains
- cam3_to_cam5 COLMAP_motion: no valid chains

Final extrinsics:
- Only cam_edge_3_to_cam_edge_1_direct_static remains.
- Translation error: 28.84 cm
- Rotation error: 4.42 deg

Interpretation:
Strong motion blur on both static and moving cameras removes the required
ArUco anchor observations. Even though COLMAP still reconstructs a partial
moving-camera trajectory, the relay calibration cannot produce valid chains
for cam_edge_0 or cam_edge_5. This should be interpreted as a full-system
failure under severe combined image degradation.
