# Full pipeline: external_sun_sensor_glare

This experiment evaluates the bus camera-rig calibration pipeline under a realistic sunlight stress test.

Setup:
- Gazebo scene with an external sunlight source.
- Additional camera-level glare/saturation applied to the rendered moving-camera sequence.
- Full pipeline: ArUco detection, COLMAP reconstruction, ArUco metric scale estimation, relay-chain evaluation, and final extrinsic export.

Key results:
- Moving-camera ArUco detections: 141
- Missing marker IDs: none
- COLMAP registered images: 32
- ArUco metric COLMAP scale: 0.127563759157

Final extrinsic errors:
- cam3 -> cam0 COLMAP relay: 17.39 cm, 3.84 deg
- cam3 -> cam5 COLMAP relay: 112.49 cm, 7.94 deg

Interpretation:
External Gazebo sunlight alone had only a small effect on ArUco detection. After adding camera-level glare and saturation, all marker IDs were still observed at least once, but temporal detection coverage dropped strongly and COLMAP registration collapsed to 32 images. The cam3->cam0 relay remains partially usable, while cam3->cam5 becomes unreliable.
