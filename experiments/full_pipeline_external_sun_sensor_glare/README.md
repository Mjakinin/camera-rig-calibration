# Full pipeline: external_sun_sensor_glare

This experiment evaluates the moving-camera calibration pipeline under a realistic sunlight stress test.

Setup:
- Gazebo scene with an external sunlight source.
- Additional camera-level glare/saturation applied to the rendered moving-camera sequence.
- ArUco detection, COLMAP reconstruction, ArUco metric scale estimation, relay-chain evaluation, and final extrinsic export are run on the degraded sequence.

Key results:
- Moving-camera ArUco detections: 141
- Missing marker IDs: none
- COLMAP registered images: 32
- ArUco metric COLMAP scale: 0.127563759157

Final extrinsic errors:
- cam3 -> cam0 COLMAP relay: 17.39 cm, 3.84 deg
- cam3 -> cam5 COLMAP relay: 112.49 cm, 7.94 deg

Interpretation:
The sunlight/glare degradation does not completely remove marker identities, but it strongly reduces temporal detection coverage and severely degrades COLMAP registration. The cam3->cam5 relay becomes unreliable, while cam3->cam0 remains partially usable.
