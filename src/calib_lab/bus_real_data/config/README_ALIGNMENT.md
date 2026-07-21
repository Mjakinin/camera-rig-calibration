# bus_real_data alignment baseline

This setup uses the provided real Intellibus camera transforms and camera intrinsics.

Static color cameras:
- cam_edge_0 <- center_left camera_info
- cam_edge_1 <- front_right camera_info
- cam_edge_3 <- front_left camera_info
- cam_edge_5 <- back_right camera_info

Transform source:
- target_transforms.json gives the color optical frame poses relative to bus_aruco_center_link / base_link.
- bus_aruco_center_link is identical to base_link in the provided transform file.

Gazebo frame alignment:
- The Gazebo BeIntelli bus mesh does not visually share the exact same origin as the real base_link frame.
- Therefore, the real camera rig is mapped into the Gazebo bus frame with one global alignment transform.
- This global transform is applied equally to all cameras.
- The relative camera-to-camera geometry from target_transforms.json is preserved.

Current mapping:
- real y is interpreted as bus longitudinal axis.
- Gazebo x is bus longitudinal axis.
- Mapping:
  gazebo_x = -real_y + offset_x
  gazebo_y =  real_x + offset_y
  gazebo_z =  real_z + offset_z

Current baseline values:
- BASE_TO_GAZEBO_YAW = +90 deg
- CAMERA_LAYOUT_OFFSET_GAZEBO = [1.0, 0.1, 0.5]
- Intrinsics are loaded per camera from config/camera_intrinsics_by_camera.yaml
- Resolution: 1280x720
- Distortion coefficients: zero, as provided
- Horizontal FOV is derived from fx for each camera, approximately 69 deg

Important:
- This is not a per-camera manual correction.
- It is one global rig-to-mesh alignment.
- Exact pixel-level agreement with real images is not expected because the Gazebo bus mesh and real bus interior are not identical.
- This baseline is considered sufficient for the next simulation pipeline stage: A4 ArUco marker detection and PnP.
