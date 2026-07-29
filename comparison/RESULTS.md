# COLMAP Matcher and Route Comparison

## Overview

This experiment evaluates COLMAP reconstruction performance on the AP03 bus dataset.

The main goals are:

- compare exhaustive and sequential matching,
- analyze reconstruction fragmentation,
- identify why specific moving-camera frames fail,
- test mapper threshold changes,
- improve the moving-camera route,
- and determine which configuration produces one complete and geometrically plausible reconstruction.

---

## Dataset Variants

### Original Route

The original AP03 dataset contains:

- **270 moving-camera frames**
- **4 static-camera images**

Total:

```text
274 images
```

Static cameras:

- `cam_edge_0`
- `cam_edge_1`
- `cam_edge_3`
- `cam_edge_5`

### Improved Route

The improved route contains:

- **220 moving-camera frames**
- **4 static-camera images**

Total:

```text
224 images
```

The target is:

```text
1 connected sparse model
all moving-camera frames registered
all static cameras registered
continuous and plausible camera trajectory
coherent sparse point cloud
```

---

# Pipeline

## Dataset Preparation

The AP03 dataset is prepared with:

```bash
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py
```

The script supports an alternative moving-camera image directory:

```bash
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
  --moving-dir results/bus_real_data/01_marker_direct_relay_multimarker_multichain/03_moving_camera_sequence_improved/images
```

The static images are still loaded from the shared baseline dataset.

## Sparse Reconstruction

The sparse reconstruction is generated with:

```bash
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse_grouped.py
```

The matcher can be selected with:

```bash
--matcher exhaustive
```

or:

```bash
--matcher sequential
```

The script assigns:

- one COLMAP camera ID to each physical static camera,
- one shared COLMAP camera ID to all moving-camera frames,
- calibrated intrinsics from the camera-info files,
- and disables intrinsic refinement during mapping.

---

# Matcher Configuration

## Exhaustive Matcher

The exhaustive matcher compares all images against all other images.

Advantages:

- robust global connectivity,
- non-local image correspondences,
- better registration of static cameras,
- ability to bridge weak moving-camera sections.

Disadvantages:

- higher computational cost,
- worse scalability for very large datasets.

## Sequential Matcher

The sequential matcher compares temporally nearby images.

Current matching configuration:

```text
SequentialMatching.overlap = 20
SequentialMatching.quadratic_overlap = 1
```

Advantages:

- lower computational cost,
- suitable for ordered image sequences,
- better scalability.

Disadvantages:

- depends strongly on local sequence overlap,
- can fragment the reconstruction,
- static images may not be matched to spatially relevant moving frames.

---

# Original Route: Matcher Comparison

## Exhaustive Matcher

| Metric | Value |
|---|---:|
| Sparse models | 1 |
| Registered images | 268 / 274 |
| Cameras | 5 |
| 3D points | 12,735 |
| Observations | 106,495 |
| Mean track length | 8.36 |
| Mean reprojection error | 0.702 px |

The exhaustive matcher reconstructed almost the complete sequence as one connected model.

## Sequential Matcher Baseline

The sequential matcher initially produced two disconnected sparse models.

### Model 0

| Metric | Value |
|---|---:|
| Registered images | 143 |
| Cameras | 3 |
| 3D points | 6,158 |
| Observations | 46,731 |
| Mean track length | 7.59 |
| Mean reprojection error | 0.564 px |

### Model 1

| Metric | Value |
|---|---:|
| Registered images | 124 |
| Cameras | 3 |
| 3D points | 6,137 |
| Observations | 52,323 |
| Mean track length | 8.53 |
| Mean reprojection error | 0.781 px |

The missing moving-camera frames were:

```text
moving_frame_0141.png
moving_frame_0142.png
moving_frame_0143.png
moving_frame_0144.png
moving_frame_0145.png
moving_frame_0146.png
moving_frame_0147.png
```

Model 0 ended at `moving_frame_0140.png`.

Model 1 started at `moving_frame_0148.png`.

---

# Failure-Region Analysis

## SIFT Keypoints

Before the critical region, frames usually contained approximately **200–350 keypoints**.

Inside the critical region, the feature count dropped to approximately **92–149 keypoints**.

| Frame | Keypoints |
|---|---:|
| 0137 | 334 |
| 0138 | 237 |
| 0139 | 206 |
| 0140 | 149 |
| 0141 | 121 |
| 0142 | 140 |
| 0143 | 140 |
| 0144 | 124 |
| 0145 | 99 |
| 0146 | 92 |
| 0147 | 142 |
| 0148 | 130 |
| 0151 | 263 |
| 0152 | 319 |
| 0153 | 382 |

## Verified Matches Between Consecutive Frames

Before the failure region, consecutive image pairs usually contained approximately **59–109 verified matches**.

Inside the failure region, most consecutive pairs contained only **18–39 verified matches**.

| Image pair | Verified matches |
|---|---:|
| 0138–0139 | 59 |
| 0139–0140 | 39 |
| 0140–0141 | 30 |
| 0141–0142 | 25 |
| 0142–0143 | 116 |
| 0143–0144 | 37 |
| 0144–0145 | 25 |
| 0145–0146 | 18 |
| 0146–0147 | 25 |
| 0147–0148 | 28 |
| 0148–0149 | 29 |
| 0152–0153 | 96 |
| 0153–0154 | 107 |
| 0154–0155 | 118 |

The pair `0142–0143` is a local exception with 116 verified matches.

The split was therefore not caused by zero local matches. Instead, the critical section had weak multi-view connectivity and too few reliable 2D-to-3D correspondences.

---

# Mapper Threshold Ablation

The tested mapper parameters were:

```text
Mapper.min_num_matches
Mapper.abs_pose_min_num_inliers
Mapper.abs_pose_min_inlier_ratio
```

The AP03 script used:

```text
Mapper.min_num_matches = 8
```

## Baseline Thresholds

```text
Mapper.abs_pose_min_num_inliers = 30
Mapper.abs_pose_min_inlier_ratio = 0.25
```

| Metric | Value |
|---|---:|
| Sparse models | 2 |
| Registered images | 143 + 124 |
| Missing critical frames | 0141–0147 |

## Moderate Thresholds

```text
Mapper.abs_pose_min_num_inliers = 20
Mapper.abs_pose_min_inlier_ratio = 0.20
```

| Metric | Model 0 | Model 1 |
|---|---:|---:|
| Registered images | 148 | 125 |
| 3D points | 6,212 | 6,148 |
| Observations | 47,051 | 52,344 |
| Mean track length | 7.57 | 8.51 |
| Mean reprojection error | 0.558 px | 0.771 px |

Only `moving_frame_0146.png` remained unregistered.

## Relaxed Thresholds

```text
Mapper.abs_pose_min_num_inliers = 15
Mapper.abs_pose_min_inlier_ratio = 0.15
```

| Metric | Value |
|---|---:|
| Sparse models | 1 |
| Registered images | 273 / 274 |
| Cameras | 4 |
| 3D points | 12,000 |
| Observations | 95,815 |
| Mean track length | 7.98 |
| Mean reprojection error | 0.678 px |

All critical moving-camera frames were registered.

## Threshold Comparison

| Configuration | Sparse models | Registered images | Missing critical frames |
|---|---:|---:|---|
| `30 / 0.25` | 2 | 143 + 124 | 0141–0147 |
| `20 / 0.20` | 2 | 148 + 125 | 0146 |
| `15 / 0.15` | 1 | 273 | none |

The original route therefore required a relatively permissive configuration to obtain one connected moving-camera reconstruction.

---

# Moving-Camera Route Improvement

## Original Route Problem

The original failure frames 0140–0150 corresponded to interpolation segment 8:

```text
kf_008_cam5_bridge_left
to
kf_009_cam5_far_left_turn
```

This segment combined approximately:

```text
2.0 m translation
2.82 rad yaw change
```

The camera therefore translated while rotating by approximately 162 degrees.

## Improved Route

Additional route keyframes were inserted:

```text
kf_008_cam5_bridge_left
kf_008a_hold_overlap
kf_008b_gradual_turn_1
kf_008c_gradual_turn_2
kf_008d_cam5_bridge
kf_009_cam5_far_left_turn
```

The improved keyframes are stored in:

```text
src/calib_lab/bus_real_data/config/moving_camera_route_keyframes_improved.json
```

The interpolated route is stored in:

```text
src/calib_lab/bus_real_data/config/moving_camera_route_interpolated_improved.json
```

The improved route contains:

```text
220 frames
```

The captured sequence is stored in:

```text
results/bus_real_data/01_marker_direct_relay_multimarker_multichain/03_moving_camera_sequence_improved
```

---

# Improved Route Results

## Sequential Matching with Strict Thresholds

Configuration:

```text
Mapper.min_num_matches = 8
Mapper.abs_pose_min_num_inliers = 30
Mapper.abs_pose_min_inlier_ratio = 0.25
```

| Metric | Model 0 | Model 1 |
|---|---:|---:|
| Registered images | 105 | 110 |
| Cameras | 1 | 3 |
| 3D points | 5,097 | 5,388 |
| Observations | 32,764 | 42,009 |
| Mean track length | 6.43 | 7.80 |
| Mean reprojection error | 0.489 px | 0.757 px |

Missing images:

```text
moving_frame_0033.png
moving_frame_0034.png
moving_frame_0035.png
moving_frame_0036.png
moving_frame_0109.png
moving_frame_0110.png
moving_frame_0111.png
static_cam_edge_0.png
static_cam_edge_1.png
```

The improved route still fragmented under the strict thresholds.

## Sequential Matching with Moderate Thresholds

Configuration:

```text
Mapper.min_num_matches = 8
Mapper.abs_pose_min_num_inliers = 20
Mapper.abs_pose_min_inlier_ratio = 0.20
```

| Metric | Value |
|---|---:|
| Sparse models | 1 |
| Registered images | 222 / 224 |
| Registered moving frames | 220 / 220 |
| Registered static cameras | 2 / 4 |
| Cameras | 3 |
| 3D points | 10,608 |
| Observations | 75,159 |
| Mean track length | 7.09 |
| Mean reprojection error | 0.658 px |

Registered static cameras:

```text
static_cam_edge_3.png
static_cam_edge_5.png
```

Missing static cameras:

```text
static_cam_edge_0.png
static_cam_edge_1.png
```

This result shows that the improved route solved the moving-camera connectivity problem with only moderately relaxed thresholds.

---

# Exhaustive Matching on the Improved Route

Configuration:

```text
Mapper.min_num_matches = 8
Mapper.abs_pose_min_num_inliers = 20
Mapper.abs_pose_min_inlier_ratio = 0.20
```

| Metric | Value |
|---|---:|
| Sparse models | 1 |
| Registered images | 224 / 224 |
| Registered moving frames | 220 / 220 |
| Registered static cameras | 4 / 4 |
| Cameras | 5 |
| 3D points | 11,029 |
| Observations | 81,403 |
| Mean track length | 7.38 |
| Mean observations per image | 363.41 |
| Mean reprojection error | 0.663 px |

All static cameras were registered:

```text
static_cam_edge_0.png
static_cam_edge_1.png
static_cam_edge_3.png
static_cam_edge_5.png
```

This is the first configuration that registered every image in the improved dataset.

---

# Final Comparison

| Route | Matcher | Thresholds | Models | Moving frames | Static cameras | Total images |
|---|---|---:|---:|---:|---:|---:|
| Original | Exhaustive | original | 1 | partial | 4 / 4 | 268 / 274 |
| Original | Sequential | `30 / 0.25` | 2 | 263 / 270 | 4 / 4 | 267 / 274 |
| Original | Sequential | `20 / 0.20` | 2 | 269 / 270 | 4 / 4 | 273 across two models |
| Original | Sequential | `15 / 0.15` | 1 | 270 / 270 | 3 / 4 | 273 / 274 |
| Improved | Sequential | `30 / 0.25` | 2 | 213 / 220 | 2 / 4 | 215 / 224 |
| Improved | Sequential | `20 / 0.20` | 1 | 220 / 220 | 2 / 4 | 222 / 224 |
| Improved | Exhaustive | `20 / 0.20` | 1 | 220 / 220 | 4 / 4 | 224 / 224 |

---

# Main Findings

1. The original sequential reconstruction failed in a feature-poor and low-connectivity section.

2. The failure was not caused by a complete absence of matches.

3. Relaxing absolute-pose thresholds gradually improved the original-route reconstruction.

4. The original route required `15 / 0.15` to produce one connected moving-camera model.

5. The improved route registered all 220 moving-camera frames with the less permissive `20 / 0.20` configuration.

6. Sequential matching still missed two static cameras because temporal ordering did not guarantee useful static-to-moving correspondences.

7. Exhaustive matching on the improved route registered all 224 images.

8. The final target result was achieved:

```text
1 connected model
220 / 220 moving frames
4 / 4 static cameras
224 / 224 total images
```

---

# Conclusion

The original reconstruction split was caused by a combination of:

- weak scene texture,
- low local match counts,
- limited multi-view connectivity,
- and strict absolute-pose registration thresholds.

Threshold relaxation alone could solve the original moving-camera split, but required a comparatively permissive configuration.

The improved route provided better moving-camera connectivity and allowed all moving frames to be reconstructed with moderate thresholds.

Sequential matching remained insufficient for registering all static cameras because the static images required non-local correspondences to spatially relevant moving-camera views.

The final best configuration is:

```text
Improved moving-camera route
Exhaustive matcher
Mapper.min_num_matches = 8
Mapper.abs_pose_min_num_inliers = 20
Mapper.abs_pose_min_inlier_ratio = 0.20
```

This configuration produced:

```text
1 connected sparse model
224 / 224 registered images
220 / 220 moving-camera frames
4 / 4 static cameras
11,029 3D points
81,403 observations
0.663 px mean reprojection error
```

Therefore, the improved route combined with exhaustive matching is the preferred AP03 configuration.
