# Image Degradation Ablation Summary

| Run | Mode | Value | Frames | Total detections | Unique IDs | Missing IDs | Max empty run | Notes |
|---|---|---:|---:|---:|---|---|---:|---|
| baseline | none | - | TBD | TBD | TBD | TBD | TBD | original images |
| blur_k3 | gaussian_blur | 3 | 131 | 218 | 0-13 | none | 6 | mild Gaussian blur |
| blur_k5 | gaussian_blur | 5 | 131 | 222 | 0-13 | none | 6 | moderate Gaussian blur |
| blur_k9 | gaussian_blur | 9 | 131 | 220 | 0-13 | none | 6 | strong Gaussian blur |
| brightness_150 | brightness | 1.5 | 131 | 221 | 0-13 | none | 6 | brighter / overexposure-like |
| motion_blur_7 | motion_blur | 7 | 131 | 163 | 0-13 | none | 6 | moderate directed motion blur |
| motion_blur_11 | motion_blur | 11 | 131 | 133 | 0,1,2,3,5,6,7,8,9,10,11,12,13 | 4 | 7 | transition point; loses marker 4 |
| motion_blur_15 | motion_blur | 15 | 131 | 91 | 0,1,3,5,8,9,10,11,12,13 | 2,4,6,7 | 24 | strong directed motion blur |

## Interpretation

Gaussian blur with kernel sizes 3, 5, and 9 did not reduce marker ID coverage in the moving-camera sequence. All 14 expected marker IDs were still detected, and the maximum consecutive markerless gap remained 6 frames.

Brightness increase to 150% also preserved full marker ID coverage.

Directed motion blur caused the clearest degradation:
- motion_blur_7 reduced total detections to 163 but preserved all marker IDs,
- motion_blur_11 reduced total detections to 133 and lost marker 4 completely,
- motion_blur_15 reduced total detections to 91 and lost marker IDs 2, 4, 6, and 7,
- the maximum markerless gap increased to 24 frames for motion_blur_15.

This indicates that the pipeline is significantly more sensitive to directional motion blur than to symmetric Gaussian blur or moderate brightness increase.

## Relay relevance

Marker 4 is the target-side anchor for the cam_edge_3 -> cam_edge_0 relay evaluation. Losing marker 4 under motion_blur_11 means that this relay chain cannot be evaluated with the current marker-anchor selection.

Marker 7 was one of the strongest root-side anchors in the baseline COLMAP relay evaluation. Losing marker 7 under motion_blur_15 removes a high-quality candidate chain for cam_edge_3 -> cam_edge_5.
