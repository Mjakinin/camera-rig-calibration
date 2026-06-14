# Image Degradation Ablation Summary

| Run | Mode | Value | Frames | Total detections | Unique IDs | Missing IDs | Max empty run | Notes |
|---|---|---:|---:|---:|---|---|---:|---|
| baseline | none | - | TBD | TBD | TBD | TBD | TBD | original images |
| blur_k3 | gaussian_blur | 3 | 131 | 218 | 0-13 | none | 6 | mild Gaussian blur |
| blur_k5 | gaussian_blur | 5 | 131 | 222 | 0-13 | none | 6 | moderate Gaussian blur |
| blur_k9 | gaussian_blur | 9 | 131 | 220 | 0-13 | none | 6 | strong Gaussian blur |
| brightness_150 | brightness | 1.5 | 131 | 221 | 0-13 | none | 6 | brighter / overexposure-like |
| motion_blur_15 | motion_blur | 15 | 131 | 91 | 0,1,3,5,8,9,10,11,12,13 | 2,4,6,7 | 24 | strong directed motion blur |

## Interpretation

Gaussian blur with kernel sizes 3, 5, and 9 did not reduce marker ID coverage in the moving-camera sequence. All 14 expected marker IDs were still detected, and the maximum consecutive markerless gap remained 6 frames.

Brightness increase to 150% also preserved full marker ID coverage.

Directed motion blur with length 15 caused a strong degradation:
- total detections dropped to 91,
- marker IDs 2, 4, 6, and 7 disappeared completely,
- the maximum markerless gap increased to 24 frames.

This indicates that the pipeline is more sensitive to directional motion blur than to symmetric Gaussian blur or moderate brightness increase.
