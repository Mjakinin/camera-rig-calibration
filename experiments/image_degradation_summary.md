# Image Degradation Ablation Summary

| Run | Mode | Value | Frames | Total detections | Unique IDs | Missing IDs | Max empty run | Notes |
|---|---|---:|---:|---:|---|---|---:|---|
| baseline | none | - | TBD | TBD | TBD | TBD | TBD | original images |
| blur_k3 | gaussian_blur | 3 | 131 | 218 | 0-13 | none | 6 | mild blur |
| blur_k5 | gaussian_blur | 5 | 131 | 222 | 0-13 | none | 6 | moderate blur |
| blur_k9 | gaussian_blur | 9 | 131 | 220 | 0-13 | none | 6 | strong Gaussian blur |

## Preliminary interpretation

Gaussian blur with kernel sizes 3, 5, and 9 did not reduce marker ID coverage in the moving-camera sequence. All 14 expected marker IDs were still detected, and the maximum consecutive markerless gap remained 6 frames. Detection counts stayed within a narrow range of 218-222 detections.
