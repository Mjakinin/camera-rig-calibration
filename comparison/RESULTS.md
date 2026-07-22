# COLMAP Matcher Comparison Results

## Dataset

The AP03 dataset contains **270 moving-camera frames** and four static cameras:

- `cam_edge_0`
- `cam_edge_1`
- `cam_edge_3`
- `cam_edge_5`

The experiment compares COLMAP's **exhaustive matcher** and **sequential matcher** with respect to reconstruction completeness and connectivity.

---

## Exhaustive Matcher

The exhaustive matcher produced one connected sparse model.

| Metric | Value |
|---|---:|
| Registered images | 268 |
| Cameras | 5 |
| 3D points | 12,735 |
| Observations | 106,495 |
| Mean track length | 8.36 |
| Mean reprojection error | 0.702 px |

The result contains almost the complete sequence in a single reconstruction.

---

## Sequential Matcher

The sequential matcher produced two disconnected sparse models.

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

Together, the two models contain 267 registered images. However, they do not form one continuous reconstruction.

---

## Reconstruction Split

The sequential reconstruction is interrupted at:

- `moving_frame_0141.png`
- `moving_frame_0142.png`
- `moving_frame_0143.png`
- `moving_frame_0144.png`
- `moving_frame_0145.png`
- `moving_frame_0146.png`
- `moving_frame_0147.png`

Model 0 ends at `moving_frame_0140.png`.

Model 1 starts at `moving_frame_0148.png`.

Therefore, connectivity is lost in the transition between frames **0140 and 0148**.

The exhaustive matcher is able to register `moving_frame_0147.png`, while the sequential matcher cannot. This suggests that non-local image correspondences outside the immediate temporal neighborhood help bridge the weak section.

---

## Feature Analysis Around the Failure Region

### SIFT Keypoints

Before the critical region, frames usually contain about **200–350 keypoints**.

| Frame | Keypoints |
|---|---:|
| 0130 | 347 |
| 0131 | 296 |
| 0134 | 269 |
| 0137 | 334 |
| 0139 | 206 |

In the critical region, the feature count drops substantially:

| Frame | Keypoints |
|---|---:|
| 0140 | 149 |
| 0141 | 121 |
| 0142 | 140 |
| 0143 | 140 |
| 0144 | 124 |
| 0145 | 99 |
| 0146 | 92 |
| 0147 | 142 |
| 0148 | 130 |
| 0149 | 115 |
| 0150 | 144 |

Afterwards, the feature count rises again:

| Frame | Keypoints |
|---|---:|
| 0151 | 263 |
| 0152 | 319 |
| 0153 | 382 |
| 0154 | 380 |
| 0155 | 441 |

This confirms a pronounced feature-poor section around frames 0140–0150.

---

## Verified Matches Between Consecutive Frames

Before the critical region, neighboring image pairs usually contain approximately **59–109 verified matches**.

| Image pair | Verified matches |
|---|---:|
| 0130–0131 | 109 |
| 0131–0132 | 92 |
| 0133–0134 | 101 |
| 0134–0135 | 95 |
| 0138–0139 | 59 |

In the critical region, the count drops to mostly **18–39 verified matches**:

| Image pair | Verified matches |
|---|---:|
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
| 0149–0150 | 29 |
| 0150–0151 | 41 |

The pair `0142–0143` is a local exception with 116 verified matches. However, this isolated strong pair is not sufficient to maintain stable multi-view connectivity across the complete low-feature section.

After frame 0151, the match count increases again:

| Image pair | Verified matches |
|---|---:|
| 0151–0152 | 51 |
| 0152–0153 | 96 |
| 0153–0154 | 107 |
| 0154–0155 | 118 |

---

## Interpretation

The split is not caused by a complete absence of feature matches. All consecutive frames in the critical interval still have geometrically verified correspondences.

However, the section contains:

- significantly fewer SIFT keypoints,
- mostly only 18–39 verified matches between neighboring images,
- likely fewer persistent feature tracks over multiple views,
- and therefore too few reliable 2D-to-3D correspondences for incremental image registration.

This creates a **low-connectivity section** in the image graph.

The sequential matcher mainly relies on temporally local image pairs. When this local connectivity becomes too weak, COLMAP cannot reliably register the intermediate images into the existing model and starts a second independent reconstruction.

The exhaustive matcher evaluates all image pairs. It can therefore introduce additional non-local correspondences and bridge the weak section.

The lower reprojection error of sequential model 0 does not mean that sequential matching performs better overall. Reprojection error only measures consistency inside the reconstructed model and does not penalize missing frames or model fragmentation.

---

## Possible Improvements

Potential improvements for sequential matching include:

- capturing denser frames in the critical route section,
- keeping textured and previously observed scene regions visible for longer,
- avoiding views dominated by smooth, dark, reflective, or weakly textured surfaces,
- reducing motion blur,
- introducing camera translation instead of mainly rotational motion,
- increasing route overlap before entering the weak section,
- or using a retrieval-based matcher when exhaustive matching becomes too expensive.

Denser frames may help, but they are not guaranteed to solve the issue if the scene remains feature-poor.

---

## Conclusion

For the AP03 moving-camera dataset:

- Exhaustive matching reconstructs almost the complete sequence as one connected sparse model.
- Sequential matching fragments the sequence into two disconnected models.
- Seven consecutive frames, 0141–0147, are missing from the sequential reconstruction.
- The split coincides with a strong reduction in SIFT keypoints and verified local matches.
- Local pairwise matches still exist, but they are insufficient for stable multi-view registration.
- Exhaustive matching remains connected because it can use additional non-local image correspondences.
- Sequential matching is computationally cheaper, but less robust for this dataset.
- For the current AP03 sequence, exhaustive matching is the preferred option.
