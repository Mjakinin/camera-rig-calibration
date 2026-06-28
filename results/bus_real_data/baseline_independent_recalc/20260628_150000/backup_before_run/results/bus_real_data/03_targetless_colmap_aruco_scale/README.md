# AP03 Results: Targetless COLMAP + ArUco Scale

AP03 result stages:

```text
01_colmap_dataset/
02_colmap_sparse/
03_reconstruction_inspection/
06_triangulated_ref_aruco_registration/
```

Primary final report:

```text
06_triangulated_ref_aruco_registration/AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt
```

Validated result:

```text
registered images: 208
sparse 3D points: 9401
mean translation error: 7.501966 cm
mean rotation error:    0.367828 deg
```

The `--reuse-existing` mode in the AP03 runner can regenerate the final report without rerunning COLMAP.
