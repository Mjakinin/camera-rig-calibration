# COLMAP Matcher Comparison

This folder compares the AP03 reconstruction using two different COLMAP matching strategies.

## Exhaustive matcher

```
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse_grouped.py \
    --matcher exhaustive
```

Results were copied to

```
comparison/exhaustive/
```

---

## Sequential matcher

```
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse_grouped.py \
    --matcher sequential
```

Results were copied to

```
comparison/sequential/
```

The purpose is to evaluate which matching strategy is more suitable for the moving-camera setup used in AP03.