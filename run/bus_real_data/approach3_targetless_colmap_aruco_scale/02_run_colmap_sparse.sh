#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

AP3_ROOT="results/bus_real_data/03_targetless_colmap_aruco_scale"
DATASET_ROOT="$AP3_ROOT/01_colmap_dataset"
IMAGE_DIR="$DATASET_ROOT/images"

RUN_ROOT="$AP3_ROOT/02_colmap_sparse"
DB="$RUN_ROOT/database.db"
SPARSE_ROOT="$RUN_ROOT/sparse"
TXT_ROOT="$RUN_ROOT/sparse_txt"

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap not found."
  echo "Install with:"
  echo "  apt-get update && apt-get install -y colmap"
  exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
  echo "[ERROR] Missing image dir: $IMAGE_DIR"
  echo "Run first:"
  echo "  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py"
  exit 1
fi

rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT" "$SPARSE_ROOT" "$TXT_ROOT"

echo "=== AP03 COLMAP Phase 1 ==="
echo "[INFO] Image dir: $IMAGE_DIR"
echo "[INFO] Images: $(find "$IMAGE_DIR" -maxdepth 1 -type f | wc -l)"
echo "[INFO] Database: $DB"

echo
echo "=== 1/4 Feature extraction ==="
colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 0 \
  --SiftExtraction.use_gpu 0

echo
echo "=== 2/4 Exhaustive matching ==="
colmap exhaustive_matcher \
  --database_path "$DB" \
  --SiftMatching.use_gpu 0

echo
echo "=== 3/4 Sparse mapping ==="
colmap mapper \
  --database_path "$DB" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE_ROOT" \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 \
  --Mapper.min_num_matches 8

echo
echo "=== 4/4 Convert models to TXT ==="
shopt -s nullglob
for model_dir in "$SPARSE_ROOT"/*; do
  if [ -d "$model_dir" ]; then
    model_name="$(basename "$model_dir")"
    out_dir="$TXT_ROOT/$model_name"
    mkdir -p "$out_dir"

    colmap model_converter \
      --input_path "$model_dir" \
      --output_path "$out_dir" \
      --output_type TXT

    echo "[OK] converted model $model_name to $out_dir"
  fi
done

echo
echo "[OK] AP03 COLMAP sparse run complete."
echo "[INFO] Sparse models:"
find "$SPARSE_ROOT" -maxdepth 1 -mindepth 1 -type d | sort || true
echo
echo "[INFO] TXT models:"
find "$TXT_ROOT" -maxdepth 1 -mindepth 1 -type d | sort || true
