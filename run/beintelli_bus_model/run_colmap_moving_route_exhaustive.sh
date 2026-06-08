#!/usr/bin/env bash
set -eo pipefail

DATASET="${1:-results/beintelli_bus_model/colmap/moving_route_poc}"

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap command not found. Install COLMAP first:"
  echo "        sudo apt update && sudo apt install colmap -y"
  exit 1
fi

cd "$DATASET"

rm -f database.db
rm -rf sparse sparse_txt
mkdir -p sparse

echo "[INFO] COLMAP version:"
colmap -h | head -5 || true

echo ""
echo "[INFO] Dataset: $(pwd)"
echo "[INFO] Images:  $(pwd)/images"
echo "[INFO] Image count:"
ls images | wc -l

# GPU option names differ between COLMAP versions.
FE_GPU_ARGS=()
if colmap feature_extractor -h 2>&1 | grep -q "SiftExtraction.use_gpu"; then
  FE_GPU_ARGS=(--SiftExtraction.use_gpu 0)
elif colmap feature_extractor -h 2>&1 | grep -q "FeatureExtraction.use_gpu"; then
  FE_GPU_ARGS=(--FeatureExtraction.use_gpu 0)
fi

MATCH_GPU_ARGS=()
if colmap sequential_matcher -h 2>&1 | grep -q "SiftMatching.use_gpu"; then
  MATCH_GPU_ARGS=(--SiftMatching.use_gpu 0)
elif colmap sequential_matcher -h 2>&1 | grep -q "FeatureMatching.use_gpu"; then
  MATCH_GPU_ARGS=(--FeatureMatching.use_gpu 0)
fi

echo ""
echo "[INFO] Feature GPU args: ${FE_GPU_ARGS[*]:-(none)}"
echo "[INFO] Match GPU args:   ${MATCH_GPU_ARGS[*]:-(none)}"

echo ""
echo "[INFO] Running COLMAP feature_extractor..."
colmap feature_extractor \
  --database_path database.db \
  --image_path images \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params "320,320,320,240" \
  "${FE_GPU_ARGS[@]}"

echo ""
echo "[INFO] Running COLMAP exhaustive_matcher..."
colmap exhaustive_matcher \
  --database_path database.db \
  "${MATCH_GPU_ARGS[@]}"

echo ""
echo "[INFO] Running COLMAP mapper..."
colmap mapper \
  --database_path database.db \
  --image_path images \
  --output_path sparse

if [ ! -d sparse/0 ]; then
  echo "[ERROR] COLMAP did not create sparse/0 reconstruction."
  echo "[INFO] Existing sparse folders:"
  find sparse -maxdepth 2 -type d | sort
  exit 2
fi

mkdir -p sparse_txt

echo ""
echo "[INFO] Converting sparse model to TXT..."
colmap model_converter \
  --input_path sparse/0 \
  --output_path sparse_txt \
  --output_type TXT

echo ""
echo "[INFO] Model analyzer:"
colmap model_analyzer \
  --path sparse/0 || true

echo ""
echo "[OK] COLMAP reconstruction finished."
echo "TXT model:"
echo "  $(pwd)/sparse_txt"
echo "Important file:"
echo "  $(pwd)/sparse_txt/images.txt"
