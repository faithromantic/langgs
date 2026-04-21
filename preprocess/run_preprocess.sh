#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-}
SAM_CKPT=${2:-}

if [ -z "$DATASET" ]; then
  echo "Usage: bash preprocess/run_preprocess.sh /path/to/dataset [/path/to/sam_vit_h_4b8939.pth]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_CKPT="$SCRIPT_DIR/checkpoints/ae_dim3.pth"

echo "[1/3] Extracting 512-d language features to $DATASET/language_features"
if [ -n "$SAM_CKPT" ]; then
  python "$SCRIPT_DIR/extract_language_features.py" \
    --dataset "$DATASET" \
    --image_dirname images \
    --output_dirname language_features \
    --use_sam \
    --sam_ckpt "$SAM_CKPT"
else
  python "$SCRIPT_DIR/extract_language_features.py" \
    --dataset "$DATASET" \
    --image_dirname images \
    --output_dirname language_features
fi

echo "[2/3] Training 512-3-512 autoencoder to $AE_CKPT"
python "$SCRIPT_DIR/train_autoencoder.py" \
  --dataset "$DATASET" \
  --raw_dirname language_features \
  --checkpoint_dir "$SCRIPT_DIR/checkpoints" \
  --output_name ae_dim3.pth

echo "[3/3] Encoding language features to $DATASET/language_features_dim3"
python "$SCRIPT_DIR/encode_language_features.py" \
  --dataset "$DATASET" \
  --input_dirname language_features \
  --output_dirname language_features_dim3 \
  --checkpoint "$AE_CKPT"

echo "[Done] Preprocessing finished."
