#!/usr/bin/env bash
set -e

# ==========================================
# 用法:
# bash preprocess.sh /path/to/scene /path/to/sam_vit_h_4b8939.pth
#
# scene 目录建议至少包含:
#   scene/
#   └── images/
#       ├── 0000.png
#       ├── 0001.png
#       └── ...
# ==========================================

SCENE_ROOT=$1
SAM_CKPT=${2:-""}

if [ -z "$SCENE_ROOT" ]; then
  echo "Usage: bash preprocess.sh /path/to/scene [/path/to/sam_ckpt]"
  exit 1
fi

echo "[Stage 1/3] Extract raw language features..."
if [ -n "$SAM_CKPT" ]; then
  python preprocess/extract_and_compress.py \
    --mode extract \
    --scene_root "$SCENE_ROOT" \
    --image_dirname images \
    --raw_dirname language_features_raw \
    --clip_model ViT-B-16 \
    --clip_pretrained laion2b_s34b_b88k \
    --feature_hw 64 64 \
    --use_sam \
    --sam_ckpt "$SAM_CKPT" \
    --sam_type vit_h \
    --max_regions 32
else
  python preprocess/extract_and_compress.py \
    --mode extract \
    --scene_root "$SCENE_ROOT" \
    --image_dirname images \
    --raw_dirname language_features_raw \
    --clip_model ViT-B-16 \
    --clip_pretrained laion2b_s34b_b88k \
    --feature_hw 64 64 \
    --max_regions 32
fi

echo "[Stage 2/3] Train language autoencoder..."
python preprocess/train_autoencoder.py \
  --scene_root "$SCENE_ROOT" \
  --raw_dirname language_features_raw \
  --cache_name ae_train_cache.npy \
  --max_samples_per_file 20000 \
  --input_dim 512 \
  --latent_dim 8 \
  --encoder_hidden 256 128 64 \
  --decoder_hidden 64 128 256 \
  --epochs 25 \
  --batch_size 32768 \
  --lr 7e-4 \
  --weight_decay 1e-5 \
  --num_workers 8

echo "[Stage 3/3] Compress raw language features..."
python preprocess/extract_and_compress.py \
  --mode compress \
  --scene_root "$SCENE_ROOT" \
  --raw_dirname language_features_raw \
  --ae_ckpt "$SCENE_ROOT/ae_ckpt/best.pth"

echo "[Done] Preprocessing finished."