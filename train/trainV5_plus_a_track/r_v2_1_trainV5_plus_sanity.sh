#!/usr/bin/env bash
set -euo pipefail

# Engineering sanity draft only. Keep GPU1 idle.
cd /home/ubuntu/ssd_work/projects/stable-fast-3d
CUDA_VISIBLE_DEVICES=0 python scripts/train_material_refiner.py \
  --config configs/material_refine_train_r_v2_1_view_aware.yaml \
  --train-manifest train/trainV5_plus_a_track/trainV5_training_pairs.json \
  --val-manifest train/trainV5_plus_a_track/trainV5_training_pairs.json \
  --split-strategy manifest \
  --train-split train \
  --val-split val \
  --output-dir output/material_refine_trainV5_plus_a_track_sanity \
  --cuda-device-index 0 \
  --epochs 1 \
  --max-train-steps 8 \
  --max-validation-batches 2 \
  --batch-size 2 \
  --val-batch-size 2 \
  --num-workers 0 \
  --log-every 1 \
  --wandb-mode disabled
