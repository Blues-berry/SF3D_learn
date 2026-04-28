#!/usr/bin/env bash
set -euo pipefail

# Manual draft only. Review manifests/readiness before removing this guard.
echo "TrainV5 R-v2.1 sanity command draft only; not auto-starting training."
exit 1

# Example draft, adjust config/CLI after human review:
# cd /home/ubuntu/ssd_work/projects/stable-fast-3d
# /home/ubuntu/ssd_work/conda_envs/sf3d/bin/python scripts/train_material_refiner.py \
#   --config configs/material_refine_train_r_v2_1_view_aware.yaml \
#   --train-manifest train/trainV5_initial/trainV5_training_pairs.json
