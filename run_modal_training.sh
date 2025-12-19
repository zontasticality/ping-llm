#!/usr/bin/env bash
# Run training on Modal with network backend

modal run scripts/train/modal_train_with_wandb_sync.py::run \
  --run-name network_backend_test \
  --steps 5000 \
  --batch-size 128 \
  --wandb-project ping-llm-network-backend

echo ""
echo "Training started on Modal!"
echo "Monitor at: https://wandb.ai (check your project: ping-llm-network-backend)"
