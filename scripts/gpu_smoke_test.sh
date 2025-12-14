#!/bin/bash
# Quick GPU smoke test for MaxText with PLAN_2 config
# Run 10 steps to validate everything works before full training

set -euo pipefail

echo "=========================================="
echo "GPU Smoke Test (10 steps)"
echo "=========================================="
echo ""

# Activate venv
source .venv/bin/activate

# Set environment
export PYTHONPATH="src:${PYTHONPATH:-}"
export DECOUPLE_GCLOUD=TRUE

# Check GPU
echo "GPU Status:"
nvidia-smi -L 2>/dev/null || echo "WARNING: nvidia-smi not available"
echo ""

# Check JAX can see GPU
echo "JAX Devices:"
python -c "import jax; print(jax.devices())"
echo ""

# Run smoke test
echo "Starting MaxText training (10 steps on GPU)..."
echo ""

python -m MaxText.train \
  src/MaxText/configs/latency_network.yml \
  hardware=gpu \
  steps=10 \
  per_device_batch_size=4 \
  run_name=smoke_test_gpu \
  grain_worker_count=4 \
  enable_checkpointing=false \
  log_period=1

echo ""
echo "=========================================="
echo "Smoke test complete!"
echo "=========================================="
