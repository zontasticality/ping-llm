#!/bin/bash
# Wrapper script for running MaxText on CPU with proper environment variables
# Usage: ./scripts/train/run_maxtext_cpu_test.sh [additional args...]

set -e

# Set required environment variables for single-node CPU training
export DECOUPLE_GCLOUD=TRUE

# Run MaxText
.venv/bin/python -m MaxText.train src/MaxText/configs/latency_network.yml \
  run_name=phase2_smoke_test \
  steps=2 \
  per_device_batch_size=1 \
  "$@"
