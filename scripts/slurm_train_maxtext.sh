#!/bin/bash
#SBATCH --job-name=latency-maxtext
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# SLURM script for training MaxText model on network measurement data
#
# Usage:
#   sbatch scripts/slurm_train_maxtext.sh
#
# Prerequisites:
# 1. Data sharded into data/sharded/{train,val,test}/
# 2. MaxText installed in your environment
# 3. Tokenization integration with Grain (see PLAN.md Phase 1)

set -euo pipefail

echo "========================================"
echo "MaxText Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Load modules (adjust for your cluster)
# module load cuda/12.0
# module load cudnn/8.9

# Activate virtual environment
# source ~/venvs/maxtext/bin/activate

# JAX/XLA settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"

# Data and output directories
DATA_DIR="${DATA_DIR:-data/sharded}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/maxtext_runs}"
RUN_NAME="${RUN_NAME:-latency_${SLURM_JOB_ID}}"

# Create output directory
mkdir -p "$OUT_DIR"
mkdir -p logs

echo "Data directory: $DATA_DIR"
echo "Output directory: $OUT_DIR"
echo "Run name: $RUN_NAME"
echo ""

# Training command
# Note: Adjust paths based on your MaxText installation
python -m MaxText.train \
  maxtext/configs/latency_parquet.yml \
  run_name="$RUN_NAME" \
  base_output_directory="$OUT_DIR" \
  hardware=gpu \
  per_device_batch_size=8 \
  grain_train_files="$DATA_DIR/train/*.parquet" \
  grain_eval_files="$DATA_DIR/val/*.parquet" \
  grain_worker_count=16 \
  steps=200000 \
  eval_interval=1000 \
  checkpoint_period=5000

echo ""
echo "========================================"
echo "Training completed"
echo "========================================"
