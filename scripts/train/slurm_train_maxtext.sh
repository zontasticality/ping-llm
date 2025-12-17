#!/bin/bash
#SBATCH --job-name=ping-llm-plan2
#SBATCH --partition=gpu          # Override with: sbatch --partition=<gpu-partition> scripts/...
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# SLURM script for training PLAN_2 network measurement model
#
# Usage:
#   sbatch scripts/train/slurm_train_maxtext.sh
#
# Prerequisites:
# 1. Dataset sharded into data/sharded/{train,test}/ (180 train + 20 test)
# 2. Virtual environment with MaxText + dependencies
# 3. PLAN_2 tokenization (267 vocab, merged IP tokens, delta timestamps)
#
# Model: 95M params, 20 layers, 640 emb, 2048 MLP
# Expected runtime: ~37 hours for 200k steps on A100

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/sharded}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/latency_network}"
RUN_NAME="${RUN_NAME:-plan2_${SLURM_JOB_ID:-manual}}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/src/MaxText/configs/latency_network.yml}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "$LOG_DIR"

# Tee output to logs/ (Slurm still writes to %x-%j.out/err in the submit dir)
LOG_BASENAME="${RUN_NAME}-${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_DIR}/${LOG_BASENAME}.out") 2> >(tee -a "${LOG_DIR}/${LOG_BASENAME}.err" >&2)

echo "========================================"
echo "PLAN_2 Network Measurement Training"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "GPU: nvidia-smi not available"
fi
echo "========================================"

# Load modules (adjust for your cluster)
# Example for common SLURM clusters:
# module load cuda/12.1
# module load cudnn/8.9
# module load python/3.10

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "WARNING: venv not found at $VENV_DIR"
    echo "Attempting to use system Python..."
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# JAX/XLA settings for A100
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"

# Decouple from GCloud (for local cluster)
export DECOUPLE_GCLOUD=TRUE

# Optional: Enable Weights & Biases monitoring
# Uncomment to use wandb instead of/alongside TensorBoard
# export ENABLE_WANDB=true
# export WANDB_PROJECT="ping-llm-plan2"
# export WANDB_ENTITY="your-username"  # Optional

# Directories
mkdir -p "$OUT_DIR"

echo ""
echo "Configuration:"
echo "  Project dir: $PROJECT_DIR"
echo "  Data dir: $DATA_DIR"
echo "  Output dir: $OUT_DIR"
echo "  Run name: $RUN_NAME"
echo "  Config: $CONFIG_FILE"
echo ""

# Verify sharded data exists
echo "Checking data files..."
TRAIN_SHARDS=$(ls -1 "$DATA_DIR"/train/*.parquet 2>/dev/null | wc -l)
TEST_SHARDS=$(ls -1 "$DATA_DIR"/test/*.parquet 2>/dev/null | wc -l)
echo "  Train shards: $TRAIN_SHARDS (expected 180)"
echo "  Test shards: $TEST_SHARDS (expected 20)"

if [ "$TRAIN_SHARDS" -eq 0 ]; then
    echo "ERROR: No training shards found in $DATA_DIR/train/"
    echo "Run: python scripts/data/probe_chunk_preprocess.py first"
    exit 1
fi
if [ "$TEST_SHARDS" -eq 0 ]; then
    echo "ERROR: No test shards found in $DATA_DIR/test/"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo ""
echo "Starting training..."
echo "========================================"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Optional: Initialize wandb
if [ "${ENABLE_WANDB:-false}" = "true" ]; then
    echo "Initializing Weights & Biases..."
    if command -v wandb &> /dev/null; then
        python scripts/setup_wandb.py \
            --config "$CONFIG_FILE" \
            --run-name "$RUN_NAME" \
            --tensorboard-dir "$OUT_DIR/$RUN_NAME/tensorboard" \
            --mode init
        echo "✓ wandb initialized"
    else
        echo "WARNING: wandb not installed, skipping wandb integration"
        echo "Install with: pip install wandb"
    fi
    echo ""
fi

# Training command (PLAN_2 configuration)
python -m MaxText.train \
  "$CONFIG_FILE" \
  run_name="$RUN_NAME" \
  base_output_directory="$OUT_DIR" \
  hardware=gpu \
  per_device_batch_size=32 \
  steps=200000 \
  eval_interval=1000 \
  eval_steps=100 \
  checkpoint_period=5000 \
  log_period=100 \
  dataset_type=grain \
  grain_train_files="$DATA_DIR/train/*.parquet" \
  grain_eval_files="$DATA_DIR/test/*.parquet" \
  grain_worker_count=16

echo ""
echo "========================================"
echo "Training completed successfully"
echo "Output directory: $OUT_DIR/$RUN_NAME"
echo "========================================"
