#!/bin/bash
#SBATCH --job-name=ping-llm-h100
#SBATCH --partition=gpu          # Override with: sbatch --partition=<gpu-partition> scripts/...
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G                # Increased from 32G for larger batch size
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# SLURM script for training network measurement model on H100
#
# Usage:
#   sbatch scripts/train/slurm_train_h100.sh
#
# Configuration:
#   - GPU: H100 (80GB HBM3)
#   - Steps: 10,000
#   - Batch size: 512 per device
#   - Checkpointing: Every 1000 steps
#   - Wandb: Enabled by default
#
# Prerequisites:
# 1. Dataset in data/probe_rows/{train,test}.arrayrecord
#    OR data/sharded/{train,test}/ (legacy parquet shards)
# 2. Virtual environment with MaxText + dependencies
# 3. Wandb authentication: `wandb login` (if using wandb)
#
# Model: 95M params, 20 layers, 640 emb, 2048 MLP
# Expected runtime: ~6-8 hours for 10k steps on H100

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/probe_rows}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/latency_network}"
RUN_NAME="${RUN_NAME:-h100_10k_${SLURM_JOB_ID:-manual}}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/src/MaxText/configs/latency_network.yml}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "$LOG_DIR"

# Tee output to logs/ (Slurm still writes to %x-%j.out/err in the submit dir)
LOG_BASENAME="${RUN_NAME}-${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_DIR}/${LOG_BASENAME}.out") 2> >(tee -a "${LOG_DIR}/${LOG_BASENAME}.err" >&2)

echo "========================================"
echo "H100 Network Measurement Training"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
else
    echo "GPU: nvidia-smi not available"
fi
echo "========================================"

# Load modules (adjust for your cluster)
# Example for common SLURM clusters:
# module load cuda/12.3      # H100 works best with CUDA 12.x
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

# JAX/XLA settings optimized for H100
# H100 has enhanced Tensor Cores and Transformer Engine support
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_highest_priority_async_stream=true"

# Decouple from GCloud (for local cluster)
export DECOUPLE_GCLOUD=TRUE

# Weights & Biases monitoring (enabled by default)
export ENABLE_WANDB="${ENABLE_WANDB:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-ping-llm}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"  # Set to your username/team if needed
export WANDB_NAME="${RUN_NAME}"

# Directories
mkdir -p "$OUT_DIR"

echo ""
echo "Configuration:"
echo "  Project dir: $PROJECT_DIR"
echo "  Data dir: $DATA_DIR"
echo "  Output dir: $OUT_DIR"
echo "  Run name: $RUN_NAME"
echo "  Config: $CONFIG_FILE"
echo "  Steps: 10,000"
echo "  Batch size: 512"
echo "  Checkpoint period: 1000"
echo "  Wandb: ${ENABLE_WANDB}"
echo ""

# Verify data exists (probe_chunks format preferred)
echo "Checking data files..."
if [ -f "$DATA_DIR/train.arrayrecord" ]; then
    echo "  ✓ Found PLAN_3 ArrayRecord training data"
    TRAIN_FILES="$DATA_DIR/train.arrayrecord"
    EVAL_FILES="$DATA_DIR/test.arrayrecord"
    DATA_TYPE="probe_chunks"
elif [ -d "${PROJECT_DIR}/data/sharded/train" ]; then
    echo "  ✓ Found legacy parquet shards"
    TRAIN_SHARDS=$(ls -1 "${PROJECT_DIR}/data/sharded/train"/*.parquet 2>/dev/null | wc -l)
    TEST_SHARDS=$(ls -1 "${PROJECT_DIR}/data/sharded/test"/*.parquet 2>/dev/null | wc -l)
    echo "    Train shards: $TRAIN_SHARDS"
    echo "    Test shards: $TEST_SHARDS"
    TRAIN_FILES="${PROJECT_DIR}/data/sharded/train/*.parquet"
    EVAL_FILES="${PROJECT_DIR}/data/sharded/test/*.parquet"
    DATA_TYPE="grain"
else
    echo "ERROR: No training data found!"
    echo "  Expected: $DATA_DIR/train.arrayrecord"
    echo "       OR: ${PROJECT_DIR}/data/sharded/train/*.parquet (legacy)"
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

# Optional: Initialize wandb with config upload
if [ "${ENABLE_WANDB}" = "true" ]; then
    echo "Initializing Weights & Biases..."
    if command -v wandb &> /dev/null; then
        # Use train.py wrapper for better wandb integration
        echo "Using train.py wrapper for wandb integration"
    else
        echo "WARNING: wandb not installed, skipping wandb integration"
        echo "Install with: pip install wandb"
        export ENABLE_WANDB=false
    fi
    echo ""
fi

# Training command with H100-optimized settings
if [ "${ENABLE_WANDB}" = "true" ] && command -v wandb &> /dev/null; then
    # Use train.py wrapper for integrated wandb + tensorboard sync
    python scripts/train.py \
      --config "$CONFIG_FILE" \
      --name "$RUN_NAME" \
      --project "${WANDB_PROJECT}" \
      ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} \
      --steps 10000 \
      --batch-size 512 \
      --hardware gpu \
      --enable-checkpointing
else
    # Direct MaxText invocation without wandb
    if [ "$DATA_TYPE" = "probe_chunks" ]; then
        python -m MaxText.train \
          "$CONFIG_FILE" \
          run_name="$RUN_NAME" \
          base_output_directory="$OUT_DIR" \
          hardware=gpu \
          per_device_batch_size=512 \
          steps=10000 \
          eval_interval=500 \
          eval_steps=10 \
          checkpoint_period=1000 \
          log_period=100 \
          dataset_type=network \
          network_data_format=probe_chunks \
          network_train_files="$TRAIN_FILES" \
          network_eval_files="$EVAL_FILES" \
          grain_worker_count=16
    else
        # Legacy parquet format
        python -m MaxText.train \
          "$CONFIG_FILE" \
          run_name="$RUN_NAME" \
          base_output_directory="$OUT_DIR" \
          hardware=gpu \
          per_device_batch_size=512 \
          steps=10000 \
          eval_interval=500 \
          eval_steps=10 \
          checkpoint_period=1000 \
          log_period=100 \
          dataset_type=grain \
          grain_train_files="$TRAIN_FILES" \
          grain_eval_files="$EVAL_FILES" \
          grain_worker_count=16
    fi
fi

echo ""
echo "========================================"
echo "Training completed successfully"
echo "Output directory: $OUT_DIR/$RUN_NAME"
if [ "${ENABLE_WANDB}" = "true" ]; then
    echo "Wandb dashboard: https://wandb.ai/${WANDB_ENTITY:-your-username}/${WANDB_PROJECT}"
fi
echo "========================================"
