#!/bin/bash
#SBATCH --job-name=ping-llm-pytorch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# SLURM script for PyTorch ping-llm training
#
# Usage:
#   sbatch scripts/train/slurm_train_pytorch.sh
#   sbatch --gres=gpu:h100:1 scripts/train/slurm_train_pytorch.sh
#   sbatch --gres=gpu:a100:1 scripts/train/slurm_train_pytorch.sh
#
# Configuration via environment variables:
#   STEPS=14000 BATCH_SIZE=256 RUN_NAME=test sbatch scripts/train/slurm_train_pytorch.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/probe_rows}"
RUN_NAME="${RUN_NAME:-pytorch_${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "$LOG_DIR"

# Tee output to logs/
LOG_BASENAME="${RUN_NAME}-${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_DIR}/${LOG_BASENAME}.out") 2> >(tee -a "${LOG_DIR}/${LOG_BASENAME}.err" >&2)

echo "========================================"
echo "PyTorch ping-llm Training"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
fi
echo "========================================"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run: bash setup_venv.sh"
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Configurable parameters
STEPS="${STEPS:-14000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WANDB_PROJECT="${WANDB_PROJECT:-ping-llm}"
WANDB_MODE="${WANDB_MODE:-online}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/outputs/checkpoints}"

# Verify data
echo "Checking data..."
if [ ! -f "$DATA_DIR/train.arrayrecord" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/train.arrayrecord"
    exit 1
fi
echo "  Found: $DATA_DIR/train.arrayrecord"

echo ""
echo "Configuration:"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Run name: $RUN_NAME"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Wandb: $WANDB_PROJECT ($WANDB_MODE)"
echo ""

echo "Starting training..."
echo "========================================"

cd "$PROJECT_DIR"

python -m ping_llm.train \
    --run-name "$RUN_NAME" \
    --total-steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --train-data "$DATA_DIR/train.arrayrecord" \
    --eval-data "$DATA_DIR/test.arrayrecord" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-mode "$WANDB_MODE"

echo ""
echo "========================================"
echo "Training completed"
echo "Checkpoints: $CHECKPOINT_DIR/$RUN_NAME"
echo "========================================"
