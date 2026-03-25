#!/bin/bash
#SBATCH --job-name=ping-llm-preempt
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Preemptible A100 training for ping-llm (95M model, 14k steps).
#
# Features:
#   - Auto-requeues on preemption (--requeue)
#   - SIGTERM sent 120s before wall-time limit (--signal)
#   - Checkpoints every 200 steps for safe resume
#   - Wandb run ID persisted in checkpoint for continuous logging
#
# Usage:
#   sbatch scripts/train/slurm_gpu_preempt.sh
#   RUN_NAME=my-run STEPS=1000 sbatch scripts/train/slurm_gpu_preempt.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.slurm_venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/probe_rows}"
RUN_NAME="${RUN_NAME:-95m-14k-preempt}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/outputs/checkpoints}"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "========================================"
echo "ping-llm PyTorch Training (gpu-preempt)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi
echo "========================================"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
else
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run: uv venv .slurm_venv --python 3.12 && source .slurm_venv/bin/activate && uv pip install ..."
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore"
export PYTHONUNBUFFERED=1

# torch.compile cache (persists across preemptions)
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/outputs/torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Configurable parameters
STEPS="${STEPS:-14000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-ping-llm}"

# Build train data path (4 shards)
TRAIN_DATA=""
for i in 0 1 2 3; do
    SHARD="${DATA_DIR}/train_shards/train.arrayrecord-0000${i}-of-00004"
    if [ ! -f "$SHARD" ]; then
        echo "ERROR: Missing shard $SHARD"
        exit 1
    fi
    [ -n "$TRAIN_DATA" ] && TRAIN_DATA="${TRAIN_DATA},"
    TRAIN_DATA="${TRAIN_DATA}${SHARD}"
done
echo "Train data: 4 shards OK"

EVAL_DATA="${DATA_DIR}/test.arrayrecord"
if [ ! -f "$EVAL_DATA" ]; then
    echo "ERROR: Missing eval data $EVAL_DATA"
    exit 1
fi
echo "Eval data: OK"

echo ""
echo "Configuration:"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Run name: $RUN_NAME"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Checkpoint interval: 200"
echo ""

echo "Starting training..."
echo "========================================"

cd "$PROJECT_DIR"

python -m ping_llm.train \
    --run-name "$RUN_NAME" \
    --total-steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --train-data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-interval 200 \
    --wandb-project "$WANDB_PROJECT"

echo ""
echo "========================================"
echo "Training completed"
echo "Checkpoints: $CHECKPOINT_DIR/$RUN_NAME"
echo "========================================"
