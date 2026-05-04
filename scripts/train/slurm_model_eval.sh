#!/bin/bash
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.slurm_venv}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_DIR}/data/eval}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a .pt checkpoint path}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME for the output parquet name}"
NUM_CONTEXT="${NUM_CONTEXT:-0,1,2,5,10}"
MAX_TEST="${MAX_TEST:-10000}"
DTYPE="${DTYPE:-bfloat16}"
STRIP_TIMESTAMPS="${STRIP_TIMESTAMPS:-0}"

cd "$PROJECT_DIR"
mkdir -p logs "${EVAL_DIR}/model_preds"
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

echo "========================================"
echo "ping-llm model eval"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Checkpoint: $CHECKPOINT"
echo "Run name: $RUN_NAME"
echo "Eval dir: $EVAL_DIR"
echo "Num context: $NUM_CONTEXT"
echo "Max test: $MAX_TEST"
echo "Dtype: $DTYPE"
echo "Strip timestamps: $STRIP_TIMESTAMPS"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
fi
echo "========================================"

EXTRA_ARGS=()
if [ "$STRIP_TIMESTAMPS" = "1" ]; then
    EXTRA_ARGS+=(--strip-timestamps)
fi

python -m ping_llm.eval.model_eval \
    --checkpoint "$CHECKPOINT" \
    --eval-dir "$EVAL_DIR" \
    --run-name "$RUN_NAME" \
    --num-context "$NUM_CONTEXT" \
    --max-test "$MAX_TEST" \
    --dtype "$DTYPE" \
    "${EXTRA_ARGS[@]}"
