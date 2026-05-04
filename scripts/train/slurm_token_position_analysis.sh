#!/bin/bash
#SBATCH --job-name=token-pos-analysis
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.slurm_venv}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_DIR}/data/eval_timeclean}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/eval_timeclean_models}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a .pt checkpoint path}"
MAX_TEST="${MAX_TEST:-2000}"
DTYPE="${DTYPE:-bfloat16}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

echo "========================================"
echo "ping-llm token position analysis"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Checkpoint: $CHECKPOINT"
echo "Eval dir: $EVAL_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Max test: $MAX_TEST"
echo "Dtype: $DTYPE"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
fi
echo "========================================"

python -m ping_llm.eval.token_position_analysis \
    --checkpoint "$CHECKPOINT" \
    --eval-dir "$EVAL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-test "$MAX_TEST" \
    --dtype "$DTYPE"
