#!/bin/bash
#SBATCH --job-name=eval-pipeline
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Full 3-stage eval pipeline: harness (CPU) → model_eval (GPU) → analysis (CPU).
#
# Usage:
#   sbatch scripts/train/slurm_eval_pipeline.sh
#   CHECKPOINT=outputs/checkpoints/680m-200k/latest.pt RUN_NAME=680m-200k \
#     sbatch scripts/train/slurm_eval_pipeline.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.slurm_venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/probe_rows}"
HARNESS_DIR="${HARNESS_DIR:-${PROJECT_DIR}/outputs/eval_harness/default}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/outputs/checkpoints/deep60-60k/latest.pt}"
RUN_NAME="${RUN_NAME:-deep60-60k}"
NUM_SEQUENCES="${NUM_SEQUENCES:-200}"
SEED="${SEED:-42}"
MODEL_RUNS="${MODEL_RUNS:-${RUN_NAME}}"

mkdir -p "${PROJECT_DIR}/logs"

echo "========================================"
echo "ping-llm Eval Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
fi
echo "  Checkpoint: $CHECKPOINT"
echo "  Run name: $RUN_NAME"
echo "  Test data: $DATA_DIR/test.arrayrecord"
echo "  Harness dir: $HARNESS_DIR"
echo "  Sequences: $NUM_SEQUENCES (seed=$SEED)"
echo "========================================"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo ""
echo "=== Stage 1: Harness (baselines + Vivaldi + TRMF) ==="
python -m ping_llm.eval.harness \
    --test-data "${DATA_DIR}/test.arrayrecord" \
    --output-dir "$HARNESS_DIR" \
    --num-sequences "$NUM_SEQUENCES" \
    --seed "$SEED"

echo ""
echo "=== Stage 2: Model Eval ($RUN_NAME) ==="
python -m ping_llm.eval.model_eval \
    --checkpoint "$CHECKPOINT" \
    --test-data "${DATA_DIR}/test.arrayrecord" \
    --harness-dir "$HARNESS_DIR" \
    --run-name "$RUN_NAME"

echo ""
echo "=== Stage 3: Analysis ==="
python -m ping_llm.eval.analysis \
    --harness-dir "$HARNESS_DIR" \
    --model-runs "$MODEL_RUNS" \
    --output-dir "${PROJECT_DIR}/outputs"

echo ""
echo "=== Done ==="
echo "Figures: ${PROJECT_DIR}/outputs/figures/"
echo "Tables: ${PROJECT_DIR}/outputs/tables/"
echo "Harness: ${HARNESS_DIR}/"
