#!/bin/bash
#SBATCH --job-name=eval-baselines
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Baseline evaluation pipeline: scan → neighbors → extract → train → predict → analysis
# Streams 699GB of parquet data in two passes. No GPU needed.
#
# Usage:
#   sbatch scripts/train/slurm_eval_pipeline.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.slurm_venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/parquet_ping}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_DIR}/data/eval}"
N_NEIGHBORS="${N_NEIGHBORS:-100}"
MAX_PER_PAIR="${MAX_PER_PAIR:-100}"

mkdir -p "${PROJECT_DIR}/logs"

echo "========================================"
echo "ping-llm Baseline Eval Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "  Data dir: $DATA_DIR"
echo "  Eval dir: $EVAL_DIR"
echo "  Neighbors: $N_NEIGHBORS"
echo "  Max per pair: $MAX_PER_PAIR"
echo "========================================"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo ""
echo "=== Step 1: Scan (build bidir IP set + pair stats) ==="
python -m ping_llm.eval.harness scan \
    --data-dir "$DATA_DIR" \
    --eval-dir "$EVAL_DIR"

echo ""
echo "=== Step 2: Neighbor selection ==="
python -m ping_llm.eval.harness neighbors \
    --eval-dir "$EVAL_DIR" \
    --n-neighbors "$N_NEIGHBORS"

echo ""
echo "=== Step 3: Extract train/test measurements ==="
python -m ping_llm.eval.harness extract \
    --data-dir "$DATA_DIR" \
    --eval-dir "$EVAL_DIR" \
    --max-per-pair "$MAX_PER_PAIR"

echo ""
echo "=== Step 4: Train baselines ==="
python -m ping_llm.eval.harness train \
    --eval-dir "$EVAL_DIR"

echo ""
echo "=== Step 5: Predict ==="
python -m ping_llm.eval.harness predict \
    --eval-dir "$EVAL_DIR"

echo ""
echo "=== Step 6: Analysis ==="
python -m ping_llm.eval.analysis \
    --harness-dir "$EVAL_DIR" \
    --output-dir "${PROJECT_DIR}/outputs"

echo ""
echo "=== Done ==="
echo "Observations: ${EVAL_DIR}/observations.parquet"
echo "Figures: ${PROJECT_DIR}/outputs/figures/"
echo "Tables: ${PROJECT_DIR}/outputs/tables/"
