#!/bin/bash
#SBATCH --job-name=eval-dataprep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Baseline evaluation data prep: scan → neighbors → extract.
# Baseline train/predict/analysis can be enabled after inspecting the split.
# Streams the raw parquet data with DuckDB and writes bounded parquet outputs.
#
# Usage:
#   sbatch scripts/train/slurm_eval_pipeline.sh
#   RUN_BASELINES=1 sbatch scripts/train/slurm_eval_pipeline.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.slurm_venv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/parquet_ping}"
TIME_CLEAN="${TIME_CLEAN:-0}"
if [ -z "${EVAL_DIR+x}" ]; then
    if [ "$TIME_CLEAN" = "1" ]; then
        EVAL_DIR="${PROJECT_DIR}/data/eval_timeclean"
    else
        EVAL_DIR="${PROJECT_DIR}/data/eval"
    fi
fi
if [ -z "${GRAPH_WINDOW+x}" ]; then
    if [ "$TIME_CLEAN" = "1" ]; then
        GRAPH_WINDOW="train"
    else
        GRAPH_WINDOW="all"
    fi
fi
N_NEIGHBORS="${N_NEIGHBORS:-100}"
MAX_PER_PAIR="${MAX_PER_PAIR:-20}"
MAX_TEST_PER_PAIR="${MAX_TEST_PER_PAIR:-2}"
MAX_TRAIN_OBSERVATIONS="${MAX_TRAIN_OBSERVATIONS:-5000000}"
MAX_TEST_OBSERVATIONS="${MAX_TEST_OBSERVATIONS:-1000000}"
TEST_SAMPLE_RATE="${TEST_SAMPLE_RATE:-0.002}"
EXTRACT_SPLIT="${EXTRACT_SPLIT:-auto}"
MAX_BASELINE_TRAIN_OBSERVATIONS="${MAX_BASELINE_TRAIN_OBSERVATIONS:-500000}"
DMFSGD_EPOCHS="${DMFSGD_EPOCHS:-8}"
DMFSGD_DIM="${DMFSGD_DIM:-10}"
DMFSGD_LR="${DMFSGD_LR:-0.02}"
DMFSGD_REG="${DMFSGD_REG:-0.001}"
DMFSGD_SCALE_QUANTILE="${DMFSGD_SCALE_QUANTILE:-0.99}"
RUN_TIME_BASELINES="${RUN_TIME_BASELINES:-1}"
RUN_PAPER_DMFSGD="${RUN_PAPER_DMFSGD:-1}"
PAPER_DMFSGD_EPOCHS="${PAPER_DMFSGD_EPOCHS:-3}"
PAPER_DMFSGD_DIM="${PAPER_DMFSGD_DIM:-10}"
PAPER_DMFSGD_REG="${PAPER_DMFSGD_REG:-1.0}"
PAPER_DMFSGD_ETA_INIT="${PAPER_DMFSGD_ETA_INIT:-0.01}"
PAPER_DMFSGD_LINE_SEARCH_STEPS="${PAPER_DMFSGD_LINE_SEARCH_STEPS:-8}"
PAPER_DMFSGD_LINE_SEARCH_DELTA="${PAPER_DMFSGD_LINE_SEARCH_DELTA:-1e-6}"
PAPER_DMFSGD_NEIGHBOR_CAP="${PAPER_DMFSGD_NEIGHBOR_CAP:-32}"
PAPER_DMFSGD_SCALE_QUANTILE="${PAPER_DMFSGD_SCALE_QUANTILE:-1.0}"
PAPER_DMFSGD_DECAY="${PAPER_DMFSGD_DECAY:-1}"
BIASED_MF_EPOCHS="${BIASED_MF_EPOCHS:-10}"
BIASED_MF_DIM="${BIASED_MF_DIM:-64}"
BIASED_MF_LR="${BIASED_MF_LR:-0.02}"
BIASED_MF_REG="${BIASED_MF_REG:-0.001}"
VIVALDI_EPOCHS="${VIVALDI_EPOCHS:-3}"
DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-180GB}"
DUCKDB_TEMP_DIR="${DUCKDB_TEMP_DIR:-${EVAL_DIR}/duckdb_tmp}"
DUCKDB_THREADS="${DUCKDB_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
RUN_BASELINES="${RUN_BASELINES:-0}"
CLEAN_DUCKDB_TEMP="${CLEAN_DUCKDB_TEMP:-1}"

mkdir -p "${PROJECT_DIR}/logs" "$EVAL_DIR" "$DUCKDB_TEMP_DIR"
if [ "$CLEAN_DUCKDB_TEMP" = "1" ]; then
    find "$DUCKDB_TEMP_DIR" -mindepth 1 -maxdepth 1 -delete
fi

echo "========================================"
echo "ping-llm Baseline Eval Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "  Data dir: $DATA_DIR"
echo "  Eval dir: $EVAL_DIR"
echo "  Time-clean split: $TIME_CLEAN"
echo "  Graph window: $GRAPH_WINDOW"
echo "  Neighbors: $N_NEIGHBORS"
echo "  Max train per pair: $MAX_PER_PAIR"
echo "  Max test per pair: $MAX_TEST_PER_PAIR"
echo "  Max train observations: $MAX_TRAIN_OBSERVATIONS"
echo "  Max test observations: $MAX_TEST_OBSERVATIONS"
echo "  Test sample rate: $TEST_SAMPLE_RATE"
echo "  Extract split: $EXTRACT_SPLIT"
echo "  Max baseline train observations: $MAX_BASELINE_TRAIN_OBSERVATIONS"
echo "  DMFSGD epochs: $DMFSGD_EPOCHS"
echo "  DMFSGD dim/lr/reg/scale-q: $DMFSGD_DIM / $DMFSGD_LR / $DMFSGD_REG / $DMFSGD_SCALE_QUANTILE"
echo "  Time-ordered baseline variants: $RUN_TIME_BASELINES"
echo "  Paper DMFSGD enabled: $RUN_PAPER_DMFSGD"
echo "  Paper DMFSGD epochs: $PAPER_DMFSGD_EPOCHS"
echo "  Paper DMFSGD dim/reg/eta/line-search/cap/scale-q/decay: $PAPER_DMFSGD_DIM / $PAPER_DMFSGD_REG / $PAPER_DMFSGD_ETA_INIT / $PAPER_DMFSGD_LINE_SEARCH_STEPS / $PAPER_DMFSGD_NEIGHBOR_CAP / $PAPER_DMFSGD_SCALE_QUANTILE / $PAPER_DMFSGD_DECAY"
echo "  BiasedMF epochs: $BIASED_MF_EPOCHS"
echo "  BiasedMF dim/lr/reg: $BIASED_MF_DIM / $BIASED_MF_LR / $BIASED_MF_REG"
echo "  Vivaldi epochs: $VIVALDI_EPOCHS"
echo "  DuckDB memory: $DUCKDB_MEMORY_LIMIT"
echo "  DuckDB temp dir: $DUCKDB_TEMP_DIR"
echo "  DuckDB threads: $DUCKDB_THREADS"
echo "  Clean DuckDB temp: $CLEAN_DUCKDB_TEMP"
echo "  Run baselines: $RUN_BASELINES"
echo "========================================"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo ""
echo "=== Step 1: Scan (build bidir IP set + pair stats) ==="
SCAN_OK=0
SCAN_REBUILT=0
if [ -s "${EVAL_DIR}/scan_meta.json" ] && [ -s "${EVAL_DIR}/pair_stats.parquet" ]; then
    EXISTING_GRAPH_WINDOW="$(python - <<'PY' "${EVAL_DIR}/scan_meta.json"
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("graph_window", "all"))
PY
)"
    if [ "$EXISTING_GRAPH_WINDOW" = "$GRAPH_WINDOW" ]; then
        SCAN_OK=1
    else
        echo "Existing scan graph_window=$EXISTING_GRAPH_WINDOW; requested $GRAPH_WINDOW, rebuilding scan outputs."
    fi
fi
if [ "$SCAN_OK" = "1" ]; then
    echo "Reusing existing scan outputs."
else
    SCAN_REBUILT=1
    python -m ping_llm.eval.harness scan \
        --data-dir "$DATA_DIR" \
        --eval-dir "$EVAL_DIR" \
        --graph-window "$GRAPH_WINDOW" \
        --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT" \
        --duckdb-temp-dir "$DUCKDB_TEMP_DIR" \
        --duckdb-threads "$DUCKDB_THREADS"
fi

echo ""
echo "=== Step 2: Neighbor selection ==="
NEIGHBORS_REBUILT=0
if [ "$SCAN_REBUILT" = "0" ] && [ -s "${EVAL_DIR}/neighbor_graph.json" ] && [ -s "${EVAL_DIR}/neighbor_pairs.parquet" ]; then
    echo "Reusing existing neighbor outputs."
else
    NEIGHBORS_REBUILT=1
    python -m ping_llm.eval.harness neighbors \
        --eval-dir "$EVAL_DIR" \
        --n-neighbors "$N_NEIGHBORS" \
        --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT" \
        --duckdb-temp-dir "$DUCKDB_TEMP_DIR" \
        --duckdb-threads "$DUCKDB_THREADS"
fi

echo ""
echo "=== Step 3: Extract train/test measurements ==="
RUN_EXTRACT_SPLIT="$EXTRACT_SPLIT"
if [ "$RUN_EXTRACT_SPLIT" = "auto" ]; then
    if [ "$NEIGHBORS_REBUILT" = "1" ]; then
        RUN_EXTRACT_SPLIT="both"
    elif [ -s "${EVAL_DIR}/train_measurements.parquet" ] && [ ! -s "${EVAL_DIR}/test_measurements.parquet" ]; then
        RUN_EXTRACT_SPLIT="test"
    elif [ -s "${EVAL_DIR}/train_measurements.parquet" ] && [ -s "${EVAL_DIR}/test_measurements.parquet" ]; then
        RUN_EXTRACT_SPLIT="skip"
    else
        RUN_EXTRACT_SPLIT="both"
    fi
fi
echo "Resolved extract split: $RUN_EXTRACT_SPLIT"
if [ "$RUN_EXTRACT_SPLIT" = "skip" ]; then
    echo "Reusing existing train/test outputs."
else
python -m ping_llm.eval.harness extract \
    --data-dir "$DATA_DIR" \
    --eval-dir "$EVAL_DIR" \
    --max-per-pair "$MAX_PER_PAIR" \
    --max-test-per-pair "$MAX_TEST_PER_PAIR" \
    --max-train-observations "$MAX_TRAIN_OBSERVATIONS" \
    --max-test-observations "$MAX_TEST_OBSERVATIONS" \
    --test-sample-rate "$TEST_SAMPLE_RATE" \
    --extract-split "$RUN_EXTRACT_SPLIT" \
    --duckdb-memory-limit "$DUCKDB_MEMORY_LIMIT" \
    --duckdb-temp-dir "$DUCKDB_TEMP_DIR" \
    --duckdb-threads "$DUCKDB_THREADS"
fi

if [ "$RUN_BASELINES" != "1" ]; then
    echo ""
    echo "=== Data prep done; skipping baseline train/predict/analysis ==="
    echo "Train: ${EVAL_DIR}/train_measurements.parquet"
    echo "Test: ${EVAL_DIR}/test_measurements.parquet"
    exit 0
fi

echo ""
echo "=== Step 4: Train baselines ==="
PAPER_DMFSGD_FLAG="--paper-dmfsgd"
if [ "$RUN_PAPER_DMFSGD" != "1" ]; then
    PAPER_DMFSGD_FLAG="--no-paper-dmfsgd"
fi
PAPER_DMFSGD_DECAY_FLAG="--paper-dmfsgd-decay"
if [ "$PAPER_DMFSGD_DECAY" != "1" ]; then
    PAPER_DMFSGD_DECAY_FLAG="--no-paper-dmfsgd-decay"
fi
TIME_BASELINES_FLAG="--time-baselines"
if [ "$RUN_TIME_BASELINES" != "1" ]; then
    TIME_BASELINES_FLAG="--no-time-baselines"
fi
python -m ping_llm.eval.harness train \
    --eval-dir "$EVAL_DIR" \
    --max-baseline-train-observations "$MAX_BASELINE_TRAIN_OBSERVATIONS" \
    --dmfsgd-epochs "$DMFSGD_EPOCHS" \
    --dmfsgd-dim "$DMFSGD_DIM" \
    --dmfsgd-lr "$DMFSGD_LR" \
    --dmfsgd-reg "$DMFSGD_REG" \
    --dmfsgd-scale-quantile "$DMFSGD_SCALE_QUANTILE" \
    "$TIME_BASELINES_FLAG" \
    "$PAPER_DMFSGD_FLAG" \
    --paper-dmfsgd-epochs "$PAPER_DMFSGD_EPOCHS" \
    --paper-dmfsgd-dim "$PAPER_DMFSGD_DIM" \
    --paper-dmfsgd-reg "$PAPER_DMFSGD_REG" \
    --paper-dmfsgd-eta-init "$PAPER_DMFSGD_ETA_INIT" \
    --paper-dmfsgd-line-search-steps "$PAPER_DMFSGD_LINE_SEARCH_STEPS" \
    --paper-dmfsgd-line-search-delta "$PAPER_DMFSGD_LINE_SEARCH_DELTA" \
    --paper-dmfsgd-neighbor-cap "$PAPER_DMFSGD_NEIGHBOR_CAP" \
    --paper-dmfsgd-scale-quantile "$PAPER_DMFSGD_SCALE_QUANTILE" \
    "$PAPER_DMFSGD_DECAY_FLAG" \
    --biased-mf-epochs "$BIASED_MF_EPOCHS" \
    --biased-mf-dim "$BIASED_MF_DIM" \
    --biased-mf-lr "$BIASED_MF_LR" \
    --biased-mf-reg "$BIASED_MF_REG" \
    --vivaldi-epochs "$VIVALDI_EPOCHS"

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
