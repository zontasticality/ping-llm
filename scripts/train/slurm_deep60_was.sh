#!/bin/bash
#SBATCH --job-name=deep60-was-60k
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/deep60-was-60k-%j.out
#SBATCH --error=logs/deep60-was-60k-%j.err

# deep60 architecture (60L/384E/6H/64HD, 106M params) with RTT-aware
# Wasserstein loss. Compare against deep60-60k baseline (plain CE).

set -euo pipefail
cd ~/pingdata/ping-llm
source .slurm_venv/bin/activate

export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=outputs/torch_cache

TRAIN_DATA="data/probe_rows/train_shards/train.arrayrecord-00000-of-00004"
TRAIN_DATA="${TRAIN_DATA},data/probe_rows/train_shards/train.arrayrecord-00001-of-00004"
TRAIN_DATA="${TRAIN_DATA},data/probe_rows/train_shards/train.arrayrecord-00002-of-00004"
TRAIN_DATA="${TRAIN_DATA},data/probe_rows/train_shards/train.arrayrecord-00003-of-00004"

python -m ping_llm.train \
    --run-name deep60-was-60k \
    --total-steps 60000 \
    --batch-size 32 \
    --n-layer 60 \
    --n-embd 384 \
    --n-head 6 \
    --head-dim 64 \
    --rtt-was-lambda1 0.5 \
    --rtt-was-lambda2 0.1 \
    --train-data $TRAIN_DATA \
    --eval-data data/probe_rows/test.arrayrecord \
    --checkpoint-dir outputs/checkpoints \
    --checkpoint-interval 200 \
    --wandb-project ping-llm

echo Training done. Running eval...

python -m ping_llm.eval.run_all \
    --checkpoint outputs/checkpoints/deep60-was-60k/latest.pt \
    --test-data data/probe_rows/test.arrayrecord \
    --tests loss_breakdown,baselines \
    --loss-sequences 200 \
    --baseline-sequences 100 \
    --output-dir outputs/eval_results_deep60_was_60k

echo COMPLETE
