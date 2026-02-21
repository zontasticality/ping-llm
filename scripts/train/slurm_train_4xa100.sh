#!/bin/bash
#SBATCH --job-name=ping-llm-4xa100
#SBATCH --partition=gpu
#SBATCH --account=pi_ahoumansadr_umass_edu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm/logs/train_%j.out
#SBATCH --error=/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm/logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zevwilson@umass.edu

###############################################################################
# 95M Baseline Training — 4x A100, Pure Data Parallelism
#
# Architecture: 95M decoder-only Transformer (20L, 640 emb, 10 heads, 2048 MLP)
# Data: 25.3B measurements in probe_rows/{train,test}.arrayrecord
# Parallelism: 4-way data parallel (ici_data_parallelism=4)
# Global batch: 4 * 256 = 1024
#
# Goal: Validate pipeline end-to-end, baseline loss curve, debug perf.
###############################################################################

set -euo pipefail

echo "=============================================="
echo "95M Baseline Training — 4x A100"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

PROJECT_DIR="/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm"
VENV_DIR="${PROJECT_DIR}/.train_venv"
CONFIG_FILE="${PROJECT_DIR}/src/MaxText/configs/latency_network.yml"
RUN_NAME="baseline_95m_4xa100_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs"

# System info
echo ""
echo "System info:"
echo "  CPUs: $(nproc)"
echo "  RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Activate venv
echo "Activating virtual environment: ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Verify imports
echo "Verifying Python environment..."
python3 -c "
import jax; print(f'  JAX {jax.__version__}')
print(f'  Devices: {jax.device_count()} ({[d.platform for d in jax.devices()]})')
print(f'  Local devices: {jax.local_device_count()}')
"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export DECOUPLE_GCLOUD=TRUE

# XLA flags for A100
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True"

# Verify data exists
echo ""
echo "Checking data..."
ls -lh "${PROJECT_DIR}/data/probe_rows/"
echo ""

echo "Configuration:"
echo "  Config: ${CONFIG_FILE}"
echo "  Run name: ${RUN_NAME}"
echo "  GPUs: 4x A100 (data parallel)"
echo "  Per-device batch: 256"
echo "  Global batch: 1024"
echo "  Steps: 1000"
echo ""

echo "Starting training..."
echo "=============================================="

cd "${PROJECT_DIR}"

python -m MaxText.train \
  "${CONFIG_FILE}" \
  run_name="${RUN_NAME}" \
  hardware=gpu \
  skip_jax_distributed_system=false \
  ici_data_parallelism=4 \
  ici_fsdp_parallelism=1 \
  steps=1000 \
  per_device_batch_size=256 \
  log_period=10 \
  eval_interval=100 \
  eval_steps=5 \
  checkpoint_period=500 \
  enable_checkpointing=true

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
