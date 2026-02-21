#!/bin/bash
#SBATCH --job-name=setup-venv
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_ahoumansadr_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm/logs/setup_venv_%j.out
#SBATCH --error=/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm/logs/setup_venv_%j.err

###############################################################################
# Create .train_venv with JAX + CUDA12 + MaxText dependencies
# Run once, then use for all training jobs.
###############################################################################

set -euo pipefail

PROJECT_DIR="/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm"
VENV_DIR="${PROJECT_DIR}/.train_venv"

echo "=============================================="
echo "Setting up training venv"
echo "Node: $(hostname)"
echo "=============================================="

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ""

# Create fresh venv
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

echo "Creating venv at ${VENV_DIR}..."
python3 -m venv "$VENV_DIR"
source "${VENV_DIR}/bin/activate"

echo "Python: $(python3 --version)"
echo "pip: $(pip --version)"
echo ""

# Upgrade pip
pip install --upgrade pip

# Install MaxText with CUDA12 deps
# Skip transformer-engine (requires NVTX headers not available on this cluster;
# only needed for H100 FP8, not A100 training)
echo "Installing MaxText[cuda12] (excluding transformer-engine)..."
cd "$PROJECT_DIR"
pip install -e ".[cuda12]" 2>&1 | tee /dev/stderr || {
    echo ""
    echo "Full install failed, retrying without transformer-engine..."
    # Install requirements directly, filtering out transformer-engine
    grep -v '^transformer-engine' \
        dependencies/requirements/generated_requirements/cuda12-requirements.txt \
        > /tmp/cuda12-filtered.txt
    pip install -r /tmp/cuda12-filtered.txt
    pip install -e .
}

# Install extra github deps (MaxText has some)
echo "Installing MaxText github deps..."
install_maxtext_github_deps 2>/dev/null || echo "No github deps script found, skipping"

# Verify
echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="
python3 -c "
import jax
print(f'JAX {jax.__version__}')
print(f'Devices: {jax.device_count()}')
for d in jax.devices():
    print(f'  {d}')
"

python3 -c "
import flax; print(f'Flax {flax.__version__}')
import optax; print(f'Optax {optax.__version__}')
import grain; print(f'Grain {grain.__version__}')
import array_record; print('array_record OK')
import tensorflow as tf; print(f'TF {tf.__version__}')
"

python3 -c "
from MaxText import pyconfig
print('MaxText import OK')
"

echo ""
echo "=============================================="
echo "Venv ready at: ${VENV_DIR}"
echo "=============================================="
