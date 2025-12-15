#!/bin/bash
# MaxText environment setup with CPU + GPU support
# Based on: chatgpt-research-to-install-deps.md recommendations

set -euo pipefail

echo "=========================================="
echo "MaxText Environment Setup (PLAN_2)"
echo "=========================================="
echo ""
echo "System Info:"
echo "  Python: $(python3 --version)"
echo "  uv: $(uv --version)"
echo "  Location: $(pwd)"
echo ""

# Clean existing venv
if [ -d ".venv" ]; then
    echo "⚠️  Removing existing .venv..."
    rm -rf .venv
    echo ""
fi

# Create venv with Python 3.12
echo "1. Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade core tools
echo ""
echo "2. Upgrading pip and uv..."
pip install --upgrade pip
pip install uv

# Install MaxText from source (editable mode with cuda12 dependencies)
# This ensures we get the correct dependencies for the local source code
echo ""
echo "3. Installing MaxText from source (editable mode)..."
uv pip install -e .[cuda12] --resolution=lowest
echo "   Running install_maxtext_github_deps..."
install_maxtext_github_deps

# Install JAX for CPU (smoke testing on login node)
echo ""
echo "4. Installing JAX CPU support..."
uv pip install --upgrade "jax[cpu]==0.4.34"

# Install JAX for GPU (will activate on GPU nodes)
echo ""
echo "5. Installing JAX GPU support (CUDA 12)..."
# Note: CUDA 12 is standard for A100s. Adjust to cuda13 if needed.
uv pip install --upgrade "jax[cuda12]"

# Install wandb for monitoring
echo ""
echo "6. Installing Wandb..."
uv pip install wandb

# Verify installation
echo ""
echo "7. Verifying installation..."
python -c "import jax; print(f'  JAX version: {jax.__version__}')"
python -c "import jax; print(f'  JAX backend: {jax.default_backend()}')"
python -c "import flax; print(f'  Flax version: {flax.__version__}')"
python -c "import grain; print(f'  Grain version: {grain.__version__}')"
python -c "import pyarrow; print(f'  PyArrow version: {pyarrow.__version__}')"
python -c "import wandb; print(f'  Wandb version: {wandb.__version__}')"

# Check if we can import MaxText (installed in editable mode, no PYTHONPATH needed)
echo ""
echo "8. Testing MaxText import..."
python -c "from MaxText import train; print('  MaxText import: ✓')"

# Freeze dependencies for reproducibility
echo ""
echo "9. Freezing dependencies..."
pip freeze > requirements.lock
echo "  Saved to: requirements.lock ($(wc -l < requirements.lock) packages)"

echo ""
echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. wandb login"
echo "  3. python test_wandb.py (test wandb connection)"
echo "  4. Run CPU smoke test (see VENV_SETUP_PLAN.md Step 2)"
echo ""
echo "Quick smoke test:"
echo "  source .venv/bin/activate"
echo "  DECOUPLE_GCLOUD=TRUE python -m MaxText.train \\"
echo "    src/MaxText/configs/latency_network.yml \\"
echo "    hardware=cpu steps=10 per_device_batch_size=2"
echo ""
echo "GPU Notes:"
echo "  - JAX will auto-detect GPUs on SLURM nodes"
echo "  - CUDA 12 wheels bundled (no system CUDA needed)"
echo "  - Same .venv works on both login + GPU nodes"
echo ""
