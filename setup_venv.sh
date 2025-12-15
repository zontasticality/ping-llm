#!/bin/bash
# Quick setup script for MaxText training environment
# This .venv will be accessible from both login and GPU nodes

set -e  # Exit on error

echo "=========================================="
echo "Setting up MaxText training environment"
echo "=========================================="

# Clean existing venv
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create venv
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core packages
echo "Installing JAX (CPU version)..."
pip install "jax[cpu]==0.4.34" "jaxlib==0.4.34"

echo "Installing JAX ecosystem..."
pip install \
  flax==0.9.0 \
  optax==0.2.4 \
  orbax-checkpoint==0.11.0 \
  chex==0.1.87

echo "Installing data handling..."
pip install \
  "pyarrow>=22.0.0" \
  pandas \
  "grain>=0.2.15"

echo "Installing utilities..."
pip install \
  pyyaml \
  absl-py \
  ml-collections \
  tqdm \
  tensorboard \
  tensorboardx

echo "Installing wandb..."
pip install wandb

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "Verify with:"
echo "  source .venv/bin/activate"
echo "  python -c \"import jax, flax, grain, pyarrow; print('Success!')\""
echo ""
echo "This .venv is on /scratch and will work on GPU nodes too!"
echo ""
