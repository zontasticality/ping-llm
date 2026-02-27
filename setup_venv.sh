#!/bin/bash
# Setup script for ping-llm PyTorch training environment (CPU)
# For GPU training, install torch with CUDA support separately.

set -e

echo "=========================================="
echo "Setting up ping-llm PyTorch environment"
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

echo "Installing PyTorch (CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing data handling..."
pip install \
  "pyarrow>=22.0.0" \
  numpy \
  "grain>=0.2.15" \
  array_record

echo "Installing wandb..."
pip install wandb

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Verify with:"
echo "  source .venv/bin/activate"
echo "  PYTHONPATH=src python -c \"from ping_llm.data.tokenization import VOCAB_SIZE; print(VOCAB_SIZE)\""
echo ""
echo "For GPU training, install torch with CUDA:"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
echo ""
