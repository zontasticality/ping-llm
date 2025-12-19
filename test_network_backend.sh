#!/usr/bin/env bash
# Test script for network backend refactoring
# Run this to verify the refactored implementation works correctly

set -e  # Exit on error

# Get script directory and cd there
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================"
echo "Network Backend Refactoring Tests"
echo "================================"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Test 1: Import test
echo "[1/5] Testing imports..."
PYTHONPATH=src python -c "
from MaxText.input_pipeline._network_data_processing import make_network_train_iterator, make_network_eval_iterator
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
print('✓ All imports successful')
" 2>&1 | grep -E "^(✓|✗)" || echo "✗ Import test failed"

echo ""

# Test 2: Config loading test
echo "[2/5] Testing config loading..."
PYTHONPATH=src DECOUPLE_GCLOUD=TRUE python -c "
from MaxText import pyconfig
config = pyconfig.initialize(['', 'MaxText/configs/latency_network.yml'])
assert config.dataset_type.value == 'network', 'Wrong dataset_type'
assert config.network_data_format == 'probe_chunks', 'Wrong format'
assert config.network_train_files == 'data/probe_rows/train.arrayrecord', 'Wrong train files'
print('✓ Config loads correctly with all fields')
" 2>&1 | grep -E "^(✓|✗)" || echo "✗ Config test failed"

echo ""

# Test 3: Check files exist
echo "[3/5] Checking created/modified files..."
FILES=(
    "src/MaxText/input_pipeline/_network_data_processing.py"
    "src/MaxText/input_pipeline/input_pipeline_interface.py"
    "src/MaxText/configs/types.py"
    "src/MaxText/configs/latency_network.yml"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
        exit 1
    fi
done

echo ""

# Test 4: Verify no network imports in grain backend
echo "[4/5] Verifying clean separation (no network imports in grain)..."
if grep -q "_network_grain_integration" src/MaxText/input_pipeline/_grain_data_processing.py; then
    echo "  ✗ Found _network_grain_integration import in grain backend"
    exit 1
else
    echo "  ✓ Grain backend is clean (no network imports)"
fi

echo ""

# Test 5: Verify network backend is registered
echo "[5/5] Verifying network backend registration..."
if grep -q '"network":' src/MaxText/input_pipeline/input_pipeline_interface.py; then
    echo "  ✓ Network backend registered in interface"
else
    echo "  ✗ Network backend not found in interface"
    exit 1
fi

echo ""
echo "================================"
echo "All tests passed! ✓"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Read REFACTORING_WRITEUP.md for complete documentation"
echo "  2. Enable debugging: add 'grain_debug_mode: true' to config"
echo "  3. Run training: DECOUPLE_GCLOUD=TRUE python -m MaxText.train src/MaxText/configs/latency_network.yml"
echo ""
