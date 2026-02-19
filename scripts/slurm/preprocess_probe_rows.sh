#!/bin/bash
#SBATCH --job-name=probe_preproc
#SBATCH --partition=cpu
#SBATCH --account=pi_ahoumansadr_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm/logs/preprocess_%j.out
#SBATCH --error=/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm/logs/preprocess_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zevwilson@umass.edu

###############################################################################
# Probe Row Preprocessing SLURM Job — Three-pass hash-partition
#
# Architecture:
#   Pass 1: Per-file GROUP BY (parallel, 30 workers * 8GB = 240GB peak)
#           (already completed — will be skipped on resume)
#   Pass 2: Hash-partition intermediates into 100 bucket files (~2GB RAM)
#   Pass 3: Per-bucket merge by src_addr + sort -> write to ArrayRecord (~5GB RAM)
#
# Resource Estimates:
#   - Input: 720 parquet files, 699 GB, ~25.3B rows
#   - Unique probes (src_addr): ~20,000
#   - Pass 1: ~6.5h (already done, skipped on resume)
#   - Pass 2: ~30 min (hash partition, ~2GB RAM)
#   - Pass 3: ~60 min (bucket merge -> ArrayRecord, ~5GB RAM)
#   - Disk needed: ~700 GB intermediate + ~420 GB buckets + ~100 GB final output
#
# Memory: 64 GB requested (Pass 1 already done; Pass 2+3 need <10GB)
#
# Train/test split: hash_src_addr(src_addr) % 10 < 9 -> train (~90/10)
###############################################################################

set -euo pipefail

echo "=============================================="
echo "Probe Row Preprocessing Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Project paths
PROJECT_DIR="/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm"
VENV_DIR="${PROJECT_DIR}/.slurm_venv"
SCRIPT="${PROJECT_DIR}/scripts/data/create_probe_rows_parallel_streaming.py"
INPUT_PATTERN="${PROJECT_DIR}/data/parquet_ping/*.parquet"
OUTPUT_DIR="${PROJECT_DIR}/data/probe_rows"
TEMP_DIR="/scratch4/workspace/zevwilson_umass_edu-pingdata/tmp/probe_preproc"

# Create directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEMP_DIR}"

# Print system info
echo ""
echo "System info:"
echo "  CPUs: $(nproc)"
echo "  RAM total: $(free -g | awk '/^Mem:/{print $2}') GB"
echo "  RAM available: $(free -g | awk '/^Mem:/{print $7}') GB"
echo "  Temp dir: ${TEMP_DIR}"
echo "  Temp dir space: $(df -h /scratch4 | tail -1 | awk '{print $4}')"
echo ""

# Activate virtual environment
echo "Activating virtual environment: ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install pytz if needed (required for timestamp .as_py() in create_arrayrecord_entry)
pip install -q pytz 2>/dev/null || true

# Verify imports
echo "Verifying Python environment..."
python3 -c "
import duckdb; print(f'  duckdb {duckdb.__version__}')
import pyarrow; print(f'  pyarrow {pyarrow.__version__}')
import pyarrow.compute; print(f'  pyarrow.compute OK')
import array_record.python.array_record_module; print(f'  array_record OK')
"

# Set temp directory via environment variable as a fallback
export TMPDIR="${TEMP_DIR}"

NUM_WORKERS=30
N_BUCKETS=100

echo ""
echo "Configuration:"
echo "  Worker processes: ${NUM_WORKERS}"
echo "  Hash buckets: ${N_BUCKETS}"
echo "  Input: ${INPUT_PATTERN}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Temp: ${TEMP_DIR}"
echo ""

# Run preprocessing
echo "Starting preprocessing..."
echo "=============================================="

python3 "${SCRIPT}" \
    --input "${INPUT_PATTERN}" \
    --output "${OUTPUT_DIR}" \
    --workers "${NUM_WORKERS}" \
    --n-buckets "${N_BUCKETS}" \
    --max-row-size-mb 8.0 \
    --temp-dir "${TEMP_DIR}"

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="

# Note: temp cleanup is handled by the Python script (only on success)
# On failure, temp files are preserved for resume

# Print output file sizes
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}/" 2>/dev/null || echo "No output files found"

echo ""
echo "Done!"
