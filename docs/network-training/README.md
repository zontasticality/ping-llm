# Network Training Pipeline

This directory contains documentation for the custom network measurement training pipeline built on MaxText.

## Quick Start

### Local Training
```bash
python scripts/train.py \
  --config src/MaxText/configs/latency_network.yml \
  --project my-project \
  --name my-run \
  --steps 5000 \
  --batch-size 128
```

### Modal Training
```bash
modal run scripts/train/modal.py::run \
  --run-name my-run \
  --steps 5000 \
  --batch-size 128 \
  --wandb-project my-project
```

Or use the helper script:
```bash
./run_modal_training.sh
```

## Documentation

- **[Architecture](architecture.md)** - Design decisions, PLAN_3 implementation details
- **[Data Pipeline](data-pipeline.md)** - Data preprocessing and loading
- **[Modal Deployment](modal-deployment.md)** - Running on Modal infrastructure
- **[Troubleshooting](troubleshooting/)** - Common issues and solutions

## Key Features

- **PLAN_3 Design**: Probe-centric big-row data loading
- **Minimal Padding**: <5% padding vs 50-90% in previous approaches
- **Multi-scale Learning**: Log-uniform window sampling for temporal patterns
- **Runtime Tokenization**: Data augmentation with 3 timestamp modes
- **Performance Optimized**: Multiprocessing, efficient pipeline ordering

## Architecture Overview

The network training pipeline is implemented as a separate backend for MaxText:

```
src/MaxText/input_pipeline/
├── _network_data_processing.py    # Backend interface
├── probe_chunk_pipeline.py        # Dataset builder
├── _probe_chunk_datasource.py     # Core data logic
└── network_tokenization.py        # Tokenization
```

Minimal MaxText modifications (~20 lines across 3 files):
- `input_pipeline_interface.py` - Backend registration
- `configs/types.py` - Config types
- `_grain_data_processing.py` - Cleanup (removed special cases)

## Data Preparation

See [data-pipeline.md](data-pipeline.md) for full details.

Quick version:
```bash
# Preprocess parquet files into ArrayRecord format
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/raw/*.parquet" \
  --output data/probe_rows \
  --max-row-size-mb 8
```

## Performance

Expected throughput on B200 GPU:
- **320K-850K tokens/sec** with 8 CPUs
- **<5% padding overhead**
- **Multiprocessing enabled** for parallel tokenization

See [troubleshooting/performance.md](troubleshooting/performance.md) for optimization tips.
