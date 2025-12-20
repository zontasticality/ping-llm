# Architecture and Design

This document covers the architecture of the network training pipeline (PLAN_3).

## Overview

The network training pipeline trains transformer models on network latency measurements rather than text. It implements a custom backend for MaxText with minimal modifications to the core framework.

## Design Goals

1. **Minimize I/O overhead** during training
2. **Maximize data reuse** per row read
3. **Enable multi-scale temporal learning**
4. **Support probe-centric federated deployment**
5. **Maintain upstream MaxText compatibility**

## PLAN_3: Probe-Centric Big-Row Pipeline

### Key Characteristics

**Dataset**: 200M latency measurements (sampled from 35B total)
**Source**: Parquet files
**Target**: ArrayRecord shards for efficient random access

### Data Format

**Preprocessing output** (ArrayRecord schema):
```python
{
    'src_id': int64,                    # Probe identifier
    'measurements': binary,              # PyArrow IPC serialized RecordBatch
    'n_measurements': int32,             # Count
    'time_span_seconds': float64,       # last_time - first_time
    'first_timestamp': timestamp('us'), # Metadata
    'last_timestamp': timestamp('us'),  # Metadata
}
```

**Measurements schema** (inside binary blob):
```python
{
    'event_time': timestamp('us'),
    'dst_addr': string,
    'ip_version': int8,
    'rtt': float32,
}
```

### Row Splitting Policy

- **Threshold**: 8MB per row
- **Split behavior**: When group exceeds 8MB, finish current measurement and start new row
- **Result**: Large probes span multiple rows, small probes fit in one row
- **Rationale**: Balance between I/O efficiency and memory overhead

### Training Pipeline

#### Row Sampling
- Random row selection (not src_addr selection)
- Simpler implementation
- Natural data reuse across all probes

#### Context Generation

**K contexts per row**:
```python
K = min(ceil(n_measurements / 30), 16)
```
- Rationale: ~30 tokens/measurement average
- Minimum: 1 context (small rows)
- Maximum: 16 contexts (large rows)

**For each of K contexts**:

*Small row path* (n_measurements < 34):
- Tokenize entire row
- Create additional contexts if needed (with padding)
- Result: Full data coverage, minimal padding

*Large row path* (n_measurements ≥ 34):
- Sample window size: log-uniform over [1, n_measurements]
- Sample window offset: random within valid range
- Sample measurements from window
- Maintain timestamp order
- Result: Dense temporal subsampling

#### Timestamp Mode Selection

Per context, select mode (stochastic):
- **Full timestamps (40%)**: Keep all timestamps, enable delta encoding
- **Partial timestamps (30%)**: Extract 10-90% of measurements, remove their timestamps
- **No timestamps (30%)**: Remove all timestamps, randomize measurement order

This teaches the model to work with incomplete temporal information.

#### Tokenization

- **Field order randomization**: Randomize field sequence per measurement
- **Timestamp encoding**: Delta encoding when previous timestamp exists
- **Padding**: Pad to 1024 tokens as needed
- **Output**: Standard MaxText batch format

## Implementation

### Module Structure

```
src/MaxText/input_pipeline/
├── _network_data_processing.py      # Backend interface (253 lines)
│   ├── make_network_train_iterator()
│   └── make_network_eval_iterator()
├── probe_chunk_pipeline.py          # Dataset builder (111 lines)
│   └── build_probe_chunk_dataset()
├── _probe_chunk_datasource.py       # Core logic (470 lines)
│   ├── ProbeRowDataSource (reads ArrayRecord)
│   └── ProbeRowSampler (generates contexts)
└── network_tokenization.py          # Tokenization (179 lines)
    └── encode_measurement()
```

### MaxText Integration

**Minimal modifications** (~20 lines across 3 files):

**1. input_pipeline_interface.py**:
```python
from MaxText.input_pipeline._network_data_processing import (
    make_network_train_iterator, make_network_eval_iterator
)

dataset_type_to_train_eval_iterator = {
    # ... existing backends ...
    "network": (make_network_train_iterator, make_network_eval_iterator),
}
```

**2. configs/types.py**:
```python
class DatasetType(str, Enum):
    NETWORK = "network"

class NetworkDataset(BaseModel):
    network_data_format: str = "probe_chunks"
    network_train_files: PathStr = ""
    network_eval_files: PathStr = ""
    grain_debug_mode: bool = False
    grain_visualization_dir: PathStr = ""
```

**3. _grain_data_processing.py**:
- Removed special case handling for network data
- Cleaner grain backend (69 lines removed)

## Key Design Decisions

### Why not fixed-time buckets? (rejected PLAN_1)
- High padding overhead (50-90%)
- Doesn't leverage probe locality
- Poor temporal coverage

### Why not one-probe-per-row? (rejected PLAN_2)
- Excessive I/O for small probes
- Memory issues for large probes
- Complex splitting logic

### Why probe groups with size cap? (PLAN_3)
- Balances I/O efficiency and memory
- Simple sampling logic
- Natural data augmentation
- Multi-scale learning via log-uniform windows

### Multi-scale Learning Benefits

**Log-uniform window sizes** → Variable time spans:
- Small windows: Learn short-term patterns
- Large windows: Learn long-term trends
- Distribution ensures coverage across scales

**Random measurement subsampling** → Stochastic coverage:
- Different samples from same row each epoch
- Infinite variations without storing multiple copies

**Timestamp masking** → Robust temporal reasoning:
- 40% full: Learn delta encoding
- 30% partial: Learn with incomplete timestamps
- 30% none: Learn from unordered measurements

## Performance Characteristics

### Padding Efficiency
- Expected waste: <5% (vs 50-90% in PLAN_1)
- Achieved via adaptive measurement selection targeting 1024 tokens

### I/O Efficiency
- One 8MB row read → up to 16 training contexts
- Multiprocessing for parallel tokenization
- Grain worker threads for parallel I/O

### Memory Efficiency
- Streaming deserialization (no full dataset in RAM)
- Lazy ArrayRecord reading
- Configurable RAM budget for multiprocessing

## Refactoring History

The network backend was refactored from embedded grain special cases to a clean separate backend following MaxText patterns. See the git history for evolution:

1. **Initial**: Special cases in `_grain_data_processing.py`
2. **Intermediate**: Separate `_network_grain_integration.py`
3. **Final**: Complete backend in `_network_data_processing.py`

This maintains upstream compatibility while providing full network training capabilities.

## References

- `DATA_LOADING_PLAN_3_CLEAN.md` (original plan)
- `PLAN_3_FILE_SPECS.md` (detailed specs)
- `REFACTORING_WRITEUP.md` (refactoring details)
