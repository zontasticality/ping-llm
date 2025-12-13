# Ping LLM: Network Measurement Transformer

Training a decoder-only Transformer on network latency measurements using MaxText.

## Overview

This project trains a generative model over RIPE Atlas ping measurements to learn the joint distribution of:
- Source/Destination IP addresses (IPv4/IPv6)
- Round-trip times (RTT)
- Packet sizes and error counts
- Temporal patterns

**Dataset:** 100M network measurements (1.1GB Parquet, 28 days, June-July 2025)

**Model:** ~100M parameter decoder-only Transformer with custom byte-level tokenization

## Quick Start

### 1. Test Tokenization

```bash
# Run standalone test (no dependencies required)
python scripts/test_tokenization_standalone.py

# Expected output: ✅ ALL TESTS PASSED
```

### 2. Shard the Dataset

Split the single Parquet file into train/val/test shards:

```bash
python scripts/shard_parquet.py \
  --input data/training_data.parquet \
  --output data/sharded \
  --train-shards 200 \
  --val-shards 25 \
  --test-shards 25
```

This creates:
```
data/sharded/
  train/shard_0000.parquet ... shard_0199.parquet  (~400k rows each)
  val/shard_0000.parquet ... shard_0024.parquet
  test/shard_0000.parquet ... shard_0024.parquet
```

### 3. Validate Tokenization on Real Data

```bash
# Requires: duckdb or pyarrow
python scripts/local_grain_smoke.py --samples 5
```

### 4. Train with MaxText

**Local CPU test:**
```bash
export DECOUPLE_GCLOUD=TRUE
python -m MaxText.train \
  maxtext/configs/latency_parquet.yml \
  run_name=local_test \
  base_output_directory=/tmp/maxtext_out \
  hardware=cpu \
  per_device_batch_size=1 \
  steps=20
```

**SLURM cluster:**
```bash
sbatch scripts/slurm_train_maxtext.sh
```

## Project Structure

```
ping-llm/
├── PLAN.md                          # Detailed implementation plan
├── PROJECT_README.md                # This file
├── tokenization.py                  # Core tokenization logic
├── data/
│   ├── training_data.parquet        # Original dataset (100M rows)
│   └── sharded/                     # Sharded data (created by shard_parquet.py)
│       ├── train/*.parquet
│       ├── val/*.parquet
│       └── test/*.parquet
├── maxtext/
│   └── configs/
│       ├── latency_model_100m.yml   # Model architecture config
│       └── latency_parquet.yml      # Training run config
└── scripts/
    ├── shard_parquet.py             # Shard Parquet into train/val/test
    ├── local_grain_smoke.py         # Test Grain + tokenization
    ├── test_tokenization_standalone.py  # Standalone tokenization tests
    ├── test_tokenization.py         # DuckDB-based tests
    └── slurm_train_maxtext.sh       # SLURM training script
```

## Tokenization

### Vocabulary (266 tokens)

- **Role tokens (0-9):** `MeasurementStart`, `SrcIpStart`, `DestIpStart`, `Ipv4Start`, `Ipv6Start`, `RttStart`, `SizeStart`, `ErrorCountStart`, `TimestampStart`, `MsmIdStart`
- **Byte tokens (10-265):** `Byte0` ... `Byte255`

### Token Sequence Format

Each measurement is encoded as:
```
[MeasurementStart] + shuffled_field_blocks
```

Field blocks (order randomized per measurement):
- **Source IP:** `[SrcIpStart, Ipv4Start/Ipv6Start, byte_tokens...]`
- **Dest IP:** `[DestIpStart, Ipv4Start/Ipv6Start, byte_tokens...]`
- **RTT:** `[RttStart, 8 bytes as float64 big-endian]`
- **Size:** `[SizeStart, 2 bytes as uint16 big-endian]`
- **Errors:** `[ErrorCountStart, 1 byte as uint8]`
- **Timestamp:** `[TimestampStart, 8 bytes as int64 big-endian]`
- **Measurement ID:** `[MsmIdStart, 8 bytes as int64 big-endian]`

**Token counts:**
- IPv4 measurements: ~42 tokens
- IPv6 measurements: ~66 tokens

### Example

```python
from tokenization import encode_measurement
from datetime import datetime, timezone

row = {
    'msm_id': 12345,
    'event_time': datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc),
    'src_addr': '192.0.2.1',
    'dst_addr': '8.8.8.8',
    'ip_version': 4,
    'rtt': 42.5,
    'size': 64,
    'packet_error_count': 0,
}

tokens = encode_measurement(row)
# Returns: [0, 1, 3, 202, 10, 12, 11, ...]  (45 tokens total)
```

## Model Architecture

**Decoder-only Transformer (GPT-style)**

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Embedding dim | 768 |
| Attention heads | 12 (64 dim each) |
| MLP hidden dim | 3072 |
| Vocab size | 266 |
| Max seq length | 1024 tokens (~16-24 measurements) |
| Total params | ~100M |

See `maxtext/configs/latency_model_100m.yml` for full configuration.

## Data Pipeline

1. **Raw data:** `data/training_data.parquet` (100M rows, 1.1GB)
2. **Sharding:** `scripts/shard_parquet.py` → 250 shards (80/10/10 split)
3. **Tokenization:** On-the-fly via `encode_measurement()` (Phase 1)
4. **Loading:** Grain + MaxText (Phase 2)
5. **Pre-tokenization:** Optional optimization (Phase 4)

## Implementation Status

### ✅ Phase 0: Missing Scaffolding (COMPLETED)

- [x] `tokenization.py` - IP parsing, field encoding, deterministic shuffling
- [x] `scripts/shard_parquet.py` - Data splitting
- [x] `scripts/local_grain_smoke.py` - Grain testing
- [x] `scripts/test_tokenization_standalone.py` - Validation tests
- [x] MaxText configs (model + run)
- [x] SLURM training script
- [x] Updated `duckdb_sample.py` with correct date ranges

### 🔄 Phase 1: Data & Tokenization Validation (NEXT)

- [ ] Run `local_grain_smoke.py` on actual data
- [ ] Validate token ID ranges on full dataset
- [ ] Verify IP parsing correctness (spot-check IPv4/IPv6)
- [ ] Test randomized field order reproducibility
- [ ] Profile tokenization throughput

### 📋 Phase 2: Local MaxText Sanity

- [ ] Minimal CPU run (20 steps)
- [ ] Verify Grain integration
- [ ] Check for shape/vocab issues

### 📋 Phase 3: SLURM Training

- [ ] Single-node GPU training (1-4 GPUs)
- [ ] Tune batch size and grain workers
- [ ] Monitor loss curves

### 📋 Phase 4: Pre-tokenization

- [ ] Create `scripts/pretokenize_to_parquet.py`
- [ ] Generate pre-tokenized shards
- [ ] Re-profile throughput

### 📋 Phase 5-6: Modeling & Inference

See `PLAN.md` for details.

## Open Questions

See `PLAN.md` section 8 for detailed discussion:

1. **RTT encoding strategy:** Raw float64 vs. fixed-point vs. log-scale vs. clipped
2. **Timestamp precision:** Seconds vs. milliseconds
3. **Packing strategy:** How many measurements per 1024-token sequence?
4. **Model scaling:** 20M / 100M / 300M parameters based on dataset size
5. **Evaluation metrics:** NLL, field-specific accuracy, conditional queries

## Dataset Info

**Source:** `data/training_data.parquet`

| Metric | Value |
|--------|-------|
| Rows | 100,005,691 |
| Size | 1.1 GB |
| Date range | 2025-06-24 to 2025-07-22 (28 days) |
| Measurements | 66,275 distinct msm_id |
| IPv4 | 57.7M (57.7%) |
| IPv6 | 42.3M (42.3%) |

**Value Ranges:**
- RTT: -1.0 (failed) to 302,281 ms; p50=50ms, p90=239ms, p99=343ms
- Size: mostly 64 bytes, max 2000
- Errors: mostly 0, max 16

## Dependencies

**Core:**
- Python 3.8+
- MaxText (JAX-based)
- PyArrow or DuckDB (for Parquet I/O)

**Optional:**
- Grain (for data loading)
- CUDA/cuDNN (for GPU training)
- SLURM (for cluster training)

## Contributing

This project follows the implementation plan in `PLAN.md`. To contribute:

1. Check the current phase in PLAN.md
2. Review open questions in section 8
3. Run tests: `python scripts/test_tokenization_standalone.py`
4. Submit changes with updated PLAN.md status

## Citation

Based on RIPE Atlas measurement data. See PLAN.md for project details and design decisions.
