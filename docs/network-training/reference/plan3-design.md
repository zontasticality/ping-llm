# Probe-Centric Big-Row Pipeline (Plan 3)

## Overview

Dataset characteristics:
  - 200M latency measurements (sampled from 35B total)
  - Parquet source format
  - Target: ArrayRecord shards for efficient random access

Design goals:
  - Minimize I/O overhead during training
  - Maximize data reuse per row read
  - Enable multi-scale temporal learning
  - Support probe-centric federated deployment


## Preprocessing Pipeline

Input:
  - Sharded Parquet files with measurements

Grouping strategy:
  - Group by: src_addr
  - Order by: event_time (within each group)
  - Result: One probe's measurement history per group

Serialization format:
  - Compact binary encoding (pre-tokenization format)
    - Latency: 2-byte encoding (matches tokenizer format)
    - Timestamp: u64 unix epoch seconds
    - Other fields: similarly compressed
  - Goal: Information density comparable to tokenized form

Row splitting policy:
  - Threshold: 8MB per row
  - Split behavior:
    - When group exceeds 8MB, finish current measurement
    - Start new row with remaining measurements
    - Each split creates a separate ArrayRecord row
  - Rationale: Balance between I/O efficiency and memory overhead
  - Implication: Large probes span multiple rows

Output format:
  - ArrayRecord shards
  - Row schema:
    - src_id: probe identifier
    - measurements: binary blob (serialized measurement array)
    - n_measurements: count
    - metadata: time span, etc.


## Training Pipeline

### Row Sampling
  - Random row selection (not src_addr selection)
  - Simpler implementation
  - Natural data reuse across all probes

### Context Generation

K contexts per row:
  - K = min(ceil(n_measurements / 30), 16)
  - Rationale: ~30 tokens/measurement average
  - Minimum: 1 context (small rows)
  - Maximum: 16 contexts (large rows)

For each of K contexts:

  **Small row path** (n_measurements < 1024/30 ≈ 34):
    - Tokenize entire row
    - If measurements remain after first 1024-token context:
      - Create additional contexts (with padding) until exhausted
    - Result: Full data coverage, minimal padding

  **Large row path** (n_measurements ≥ 34):
    - Sample window size:
      - Distribution: log-uniform over [1, n_measurements]
      - Result: Multi-scale temporal coverage
    - Sample window offset:
      - Range: [0, n_measurements - window_size]
      - Result: Random temporal alignment
    - Sample measurements from window:
      - Method: Random subset from window
      - Constraint: Sorted by timestamp before tokenization
      - Target: Enough to fill ~1024 tokens
    - Result: Dense temporal subsampling

### Timestamp Mode Selection

Per context, select mode (stochastic):
  - Full timestamps (40%):
    - Keep all timestamps
    - Enable delta encoding
  - Partial timestamps (30%):
    - Extract random percentage (10-90%) of measurements
    - Extracted: remove timestamp during tokenization
      - Randomize measurement order of non-timestamp measurements and shuffle back into ordered measurements 
    - Non-extracted: include timestamp, ordered by timestamp
  - No timestamps (30%):
    - Remove all timestamps
    - Randomize measurement order (not just field order)
    - Model learns from unordered measurements

### Tokenization

Measurement buffer processing:
  - Field order randomization:
    - Randomize field sequence per measurement
    - Enables joint distribution learning
  - Timestamp encoding:
    - Delta encoding when previous timestamp exists
    - Absolute encoding otherwise
  - Padding:
    - Pad to 1024 tokens as needed

Output format:
  - Standard MaxText batch:
    - inputs: [batch_size, 1024] int32
    - targets: [batch_size, 1024] int32
    - segmentation: [batch_size, 1024] int32 (1=real, 0=pad)
    - positions: [batch_size, 1024] int32


## Key Design Decisions

Row granularity:
  - Not fixed-time buckets (PLAN_1)
  - Not one-probe-per-row (PLAN_2)
  - Hybrid: probe groups with size cap
  - Trade-off: Simpler sampling vs. clean probe boundaries

Multi-scale learning:
  - Log-uniform window sizes → variable time spans
  - Random measurement subsampling → stochastic coverage
  - Complements timestamp masking → robust temporal reasoning

Data augmentation:
  - K contexts per row → explicit reuse
  - Random window sampling → infinite variations
  - Field order randomization → permutation invariance
  - Timestamp masking → observability robustness

Padding efficiency:
  - Adaptive measurement selection → target 1024 tokens
  - Expected waste: <5% (vs. 50-90% in PLAN_1)


## Implementation Notes

Preprocessing:
  - Use DuckDB or Spark for group-by-src_addr
  - Stream serialization to ArrayRecord
  - Track row sizes for 8MB splits

Training:
  - Grain RandomAccessDataSource for ArrayRecord
  - RandomMapTransform for context generation
  - All sampling logic in transform (no preprocessing)

Testing:
  - Verify K calculation matches target tokens
  - Check padding distribution (expect <5%)
  - Validate temporal order preservation
  - Measure contexts/row distribution
