# PLAN 2.0: Network Measurement Transformer

**Status:** Active development plan (supersedes PLAN.md)
**Date:** 2025-12-13
**Focus:** Optimized tokenization + generalization-focused architecture for network path selection and IP inference

---

## Executive Summary

This project trains a decoder-only Transformer to learn the joint distribution of network measurements (source IP, destination IP, RTT, timestamp). The model is designed for three key capabilities:

1. **Forward modeling:** Given IP pair, predict RTT distribution (network topology learning)
2. **Inverse search:** Given target RTT, generate candidate IPs (path selection via multi-step reasoning)
3. **IP structure learning:** Predict IP allocations from partial context (hierarchical CIDR/ASN patterns)

**Key innovations:**
- **Compact tokenization:** 64% reduction in sequence length enables 3x more context
- **In-context network localization:** 64 measurements characterize vantage point (datacenter vs residential, geographic region, ISP characteristics)
- **Generalization-focused architecture:** Small MLP ratio forces learning routing rules rather than memorizing specific pairs

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Tokenization Schema](#2-tokenization-schema)
3. [Model Architecture](#3-model-architecture)
4. [Training Strategy](#4-training-strategy)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Evaluation & Metrics](#6-evaluation--metrics)
7. [Open Questions](#7-open-questions)

---

## 1. Dataset

### 1.1 Current Dataset

**Location:** `data/training_data.parquet`

**Statistics:**
- **Size:** 1.1 GB, 100,005,691 rows
- **Date range:** 2025-06-24 → 2025-07-22 (28 days)
- **IP distribution:** IPv4 57.7M (57.7%) / IPv6 42.3M (42.3%)
- **Measurements:** 66,275 distinct msm_id (probe streams)

**Schema (Parquet columns):**
```
msm_id: int64              - Measurement stream identifier (dataset artifact, not used)
event_time: timestamp[ns]  - Event timestamp with timezone
src_addr: string           - Source IP address ("192.0.2.1" or "2001:db8::1")
dst_addr: string           - Destination IP address (same format)
ip_version: int64          - IP version (4 or 6)
rtt: double                - Round-trip time in milliseconds (-1.0 = failed probe)
size: int64                - Packet size in bytes (dataset artifact, not used)
packet_error_count: int64  - Packet error count (dataset artifact, not used)
```

**Value ranges (observed):**
- `rtt`: min -1.0 (failed), max 302,281 ms, p50 50ms, p90 239ms, p99 343ms
- Consecutive measurements typically 60-300 seconds apart (same msm_id)

**Core network observations (used for training):**
- Source IP (IPv4/IPv6)
- Destination IP (IPv4/IPv6)
- RTT or failure indicator
- Timestamp

### 1.2 Dataset Sharding

**Current:** Single 1.1GB file
**Target:** 200 shards for better shuffling and parallelism

**Implementation:** `scripts/shard_parquet.py`
```bash
python scripts/shard_parquet.py \
  --input data/training_data.parquet \
  --output data/sharded \
  --train-shards 180 \    # 90% train (~500k rows/shard)
  --test-shards 20        # 10% test (~500k rows/shard)
```

**Rationale for no validation set:**
- Single training run with fixed architecture (not doing extensive hyperparameter search)
- Can evaluate on test set periodically during training for monitoring
- Validation set typically used for hyperparameter tuning and early stopping
- For this project, we'll train to completion and evaluate on held-out test set

**Stratification:** By time to ensure temporal coverage in both splits

---

## 2. Tokenization Schema

### 2.1 Design Principles

1. **Minimize sequence length** to maximize in-context learning capacity
2. **Remove dataset artifacts** (msm_id, size, error_count)
3. **Efficient encoding** of numeric values (2-byte floats, delta timestamps)
4. **Deterministic field shuffling** to force joint distribution learning

### 2.2 Token Vocabulary

**Total:** 267 tokens (11 role + 256 byte tokens)

**Role tokens (IDs 0-10):**
```
0:  <MeasurementStart>   - Measurement boundary
1:  <SrcIPv4>            - Source IPv4 address follows (4 bytes)
2:  <SrcIPv6>            - Source IPv6 address follows (16 bytes)
3:  <DstIPv4>            - Destination IPv4 address follows (4 bytes)
4:  <DstIPv6>            - Destination IPv6 address follows (16 bytes)
5:  <TimestampAbs>       - Absolute timestamp follows (8 bytes)
6:  <TimestampDelta1>    - 1-byte timestamp delta (0-255 seconds, ~95% of cases)
7:  <TimestampDelta4>    - 4-byte timestamp delta (large gaps >255s)
8:  <RttStart>           - RTT value follows (2 bytes, log-compressed)
9:  <ThroughputStart>    - Throughput follows (future datasets)
10: <Failed>             - Connection failed (no RTT/throughput)
```

**Byte tokens (IDs 11-266):**
```
11-266: <Byte0> ... <Byte255>  - Represent raw byte values (0x00-0xFF)
```

### 2.3 Encoding Formats

#### IP Addresses
```
IPv4: <SrcIPv4/DstIPv4> + 4 bytes           (5 tokens total)
IPv6: <SrcIPv6/DstIPv6> + 16 bytes          (17 tokens total)
```

**Note:** Merge SrcIp/DstIp with IPv4/IPv6 saves 2 tokens per measurement vs old scheme

#### RTT (Round-Trip Time)
```
Format: <RttStart> + 2 bytes (5-bit exponent + 11-bit mantissa)

Encoding: value_μs = mantissa × 2^exponent
  - exponent: 5 bits (0-31)
  - mantissa: 11 bits (0-2047)
  - Packing: EEEEE MMM | MMMM MMMM
            (5exp)(3msb)|(8lsb mantissa)

Range:
  - Min: 1 × 2^0 = 1 microsecond (local network)
  - Max: 2047 × 2^31 = 4.4 trillion μs = 51 days
  - Earth-Mars (closest): ~180 seconds ✓
  - Earth-Mars (farthest): ~1300 seconds (22 min) ✓

Precision: ~0.049% relative error (1/2047)
  - At 1ms (1000μs):    ±0.5μs
  - At 100ms (100kμs):  ±50μs
  - At 1s (1Mμs):       ±500μs
  - At 22min (Mars):    ±650ms (sufficient for interplanetary!)

Conversion:
  ms → μs: multiply by 1000
  μs → ms: divide by 1000
```

**Rationale:**
- Wide dynamic range (1μs to 51 days) supports future interplanetary networks
- 11-bit mantissa provides sufficient precision for all terrestrial measurements
- Powers-of-2 encoding is faster than logarithms
- Future-proof for Mars, lunar, or other space-based network measurement datasets

#### Timestamps
```
First measurement:  <TimestampAbs> + 8 bytes (Unix epoch seconds)
Subsequent (delta < 256s):  <TimestampDelta1> + 1 byte
Subsequent (delta ≥ 256s):  <TimestampDelta4> + 4 bytes
```

**Optimization:** Consecutive measurements typically 60-300s apart → 1-byte delta covers 95%+ cases

#### Connection Result
```
Success: <RttStart> + 2 bytes (latency)
Failure: <Failed> (1 token, replaces RTT=-1.0 encoding)
Future:  <ThroughputStart> + 2 bytes (for extended datasets)
```

### 2.4 EBNF Grammar

Complete formal specification:

```ebnf
# ============================================================================
# Network Measurement Tokenization Schema
# ============================================================================

# Token vocabulary (267 total)
Token      ::= <MeasurementStart>          # 0: Measurement boundary
             | <SrcIPv4>                   # 1: Source IPv4 address follows
             | <SrcIPv6>                   # 2: Source IPv6 address follows
             | <DstIPv4>                   # 3: Destination IPv4 address follows
             | <DstIPv6>                   # 4: Destination IPv6 address follows
             | <TimestampAbs>              # 5: Absolute timestamp (8 bytes)
             | <TimestampDelta1>           # 6: Delta timestamp (1 byte)
             | <TimestampDelta4>           # 7: Delta timestamp (4 bytes)
             | <RttStart>                  # 8: RTT follows (2 bytes)
             | <ThroughputStart>           # 9: Throughput follows (future)
             | <Failed>                    # 10: Connection failed
             | <Byte0> ... <Byte255>       # 11-266: Data bytes

# Primitive types
U8         ::= <Byte0> | <Byte1> | ... | <Byte255>
U16        ::= U8 U8                       # Big-endian uint16
U32        ::= U8 U8 U8 U8                 # Big-endian uint32
U64        ::= U8{8}                       # Big-endian uint64
Float16    ::= U8 U8                       # 5-bit exponent + 11-bit mantissa
                                           # Format: mantissa × 2^exponent (in microseconds)
                                           # Range: 1μs to 51 days
                                           # Precision: ~0.049% relative error

# Network fields
SrcIPv4    ::= <SrcIPv4> U8{4}            # 5 tokens
SrcIPv6    ::= <SrcIPv6> U8{16}           # 17 tokens
DstIPv4    ::= <DstIPv4> U8{4}            # 5 tokens
DstIPv6    ::= <DstIPv6> U8{16}           # 17 tokens

SrcIp      ::= SrcIPv4 | SrcIPv6
DstIp      ::= DstIPv4 | DstIPv6

Timestamp  ::= <TimestampAbs> U64         # 9 tokens: first measurement
             | <TimestampDelta1> U8       # 2 tokens: delta <256s (95% of cases)
             | <TimestampDelta4> U32      # 5 tokens: delta ≥256s (rare)

Result     ::= <RttStart> Float16         # 3 tokens: successful probe
             | <ThroughputStart> Float16  # 3 tokens: future datasets
             | <Failed>                   # 1 token: failed probe

# Measurement structure
Field      ::= SrcIp | DstIp | Timestamp | Result

# Timestamp is optional - measurements can have 3 or 4 fields
# Fields are deterministically shuffled (except MeasurementStart which always comes first)
Measurement ::= <MeasurementStart> Field{3,4}

# Context window (multiple measurements for in-context learning)
# Some measurements may include timestamps, others may not (see Training Modes section)
Context    ::= Measurement+
```

### 2.5 Sequence Length Analysis

**IPv4 measurement (with timestamp):**
```
Field               | First | Subsequent (delta)
--------------------|-------|-------------------
MeasurementStart    |   1   |   1
SrcIPv4             |   5   |   5
DstIPv4             |   5   |   5
Rtt (success)       |   3   |   3
Timestamp           |   9   |   2
--------------------|-------|-------------------
Total (success)     |  23   |  16
Total (failed)      |  21   |  14
```

**IPv4 measurement (NO timestamp):**
```
Field               | Tokens
--------------------|--------
MeasurementStart    |   1
SrcIPv4             |   5
DstIPv4             |   5
Rtt (success)       |   3
--------------------|--------
Total (success)     |  14
Total (failed)      |  12
```

**IPv6 measurement:**
```
First (success):     47 tokens  (23 + 24 for IPv6)
Subsequent (delta):  40 tokens  (16 + 24 for IPv6)
```

**Comparison to old scheme:**
```
Old IPv4 first:      45 tokens → 49% reduction to 23
Old IPv4 subsequent: 45 tokens → 64% reduction to 16
```

**Context capacity (1024 tokens):**
```
Old scheme:     ~22 measurements
New scheme:     ~64 measurements (IPv4 stream with deltas)
                ~45 measurements (mixed IPv4/IPv6)
                ~25 measurements (IPv6 stream)

Improvement: 3x more context for network localization!
```

### 2.6 Tokenization Implementation

**File:** `tokenization.py` (to be updated)

**Key functions:**
```python
def encode_rtt_exponent_mantissa(rtt_ms: float) -> List[int]:
    """
    Encode RTT as 5-bit exponent + 11-bit mantissa (microseconds).

    Format: value_μs = mantissa × 2^exponent
    Range: 1μs to 51 days
    Precision: ~0.049% relative error

    Returns: [RTT_START, byte1, byte2]
    """
    if rtt_ms < 0:
        return [FAILED]  # Use dedicated token for failed probes

    rtt_us = max(1.0, rtt_ms * 1000)  # Convert ms to μs, min 1μs

    # Find exponent (powers of 2)
    exponent = 0
    mantissa_float = rtt_us
    while mantissa_float >= 2048 and exponent < 31:
        mantissa_float /= 2
        exponent += 1

    mantissa = min(int(mantissa_float), 2047)

    # Pack: EEEEE MMM | MMMM MMMM
    #       5exp  3msb  8lsb
    byte1 = (exponent << 3) | (mantissa >> 8)
    byte2 = mantissa & 0xFF

    return [RTT_START, byte_to_token(byte1), byte_to_token(byte2)]

def decode_rtt_exponent_mantissa(byte1: int, byte2: int) -> float:
    """
    Decode RTT from 5-bit exponent + 11-bit mantissa.

    Returns: RTT in milliseconds
    """
    exponent = byte1 >> 3
    mantissa = ((byte1 & 0x07) << 8) | byte2

    rtt_us = mantissa * (2 ** exponent)
    rtt_ms = rtt_us / 1000

    return rtt_ms

def encode_timestamp_delta(
    event_time,
    prev_time,
    dataset_start
) -> List[int]:
    """Encode timestamp as absolute or delta."""
    if prev_time is None:
        # First measurement: absolute timestamp
        timestamp_sec = int(event_time.timestamp())
        return [TIMESTAMP_ABS] + encode_u64(timestamp_sec)

    delta_sec = int((event_time - prev_time).total_seconds())

    if delta_sec < 256:
        # 1-byte delta (most common case)
        return [TIMESTAMP_DELTA1, byte_to_token(delta_sec)]
    else:
        # 4-byte delta (rare)
        return [TIMESTAMP_DELTA4] + encode_u32(delta_sec)

def encode_measurement(
    row: Dict[str, Any],
    prev_timestamp: Optional = None,
    dataset_start: Optional = None
) -> List[int]:
    """
    Encode a single network measurement.

    Args:
        row: Parquet row with src_addr, dst_addr, ip_version, rtt, event_time
        prev_timestamp: Previous measurement timestamp (for delta encoding)
        dataset_start: Dataset start time (for absolute timestamps)

    Returns:
        List of token IDs
    """
    # Extract fields
    src_addr = row['src_addr']
    dst_addr = row['dst_addr']
    ip_version = row['ip_version']
    rtt = row['rtt']
    event_time = row['event_time']

    # Encode IP addresses (merged src/dst with IPv4/IPv6)
    src_ip_block = encode_ip_merged(src_addr, ip_version, is_src=True)
    dst_ip_block = encode_ip_merged(dst_addr, ip_version, is_src=False)

    # Encode result (RTT or failed)
    if rtt < 0:
        result_block = [FAILED]
    else:
        result_block = encode_rtt_exponent_mantissa(rtt)

    # Encode timestamp (absolute or delta)
    timestamp_block = encode_timestamp_delta(
        event_time, prev_timestamp, dataset_start
    )

    # Collect field blocks
    field_blocks = [
        src_ip_block,
        dst_ip_block,
        result_block,
        timestamp_block,
    ]

    # Deterministic shuffle using (src_ip, dst_ip, timestamp) as seed
    shuffle_seed = compute_shuffle_seed(src_addr, dst_addr, event_time)
    shuffled_blocks = shuffle_blocks_deterministic(field_blocks, shuffle_seed)

    # Build final sequence
    tokens = [MEASUREMENT_START]
    for block in shuffled_blocks:
        tokens.extend(block)

    return tokens
```

---

## 3. Model Architecture

### 3.1 Design Philosophy

**Objective:** Maximize generalization to unseen residential networks and novel IP pairs

**Key insight:** The model should learn **routing rules and network topology**, not memorize specific (IP₁, IP₂) → RTT mappings.

**Architectural choices:**
1. **Deep layers** (20): Enable multi-step reasoning for RTT→IP search
2. **Small MLP ratio** (3.2x instead of 4x): Reduce memorization capacity
3. **Moderate embedding** (640): Rich enough for byte semantics, not excessive for small vocab
4. **Large context** (1024): In-context learning for network localization

### 3.2 Model Hyperparameters

**Target:** ~100M parameters

```yaml
# Vocabulary
vocab_size: 267                # 11 role + 256 byte tokens

# Architecture (decoder-only Transformer)
base_emb_dim: 640              # Embedding dimension
base_num_decoder_layers: 20    # Deep for multi-step reasoning
base_num_query_heads: 10       # 64-dim heads
base_num_kv_heads: 10          # Standard attention (not MQA/GQA)
head_dim: 64                   # Standard head dimension
base_mlp_dim: 2048             # 3.2x ratio (small for generalization)

# Context
max_target_length: 1024        # Large context for in-context learning
max_prefill_predict_length: 1024

# Regularization
dropout_rate: 0.1
weight_decay: 0.01

# Precision
dtype: bfloat16                # Standard for modern accelerators

# Attention
attention: dot_product         # Standard scaled dot-product
scan_layers: true              # Memory-efficient layer scanning
```

**Parameter breakdown:**
```
Embeddings:  267 × 640 × 2 = 0.34M (input + output)
Per layer:   4 × 640² + 2 × 640 × 2048 = 1.64M + 2.62M = 4.26M
20 layers:   20 × 4.26M = 85.2M
Total:       ~85.5M parameters
```

**Note:** Slightly under 100M budget allows headroom for additional parameters (layer norm, positional encodings if added).

### 3.3 Rationale for Architectural Choices

#### 1. Embedding Dimension: 640

**Why not smaller (256)?**
- Small vocab (267) but **high polysemy**: Byte `0xC0` (192) means different things in different contexts:
  - IP position 1: "192.x.x.x network" (North America)
  - RTT encoding: part of log-compressed float
  - Timestamp encoding: part of Unix epoch
- Need sufficient dimensionality to separate these contexts

**Why not larger (1024)?**
- Diminishing returns for small vocab
- Want to invest parameters in depth, not width

**640 = sweet spot:** 2.4 bits/dimension for 267 tokens, sufficient for rich representations

#### 2. Depth: 20 Layers

**Critical for inverse search (RTT→IP):**

Multi-step reasoning example:
```
Input:   "Generate IPs with RTT ~50ms to 8.8.8.8"
Layer 1-4:   "50ms suggests North America or Europe"
Layer 5-8:   "8.8.8.8 is Google DNS, widely peered"
Layer 9-12:  "Likely need major metro areas with good connectivity"
Layer 13-16: "Refine to specific ASNs (universities, datacenters)"
Layer 17-20: "Output specific IP prefixes in plausible ranges"
```

**Hierarchical IP learning:**
```
Layer 1-5:   Learn /8 blocks (continent-level)
Layer 6-10:  Learn /16 patterns (country/region)
Layer 11-15: Learn /24 subnets (city/ISP)
Layer 16-20: Learn host allocations
```

**Research support:** Compositional/hierarchical tasks benefit from depth >> width ([Transformer Circuits, Anthropic])

#### 3. MLP Dimension: 2048 (3.2x ratio)

**Standard LLM:** 4x ratio (e.g., 640 emb → 2560 MLP)

**Our choice:** 3.2x ratio (640 → 2048) **for generalization**

**Why smaller MLP?**
- **MLPs are memorization layers:** Store lookup-table-like mappings
- Large MLPs enable memorizing specific (IP₁, IP₂, timestamp) → RTT triples
- **We want topology learning, not memorization:**
  - "AS X peers with AS Y → low latency"
  - "Transatlantic links → +80-120ms"
  - "Residential CGNAT → +10-30ms variance"

**Trade-off:**
- Smaller MLP → worse at memorizing specific training examples
- Better at learning generalizable patterns
- Critical for residential network generalization (limited training data)

#### 4. Context Length: 1024

**Why large context is critical:**

**In-context network localization** (64 measurements @ 16 tokens avg):

```
Recent measurements:
  10ms to 1.1.1.1 (Cloudflare)
  12ms to 8.8.8.8 (Google)
  78ms to AWS us-west-2
  150ms to London
  →  Inference: "Measuring from US West Coast, datacenter connection"

  Prediction: "IP in Seattle → expect ~5ms"
```

**Residential vs datacenter detection:**
```
Datacenter:
  - Consistent RTTs (low variance)
  - Symmetric routes
  - Good peering (low RTTs to major CDNs)

Residential:
  - High variance (+/- 20%)
  - Asymmetric routes (via ISP backbone)
  - CGNAT artifacts
  - Time-of-day effects
```

**With 64 measurements, model can:**
- Identify geographic region (triangulation from multiple targets)
- Infer connection type (ISP, hosting provider, mobile)
- Detect routing anomalies (BGP changes, congestion)
- Adapt predictions to local network characteristics

**Rule of thumb:** More context = better localization = better generalization to unseen networks

### 3.4 Alternative Architecture (Deeper Search)

If RTT→IP search is primary use case (>80% of queries):

```yaml
base_emb_dim: 576              # Slightly smaller (9 × 64-dim heads)
base_num_decoder_layers: 24    # Even deeper reasoning
base_num_query_heads: 9
base_mlp_dim: 1792             # 3.1x ratio (more generalization)
```

**Parameters:** ~95M (fits budget)

**When to prefer:**
- IP sampling/search is dominant workload
- Willing to trade some IP structure learning for search depth

---

## 4. Training Strategy

### 4.1 Training Hyperparameters

```yaml
# Optimization
optimizer: adamw
learning_rate: 3.0e-4          # Standard for small-medium models
learning_rate_schedule: cosine
warmup_steps: 2000
max_steps: 200000              # ~5.4B tokens × 2-3 epochs

adam_b1: 0.9
adam_b2: 0.999
adam_eps: 1.0e-8
adam_eps_root: 0.0
weight_decay: 0.01

# Batching (Single GPU)
per_device_batch_size: 32      # Adjust based on A100 memory
global_batch_size: 32          # Single GPU (no data parallelism)
sequence_length: 1024          # Match max_target_length

# Effective tokens per step: 32 × 1024 = 32.7k tokens
# Total training tokens: 200k steps × 32.7k = 6.5B tokens (~1.2 epochs on 5.4B token dataset)

# Precision
param_dtype: bfloat16
activation_dtype: bfloat16
```

### 4.2 Data Loading with Timestamp Training Modes

**Framework:** Grain (Google's data loading library)

**Three Training Modes (Critical for generalization):**

1. **Mode 1: Full Timestamp (40% of batches)**
   - All measurements include timestamps
   - Measurements ordered temporally
   - Delta timestamps refer to previous measurement
   - Model learns: Temporal patterns, diurnal effects, routing stability over time

2. **Mode 2: No Timestamp (30% of batches)**
   - No measurements include timestamps
   - Measurements randomly shuffled (no temporal order)
   - Model learns: Pure network topology, geographic patterns (atemporal)
   - Critical for: Queries without timestamp context, generalization to unseen times

3. **Mode 3: Mixed Timestamp (30% of batches)**
   - Some measurements have timestamps, others don't (randomly assigned)
   - Timestamped measurements maintain temporal order
   - Non-timestamped measurements shuffled randomly
   - Then interleaved while preserving timestamped order
   - Delta timestamps skip over non-timestamped measurements
   - Model learns: Robust to missing data, use temporal info when available

**Rationale:**
- 40% full timestamp: Learn temporal patterns and enable efficient delta encoding
- 30% no timestamp: Force learning of atemporal network topology (prevents overfitting to "3pm traffic patterns")
- 30% mixed: Realistic scenario (missing/unreliable timestamps), teaches robustness

**Pipeline Implementation:**
```python
import grain
import random

class ContextWindowSampler(grain.transforms.Map):
    """Sample context windows with timestamp training modes."""

    def __init__(self, window_size=64):
        self.window_size = window_size

    def map(self, measurements):
        """
        Sample a context window and apply timestamp training mode.

        Returns: List of token IDs for the context window
        """
        # Sample measurements for this context window
        window = self._sample_measurements(measurements, self.window_size)

        # Decide training mode (40/30/30 split)
        mode = random.choices(
            ['full_timestamp', 'no_timestamp', 'mixed'],
            weights=[0.40, 0.30, 0.30]
        )[0]

        if mode == 'full_timestamp':
            # All have timestamps, keep temporal order
            window.sort(key=lambda m: m['event_time'])
            has_timestamp = [True] * len(window)

        elif mode == 'no_timestamp':
            # None have timestamps, random order
            random.shuffle(window)
            has_timestamp = [False] * len(window)

        else:  # mode == 'mixed'
            # Randomly assign timestamps (40-60% coverage)
            has_timestamp = [random.random() < 0.5 for _ in window]

            # Separate timestamped and non-timestamped
            timestamped = [m for m, has_ts in zip(window, has_timestamp) if has_ts]
            non_timestamped = [m for m, has_ts in zip(window, has_timestamp) if not has_ts]

            # Keep timestamped ordered, shuffle non-timestamped
            timestamped.sort(key=lambda m: m['event_time'])
            random.shuffle(non_timestamped)

            # Interleave randomly (preserving timestamped order)
            window = self._interleave_preserving_order(timestamped, non_timestamped)
            # Update has_timestamp to match new order
            has_timestamp = [m in timestamped for m in window]

        # Encode window with optional timestamps
        return self._encode_window(window, has_timestamp)

    def _encode_window(self, window, has_timestamp):
        """
        Encode measurements with delta timestamps.

        Delta timestamps skip over non-timestamped measurements:
        - M1(t=100) → M2(no ts) → M3(t=220)
        - M3's delta is 220-100=120s, not relative to M2
        """
        tokens = []
        prev_timestamped = None

        for meas, include_ts in zip(window, has_timestamp):
            # Encode timestamp block (if included)
            if include_ts:
                timestamp_block = encode_timestamp_delta(
                    meas['event_time'],
                    prev_timestamped  # Delta to previous TIMESTAMPED measurement
                )
                prev_timestamped = meas['event_time']
            else:
                timestamp_block = []  # Omit timestamp entirely

            # Encode measurement fields (shuffled, with optional timestamp)
            meas_tokens = encode_measurement(
                meas,
                timestamp_block=timestamp_block
            )
            tokens.extend(meas_tokens)

        return {"tokens": tokens, "length": len(tokens)}

    def _interleave_preserving_order(self, ordered_list, random_list):
        """Randomly interleave two lists, preserving order of first list."""
        result = []
        i, j = 0, 0

        while i < len(ordered_list) or j < len(random_list):
            # Randomly decide whether to take from ordered or random
            # (with bias to maintain roughly even distribution)
            if i >= len(ordered_list):
                result.append(random_list[j])
                j += 1
            elif j >= len(random_list):
                result.append(ordered_list[i])
                i += 1
            elif random.random() < 0.5:
                result.append(ordered_list[i])
                i += 1
            else:
                result.append(random_list[j])
                j += 1

        return result


# Data pipeline
train_ds = grain.experimental.ParquetIterDataset("data/sharded/train/*.parquet")
train_ds = train_ds.map(ContextWindowSampler(window_size=64))
train_ds = train_ds.batch(batch_size=32, drop_remainder=True)
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.repeat()
```

**Note:** Grain runs in separate worker processes for CPU/GPU overlap

### 4.3 Checkpointing

```yaml
enable_checkpointing: true
checkpoint_period: 5000        # Save every 5k steps
max_checkpoints_to_keep: 10    # Keep last 10 checkpoints
checkpoint_type: orbax         # Modern JAX checkpointing

# Checkpoint directory structure:
# outputs/
#   latency_network/
#     run_name/
#       checkpoints/
#         0005000.orbax-checkpoint/
#         0010000.orbax-checkpoint/
#         ...
```

### 4.4 Logging & Monitoring

```yaml
# TensorBoard
log_period: 100                # Log every 100 steps
eval_interval: 1000            # Eval every 1k steps
eval_steps: 100                # 100 eval batches per eval

# Metrics to track:
# - Training loss (perplexity)
# - Learning rate (cosine schedule)
# - Gradient norm (detect instability)
# - Throughput (tokens/sec, samples/sec)
# - Evaluation loss (validation set)
# - Field-specific metrics (RTT MAE, IP byte accuracy)
```

### 4.5 Positional Encoding Strategy

**Question:** Do we need positional embeddings when fields are shuffled within measurements?

**Answer:** Yes, positional encoding is still valuable, but for **between-measurement** position, not within-measurement.

**Three options to test:**

1. **RoPE (Rotary Position Embeddings)** - RECOMMENDED STARTING POINT
   ```yaml
   position_embedding: "rope"
   ```
   - **Pros:** Standard, well-tested, encodes relative position efficiently
   - **Cons:** Adds noise within measurements (shuffled fields don't have meaningful position)
   - **Verdict:** Noise is minor, causal structure + RoPE on measurement sequence position is helpful

2. **Learned Positional Embeddings**
   ```yaml
   position_embedding: "learned"
   ```
   - **Pros:** Model can learn to ignore position within measurements, use it between measurements
   - **Cons:** Extra parameters (1024 × 640 = 655k params)
   - **Verdict:** Most flexible, worth testing as ablation

3. **No Positional Encoding**
   ```yaml
   position_embedding: "none"
   ```
   - **Pros:** No noise from shuffled fields, delta timestamps provide temporal info
   - **Cons:** Model doesn't know position in context window (recency bias lost)
   - **Verdict:** Interesting experiment, might work given field shuffling

**Why RoPE is orthogonal to delta timestamps:**
- **Delta timestamps:** Encode real-world temporal distance ("60 seconds elapsed")
- **RoPE:** Encodes position in context window ("This is the 5th measurement in the prompt")
- Both are useful: Delta for temporal patterns, RoPE for in-context learning recency

**Recommendation:** Start with RoPE, test "learned" and "none" as ablations if time permits.

### 4.6 Hardware Requirements

**Target Platform: Single A100 (40GB or 80GB) on SLURM**

```yaml
# Hardware configuration
hardware: gpu
ici_data_parallelism: 1        # No multi-GPU parallelism
ici_tensor_parallelism: 1
ici_pipeline_parallelism: 1
```

**Requirements:**
```
GPU: 1× A100 (40GB minimum, 80GB recommended)
CPU: 16+ cores (for Grain data loading)
RAM: 64GB
Disk: 100GB SSD for checkpoints + sharded data
```

**Expected throughput:**
- Single A100 (40GB): ~40-50k tokens/sec
- Batch size 32 × sequence length 1024 = 32.7k tokens/batch
- ~1.2-1.5 batches/sec
- **Time to 200k steps: ~37 hours** (1.5 days)

**Memory estimate:**
```
Model parameters: ~95M × 2 bytes (bfloat16) = 190MB
Optimizer state (AdamW): ~95M × 8 bytes = 760MB
Gradients: ~95M × 2 bytes = 190MB
Activations (batch=32, seq=1024): ~2-3GB
Total: ~4-5GB (fits comfortably in 40GB A100)
```

---

## 5. Implementation Roadmap

### Phase 0: Tokenization Update ✅ CURRENT

**Tasks:**
- [ ] Update `tokenization.py` with optimized encoding:
  - [ ] Implement 5-bit exponent + 11-bit mantissa RTT encoding (microseconds, 2 bytes)
  - [ ] Implement delta timestamps (1/4 bytes)
  - [ ] Merge SrcIp/DstIp with IPv4/IPv6 tokens
  - [ ] Add `<Failed>` token for failed probes
  - [ ] Remove msm_id, size, error_count encodings
- [ ] Update `scripts/test_tokenization_standalone.py` for new scheme
- [ ] Create `scripts/verify_tokenization.py` to validate:
  - [ ] Token counts (23 for IPv4 first, 16 for subsequent)
  - [ ] RTT encoding/decoding accuracy (±0.1% error)
  - [ ] Delta timestamp coverage (95%+ use 1-byte)
- [ ] Test on real dataset (100 samples from Parquet)

**Validation criteria:**
- All token IDs in [0, 266]
- IPv4 first measurement: exactly 23 tokens (with timestamp), 14 tokens (without)
- IPv4 subsequent (delta): exactly 16 tokens (with timestamp), 14 tokens (without)
- RTT round-trip error < 0.049% for all observed values (5-bit exponent + 11-bit mantissa)
- Failed probes (RTT=-1.0) encoded as single `<Failed>` token

### Phase 0.5: Training Modes Implementation

**Tasks:**
- [ ] Implement `ContextWindowSampler` in Grain pipeline:
  - [ ] Mode 1 (40%): Full timestamp, temporal order
  - [ ] Mode 2 (30%): No timestamp, random shuffle
  - [ ] Mode 3 (30%): Mixed timestamp with interleaving logic
- [ ] Update `encode_measurement()` to accept optional timestamp block
- [ ] Implement `_interleave_preserving_order()` for Mode 3
- [ ] Test delta timestamp skipping (Mode 3):
  - [ ] Verify deltas refer to previous timestamped measurement
  - [ ] Test with various timestamp coverage ratios (30-70%)
- [ ] Validate training modes:
  - [ ] Sample 100 context windows, verify mode distribution (40/30/30)
  - [ ] Check Mode 1: all measurements have timestamps, sorted by time
  - [ ] Check Mode 2: no measurements have timestamps, random order
  - [ ] Check Mode 3: mixed measurements interleaved correctly

**Validation criteria:**
- Mode distribution matches 40/30/30 split over 1000 samples
- Mode 1 sequences are strictly temporally ordered
- Mode 2 sequences have no timestamps
- Mode 3 delta timestamps skip non-timestamped measurements correctly

### Phase 1: MaxText Configuration

**Tasks:**
- [ ] Update `src/MaxText/configs/latency_network.yml`:
  - [ ] Set vocab_size: 267
  - [ ] Set architecture params (640 emb, 20 layers, 2048 MLP)
  - [ ] Configure max_target_length: 1024
  - [ ] Update checkpoint paths (absolute)
- [ ] Create smoke test script `scripts/run_maxtext_smoke.sh`:
  - [ ] CPU test with synthetic data
  - [ ] 10 steps, batch_size=1
  - [ ] Verify shapes, no NaN losses
- [ ] Test with optimized tokenization:
  - [ ] Update Grain pipeline with delta timestamps
  - [ ] Verify sequence packing works
  - [ ] Check context window utilization

**Validation criteria:**
- Model initializes without errors
- Parameter count: ~85-95M (target 100M)
- Forward pass produces finite logits
- Loss decreases over 10 steps
- Sequences utilize full 1024 context

### Phase 2: Dataset Preparation

**Tasks:**
- [ ] Run `scripts/shard_parquet.py`:
  - [ ] Create 200 train shards
  - [ ] Create 25 val shards
  - [ ] Create 25 test shards
  - [ ] Verify temporal stratification
- [ ] Update data paths in config:
  - [ ] Point to sharded directories
  - [ ] Configure Grain worker count
- [ ] Test data loading:
  - [ ] Verify shuffle quality
  - [ ] Check I/O throughput
  - [ ] Profile bottlenecks

**Validation criteria:**
- All shards have similar size (~400k rows/shard for train)
- Temporal distribution balanced across splits
- Data loader achieves >50k tokens/sec (CPU)
- No duplicate measurements across train/val/test

### Phase 3: Local GPU Training

**Tasks:**
- [ ] Run 1-GPU training (1k steps):
  - [ ] Monitor loss convergence
  - [ ] Check for NaN/Inf issues
  - [ ] Verify checkpoint saving works
- [ ] Profile performance:
  - [ ] Tokens/second throughput
  - [ ] GPU memory utilization
  - [ ] Identify bottlenecks
- [ ] Tune hyperparameters:
  - [ ] Learning rate (try 1e-4, 3e-4, 1e-3)
  - [ ] Batch size (maximize GPU utilization)
  - [ ] Gradient clipping (if instability)

**Validation criteria:**
- Loss decreases consistently
- No NaN/Inf in gradients
- GPU utilization >80%
- Throughput >30k tokens/sec (single A100)

### Phase 4: SLURM Single-GPU Training

**Tasks:**
- [ ] Update `scripts/slurm_train_maxtext.sh`:
  - [ ] Configure single A100 job (40GB or 80GB)
  - [ ] Set data paths for cluster storage
  - [ ] Add TensorBoard logging
  - [ ] Configure environment variables (DECOUPLE_GCLOUD=TRUE)
- [ ] Submit training job:
  - [ ] Run for 200k steps (~37 hours on A100)
  - [ ] Monitor TensorBoard remotely
  - [ ] Save checkpoints every 5k steps
- [ ] Evaluate on test set (periodically during training):
  - [ ] Compute perplexity
  - [ ] Measure RTT prediction MAE
  - [ ] Check IP byte accuracy

**Validation criteria:**
- Training completes without OOM (should use ~4-5GB of 40GB)
- Test loss decreases or plateaus
- RTT prediction MAE < 10ms for typical measurements
- Checkpoints save successfully (absolute paths)

### Phase 5: Model Evaluation

**Tasks:**
- [ ] Create evaluation scripts:
  - [ ] `scripts/eval_forward_modeling.py`: IP → RTT prediction
  - [ ] `scripts/eval_inverse_search.py`: RTT → IP sampling
  - [ ] `scripts/eval_ip_completion.py`: Partial IP → Full IP
- [ ] Quantitative metrics:
  - [ ] RTT prediction: MAE, MSE, calibration
  - [ ] IP prediction: Top-K accuracy (K=1,5,10)
  - [ ] Perplexity on held-out test set
- [ ] Qualitative analysis:
  - [ ] Sample generations for plausibility
  - [ ] Check geographic consistency
  - [ ] Verify failure mode handling

**Validation criteria:**
- RTT MAE < 20ms on test set
- Top-5 IP byte accuracy > 50%
- Generations are plausible (no random IPs)

### Phase 6: Inference & Deployment

**Tasks:**
- [ ] Export checkpoint for inference:
  - [ ] Convert to SavedModel or ONNX
  - [ ] Optimize for inference (quantization?)
- [ ] Create inference API:
  - [ ] `predict_rtt(src_ip, dst_ip, context)`
  - [ ] `sample_ips(target_rtt, src_ip, context)`
  - [ ] `complete_ip(partial_ip, context)`
- [ ] Build demo application:
  - [ ] Web interface for queries
  - [ ] Visualization of predictions
  - [ ] Interactive IP exploration

---

## 6. Evaluation & Metrics

### 6.1 Training Metrics

**Primary:**
- **Perplexity:** exp(NLL) on validation set
- **Bits per byte:** NLL / log(2) / avg_bytes_per_measurement

**Secondary:**
- **Gradient norm:** Detect instability
- **Learning rate:** Track cosine schedule
- **Throughput:** Tokens/sec, samples/sec

### 6.2 Task-Specific Metrics

#### Forward Modeling (IP → RTT)

**Input:** Source IP, Destination IP, timestamp, context
**Output:** RTT distribution

**Metrics:**
- **MAE (Mean Absolute Error):** Average |predicted_RTT - true_RTT|
- **MSE (Mean Squared Error):** Average (predicted_RTT - true_RTT)²
- **Calibration:** P(RTT < X) vs empirical frequency
- **Breakdown by:**
  - IP version (IPv4 vs IPv6)
  - Distance (same /16, same /8, different continent)
  - Connection type (datacenter, residential)

**Target:** MAE < 20ms for typical measurements (50-500ms RTT)

#### Inverse Search (RTT → IP)

**Input:** Target RTT, optional constraints (source IP, geographic hints), context
**Output:** Sampled candidate IPs

**Metrics:**
- **Diversity:** Unique IPs in top-K samples
- **Plausibility:** % samples with correct RTT range (±20%)
- **Coverage:** Geographic distribution of samples
- **Consistency:** Same query → similar samples (low variance)

**Qualitative:**
- Are sampled IPs routable?
- Do they belong to plausible networks (datacenters, universities, ISPs)?
- Do geographic patterns make sense?

#### IP Completion (Partial IP → Full IP)

**Input:** Partial IP (e.g., "192.0.2.???"), context
**Output:** Completed IP

**Metrics:**
- **Byte accuracy:** % correct bytes in generated suffix
- **Top-K accuracy:** True IP in top-K completions (K=1,5,10)
- **Entropy:** Distribution over completions (high = uncertain, low = confident)

**Breakdown by:**
- Prefix length (/8, /16, /24, /28)
- IP allocation type (datacenter, residential, mobile)

### 6.3 Generalization Tests

**Critical:** Test on unseen network types

**Scenarios:**
1. **Held-out geographic regions:**
   - Train without South America data
   - Test on South America measurements
   - Expected: Model infers routing via learned topology rules

2. **Held-out ISPs:**
   - Train without Comcast/AT&T data
   - Test on Comcast/AT&T measurements
   - Expected: Model generalizes from other residential ISPs

3. **Temporal generalization:**
   - Train on 2025-06 data
   - Test on 2025-07 data
   - Expected: Model handles routing changes, seasonal patterns

4. **Novel IP pairs:**
   - Test on IP pairs never seen in training
   - Expected: Model uses learned structure (ASN topology, geographic priors)

---

## 7. Open Questions

### 7.1 Tokenization

- [ ] **Log-scale RTT encoding precision:** Is ±0.1% sufficient, or do we need tighter precision for short RTTs (<10ms)?
- [ ] **Timestamp delta overflow:** How often do consecutive measurements have >4 byte delta (>49 days)? May need statistics.
- [ ] **Failed probe handling:** Should `<Failed>` include reason code (timeout, ICMP unreachable, etc.) if available in dataset?

### 7.2 Model Architecture

- [ ] **MLP ratio tradeoff:** Would 2.5x ratio (1600 MLP) improve generalization further? Parameter budget allows it.
- [ ] **Positional encodings:** Do we need them for shuffled fields? Or rely on learned position-invariance?
- [ ] **Attention variants:** Would local attention help (fields within measurement)? Or stick to global attention?

### 7.3 Training

- [ ] **Sequence packing:** Pack measurements from same msm_id stream (temporal locality) or random measurements?
- [ ] **Curriculum learning:** Start with IPv4 only, add IPv6 later? Or mixed from start?
- [ ] **Data augmentation:** Shuffle field order multiple ways per measurement (increase effective dataset size)?

### 7.4 Evaluation

- [ ] **Residential network detection:** How to quantitatively measure if model learned residential vs datacenter patterns?
- [ ] **Geographic inference:** Can we extract learned "geographic embeddings" from IP prefixes?
- [ ] **Routing knowledge:** How to test if model learned AS-level topology vs pure memorization?

### 7.5 Applications

- [ ] **Path selection:** Given multiple candidate IPs, rank by expected RTT (for CDN selection, mirror selection)
- [ ] **Anomaly detection:** Flag measurements with unexpected RTT given IP pair and context
- [ ] **Network planning:** "What new peering would reduce latency by >10ms?" (counterfactual queries)
- [ ] **Compression:** Can trained model compress network measurements better than generic compressors?

---

## Appendix A: Token ID Reference

```python
# Role tokens (0-10)
MEASUREMENT_START  = 0
SRC_IPV4           = 1
SRC_IPV6           = 2
DST_IPV4           = 3
DST_IPV6           = 4
TIMESTAMP_ABS      = 5
TIMESTAMP_DELTA1   = 6
TIMESTAMP_DELTA4   = 7
RTT_START          = 8
THROUGHPUT_START   = 9
FAILED             = 10

# Byte tokens (11-266)
BYTE_TOKEN_OFFSET  = 11

def byte_to_token(byte_val: int) -> int:
    return BYTE_TOKEN_OFFSET + byte_val  # 11-266

def token_to_byte(token_id: int) -> int:
    return token_id - BYTE_TOKEN_OFFSET  # 0-255

VOCAB_SIZE = 267
```

## Appendix B: Dataset Statistics

**100M measurements breakdown:**
```
Total rows:           100,005,691
IPv4:                 57,742,149 (57.7%)
IPv6:                 42,263,542 (42.3%)

Failed probes:        ~3,500,000 (3.5%, RTT = -1.0)
Empty src_addr:       81,278 (0.08%)
Empty dst_addr:       120,461 (0.12%)

RTT distribution (successful probes):
  p1:    3.2 ms
  p10:   12.5 ms
  p25:   25.8 ms
  p50:   50.3 ms
  p75:   120.7 ms
  p90:   239.4 ms
  p99:   343.1 ms
  p99.9: 1,204 ms
  max:   302,281 ms (outlier, likely measurement error)

Timestamp gaps (consecutive same msm_id):
  p50:   60 seconds
  p90:   180 seconds
  p99:   600 seconds
  max:   2,419,200 seconds (28 days, different msm_id)

Total unique IPs:
  Source:       ~850,000 unique
  Destination:  ~1,200,000 unique
  (Many-to-many: RIPE Atlas probes → diverse targets)
```

## Appendix C: Comparison to PLAN.md

**Key changes from original plan:**

| Aspect | PLAN.md | PLAN_2.md | Change |
|--------|---------|-----------|--------|
| **Vocab size** | 266 | 267 | +1 (added `<Failed>`) |
| **Tokens/measurement (IPv4)** | 42-45 | 14-23 | 49-69% reduction |
| **Context capacity** | ~22 | ~64 (IPv4 no-ts) | 3x improvement |
| **Model size** | 100M | ~95M | Optimized for generalization |
| **MLP ratio** | 4x | 3.2x | Smaller for generalization |
| **Depth** | 12 | 20 | +67% for search reasoning |
| **Embedding** | 768 | 640 | Optimized for small vocab |
| **Max length** | 512-2048 (TBD) | 1024 | Fixed for in-context learning |
| **RTT encoding** | 8 bytes float64 | 2 bytes (5exp+11mant) | 75% reduction, 1μs-51 days range |
| **Timestamp** | 8 bytes absolute | 1-9 bytes (deltas) | 78-89% reduction |
| **Removed fields** | - | msm_id, size, errors | Cleaner schema |
| **Optional timestamps** | No | Yes (40/30/30 modes) | Enables atemporal learning |
| **Dataset split** | 80/10/10 | 90/10 | No validation set |
| **GPUs** | 1-4 | 1 | Single A100 on SLURM |
| **Position encoding** | Not specified | RoPE (recommended) | Orthogonal to delta timestamps |

**Philosophy shift:**
- **Old:** General-purpose measurement modeling
- **New:** Generalization-focused design for network path selection and IP inference via in-context learning
- **Key insight:** Optional timestamps enable model to learn both temporal patterns and atemporal network topology

**Novel contributions:**
1. **5-bit exponent RTT encoding:** Future-proof for interplanetary networks (Mars!)
2. **3 training modes:** 40% full timestamp, 30% no timestamp, 30% mixed
3. **Delta timestamp skipping:** Deltas refer to previous timestamped measurement in mixed mode
4. **Timestamp-agnostic predictions:** Model learns topology independent of temporal context

---

**Document status:** Living document, updated as implementation progresses
**Last updated:** 2025-12-13 (Updated: RTT encoding, training modes, single-GPU, RoPE discussion)
**Next review:** After Phase 0.5 completion (training modes + tokenization update)
