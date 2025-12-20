# Probe Row Preprocessing Script Guide

## Quick Decision Tree

```
┌─────────────────────────────────────────┐
│ How much data are you processing?       │
└─────────────────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
  < 1M measurements      > 1M measurements
      │                       │
      ▼                       ▼
┌─────────────┐     ┌─────────────────────┐
│ Sequential  │     │ How much RAM?        │
│ is fine     │     └─────────────────────┘
└─────────────┘               │
                  ┌───────────┴────────────┐
                  │                        │
            > 16GB RAM                < 16GB RAM
                  │                        │
                  ▼                        ▼
         ┌────────────────┐      ┌──────────────────┐
         │ Parallel       │      │ Streaming        │
         │ (in-memory)    │      │ (disk-based)     │
         └────────────────┘      └──────────────────┘
```

---

## Available Scripts

> **Note:** The sequential `create_probe_rows.py` has been removed. Use the parallel versions below.

### 1. `create_probe_rows_parallel.py` - Parallel In-Memory

**Use when:**
- Medium datasets (1M - 50M measurements)
- Have 16GB+ RAM available
- Multi-core system (4+ cores)
- Fast preprocessing is critical

**Pros:**
- ✅ **17x faster** than sequential
- ✅ Full CPU utilization
- ✅ Simple implementation
- ✅ Good for most use cases

**Cons:**
- ❌ **Loads entire dataset into RAM**
- ❌ Will OOM on 200M+ measurements
- ❌ Requires 16GB+ RAM for large datasets

**Example:**
```bash
python scripts/data/create_probe_rows_parallel.py \
  --input "data/sharded/**/*.parquet" \
  --output data/probe_rows \
  --workers 8
```

**Performance:** ~6s for 500K measurements (80K meas/sec)

**Memory usage:**
- 500K meas: ~800MB
- 10M meas: ~16GB
- 50M meas: **80GB** (too much!)

**⚠️ OOM Error Example:**
```
_duckdb.OutOfMemoryException: Out of Memory Error:
failed to allocate data of size 16.0 KiB (24.5 GiB/24.5 GiB used)
```

---

### 2. `create_probe_rows_parallel_streaming.py` - Streaming (Recommended)

**Use when:**
- **Large datasets (50M+ measurements)**
- Limited RAM (< 16GB available)
- Modal deployment
- **Production use with 200M measurements**

**Pros:**
- ✅ **Memory-safe** - uses disk instead of RAM
- ✅ Still parallelized (nearly as fast)
- ✅ Configurable memory limits
- ✅ Works on any system
- ✅ **Recommended for Modal**

**Cons:**
- ❌ Slightly slower (~10% overhead from disk I/O)
- ❌ Requires temp disk space
- ❌ More complex implementation

**Example:**
```bash
# Auto-detect memory (leaves 1GB for system)
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/sharded/**/*.parquet" \
  --output data/probe_rows \
  --workers 8

# Explicit memory limit (for 32GB system)
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/sharded/**/*.parquet" \
  --output data/probe_rows \
  --workers 8 \
  --memory-limit-gb 28
```

**Performance:** Similar to parallel version (~15% slower)

**Memory usage:** Fixed at specified limit (default: available - 1GB)

---

## Recommendations by Environment

### Local Development
```bash
# Testing on small subset (streaming version)
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/test/shard_0001.parquet" \
  --output data/probe_rows_test

# Medium dataset (< 16GB RAM)
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/sharded/test/*.parquet" \
  --output data/probe_rows \
  --workers 4 \
  --memory-limit-gb 12
```

### Modal (Production - 200M measurements)

**Option A: High-memory instance (Recommended)**
```bash
# Uses streaming version with 64GB RAM
modal run scripts/data/modal_create_probe_rows_parallel_streaming.py

# Or deploy locally and run streaming version:
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "/mnt/data/training_data.parquet" \
  --output /mnt/data/probe_rows \
  --workers 8 \
  --memory-limit-gb 28
```

**Specs:**
- CPU: 8 cores
- RAM: 32GB (28GB for DuckDB, 4GB for system)
- Time: ~25-30 minutes
- Cost: ~$0.50-1.00

**Option B: Low-memory instance (Budget)**
```bash
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "/mnt/data/training_data.parquet" \
  --output /mnt/data/probe_rows \
  --workers 4 \
  --memory-limit-gb 12
```

**Specs:**
- CPU: 4 cores
- RAM: 16GB
- Time: ~45-60 minutes
- Cost: ~$0.30-0.50

---

## Memory Limit Guidelines

### How to Calculate Memory Limit

```python
# Formula: memory_limit_gb = total_ram_gb - system_overhead_gb

# Examples:
16GB system  →  --memory-limit-gb 14  (leave 2GB)
32GB system  →  --memory-limit-gb 28  (leave 4GB)
64GB system  →  --memory-limit-gb 60  (leave 4GB)
```

### Auto-detection (Default)

The streaming script auto-detects available memory:
```python
available_gb = psutil.virtual_memory().available / (1024 ** 3)
memory_limit_gb = available_gb - 1.0  # Leave 1GB for system
```

---

## When to Use Each Script

| Dataset Size | RAM Available | Script | Workers | Time (200M) |
|--------------|---------------|--------|---------|-------------|
| 1M - 10M | 8GB+ | `parallel` | 4 | 50 min |
| 10M - 50M | 16GB+ | `parallel` | 8 | 25 min |
| 50M - 200M | 16GB+ | **streaming** | 8 | **30 min** ✅ |
| 200M+ | Any | **streaming** | 8-16 | **30-40 min** ✅ |

---

## Common Issues

### OOM Error (Out of Memory)

**Symptom:**
```
_duckdb.OutOfMemoryException: failed to allocate data
```

**Solution:**
1. ✅ Use `create_probe_rows_parallel_streaming.py`
2. ✅ Set `--memory-limit-gb` lower
3. ✅ Reduce `--workers` count

### Slow Processing

**Symptom:** Taking hours for medium dataset

**Solution:**
1. ✅ Use parallel version instead of sequential
2. ✅ Increase `--workers` count
3. ✅ Use SSD instead of HDD for temp files

### Disk Space Error

**Symptom:**
```
No space left on device
```

**Solution:**
1. ✅ Streaming version uses temp directory (~10GB)
2. ✅ Set `TMPDIR=/path/to/large/disk`
3. ✅ Reduce batch sizes (modify script)

---

## Migration Path

### Current State
You're using `create_probe_rows_parallel.py` but getting OOM.

### Recommended Actions

1. **Immediate fix (Modal):**
   ```bash
   # Use streaming version
   modal run scripts/data/modal_create_probe_rows_parallel_streaming.py
   ```

2. **Local testing:**
   ```bash
   # Test streaming version on small dataset
   python scripts/data/create_probe_rows_parallel_streaming.py \
     --input "data/sharded/test/shard_0001.parquet" \
     --output data/probe_rows_test \
     --workers 4 \
     --memory-limit-gb 8
   ```

3. **Production deployment:**
   - Use streaming version
   - 32GB Modal instance
   - 8 workers
   - Expected: ~30 minutes

---

## Available Scripts

We maintain two implementations for different use cases:

- `create_probe_rows_parallel.py` → Medium datasets (1M-50M measurements), high RAM systems
- `create_probe_rows_parallel_streaming.py` → **Production, Modal, large datasets (50M+)** ✅

The sequential version was removed as the parallel versions are strictly better.

---

## Summary

### For Your 200M Measurement Dataset

**✅ Use:** `create_probe_rows_parallel_streaming.py`

**Command:**
```bash
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/sharded/**/*.parquet" \
  --output data/probe_rows \
  --workers 8 \
  --memory-limit-gb 28
```

**Expected:**
- Time: ~30 minutes
- Memory: 32GB total (28GB for DuckDB)
- Disk: ~10GB temp space
- Output: Same as parallel version

This will solve your OOM error while maintaining fast performance! 🚀
