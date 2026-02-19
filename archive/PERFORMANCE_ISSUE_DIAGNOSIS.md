# Performance Issue Diagnosis - 2025-12-20

## Problem Statement

Expected performance after optimizations: **120-150 TFLOPS (38-48% MFU)**
Actual performance: **86-90 TFLOPS (27.6-28.8% MFU)**

**This is WORSE than before the "optimizations"** (was 94 TFLOPS / 30% MFU)

## Root Cause: CLI Override

### The Issue

**Modal wrapper is overriding the config file batch size!**

```python
# modal_wrapper.py line 145:
def run(
    batch_size: int = 128,  # ← Default is 128!
    ...
):
    cmd = [
        ...
        "--batch-size",
        str(batch_size),  # ← Passes 128 to train.py
    ]
```

```python
# train.py line 279:
config_overrides = {
    "per_device_batch_size": args.batch_size,  # ← CLI overrides config!
}
```

**Result**: Even though config says `per_device_batch_size: 256`, Modal passes `--batch-size 128` which overrides it!

### What Each Change Was Supposed to Do

#### 1. Batch Size Increase (128 → 256)
**Expected**: 2x batch size saturates GPU memory, reduces per-token overhead
- **Arithmetic intensity**: More compute per memory transfer
- **GPU utilization**: Fill memory from 70% → 85-95%
- **Expected gain**: +8-12% MFU (most important change)

**Actual**: **Batch size stayed at 128 due to CLI override - NO EFFECT**

#### 2. Learning Rate Scaling (1.0e-4 → 1.414e-4)
**Expected**: Maintain training dynamics with larger batch
- **Formula**: `new_lr = old_lr * sqrt(new_batch / old_batch)`
- **Calculation**: `1.0e-4 * sqrt(256/128) = 1.414e-4`

**Actual**: **Applied but meaningless since batch size didn't change**
**Side effect**: May be causing training instability or slower convergence

#### 3. XLA Flags
**Expected**: Kernel fusion, latency hiding, optimized scheduling
- `--xla_gpu_enable_latency_hiding_scheduler`: Overlap compute with I/O
- `--xla_gpu_triton_gemm_any`: Optimized matrix multiply kernels
- `--xla_gpu_enable_highest_priority_async_stream`: Priority scheduling
- `--xla_gpu_pgle_profile_file_or_directory_path`: Profile-guided optimization

**Actual**: **May be causing overhead on first run (profiling)**
- PGLE needs warmup runs to be effective
- Some flags may not apply to small models
- Expected gain: +3-8% MFU (after warmup)

#### 4. Data Pipeline Tuning
**Expected**: Reduce GPU stalls from data loading
- Larger buffers: 8 → 16 (2x prefetch)
- More RAM: 16GB → 32GB budget

**Actual**: **Unlikely to have significant impact if GPU is underutilized**
- Data pipeline optimization helps when GPU is compute-bound
- If GPU is idle due to small batch, data isn't the bottleneck

## Why Performance Got WORSE (94 → 88 TFLOPS)

Possible explanations:

1. **XLA compilation overhead**: First runs with new flags are slower while profiling
2. **Learning rate mismatch**: Using 1.414e-4 LR with 128 batch may cause issues
3. **Measurement variance**: Normal run-to-run variation (±5-10%)
4. **PGLE profiling**: Creating profiles adds overhead initially

## Why Original Optimizations WOULD Have Worked

If batch size had actually increased to 256:

```python
# Small model arithmetic intensity calculation
model_params = 95M
bytes_per_param = 2 (bfloat16)
model_size = 0.18 GB

# Per-step calculation
batch_128_tokens = 128 * 1024 = 131k tokens
batch_256_tokens = 256 * 1024 = 262k tokens

# FLOPs calculation (6*params*tokens per forward + 2x for backward)
flops_128 = 6 * 95M * 131k * 3 = 224 TFLOPs/step
flops_256 = 6 * 95M * 262k * 3 = 448 TFLOPs/step

# Memory transfers (read all params + gradients)
memory_128 = 0.18 GB * 2 = 0.36 GB
memory_256 = 0.18 GB * 2 = 0.36 GB (same!)

# Arithmetic intensity (FLOPs per byte)
intensity_128 = 224 TFLOPs / 0.36 GB = 622 FLOPs/byte
intensity_256 = 448 TFLOPs / 0.36 GB = 1244 FLOPs/byte (2x better!)

# A100 memory bandwidth: 2039 GB/s
# Peak achievable with memory bandwidth:
peak_128 = 2039 GB/s * 622 FLOPs/byte = 1.27 PetaFLOPs (way above 312 TFLOPS)
peak_256 = 2039 GB/s * 1244 FLOPs/byte = 2.54 PetaFLOPs (way above 312 TFLOPS)
```

**Conclusion**: Model IS compute-bound (good!), not memory-bound. Larger batch WOULD help.

### Why Larger Batch Helps

1. **Amortizes overhead**: Kernel launch costs spread over more work
2. **Better GPU utilization**: More parallel work fills execution units
3. **Fewer kernel launches**: 256 tokens/batch needs fewer launches than 2x 128 batches
4. **Pipeline efficiency**: Longer-running kernels hide latency better

## The Fix

Change Modal wrapper default batch size:

```python
# modal_wrapper.py
def run(
    batch_size: int = 256,  # ← Change from 128 to 256
    ...
):
```

OR pass it explicitly when calling:

```bash
modal run scripts/train/modal_wrapper.py::run --batch-size 256
```

## Expected Performance After Fix

### Current (broken optimization)
```
Batch size: 128 (due to override)
Learning rate: 1.414e-4 (wrong for batch 128)
TFLOPS: 86-90 (28% MFU)
```

### After fixing batch size
```
Batch size: 256
Learning rate: 1.414e-4 (correct for batch 256)
TFLOPS: 120-150 (38-48% MFU)
Step time: ~0.27-0.32s (down from ~0.4s)
Tokens/s: 250k-320k (up from 180k)
```

## Why Small Models Have Low MFU

From MaxText docs and general GPU optimization principles:

1. **Kernel launch overhead**: Small models have short-running kernels
   - A100 has ~5-10μs kernel launch latency
   - If kernel runs for 20μs, 30-50% time is overhead!
   - Larger batches → longer kernels → less overhead

2. **Insufficient parallelism**: 95M params may not saturate 6912 CUDA cores
   - A100 has 108 SMs × 64 CUDA cores = 6912 cores
   - Need enough work to keep all cores busy
   - Small models may leave cores idle

3. **Memory bandwidth**: Small models are memory-bound at small batches
   - Must load all 95M params (0.18 GB) every step
   - At batch 128, FLOPs/byte ratio is low
   - At batch 256, ratio doubles → compute-bound

4. **Attention mechanism**: Flash attention has overhead for small sequences
   - Block size requirements (128 tokens)
   - More efficient at large batch sizes

## Summary

**What went wrong**: Modal CLI override prevented batch size increase
**What should have happened**: 2x batch → 1.4-1.6x throughput
**How to fix**: Change modal_wrapper default OR pass --batch-size 256
**Expected after fix**: 38-48% MFU (120-150 TFLOPS)
