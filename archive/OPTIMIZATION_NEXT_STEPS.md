# Further Optimization: 30% → 40-50% MFU

**Current Status**: 30% MFU (93.6 TFLOPS on A100-80GB)
**Target**: 40-50% MFU (125-156 TFLOPS)
**Improvement Needed**: 1.33-1.67x speedup

## Analysis

Your 95M parameter model on A100 is relatively small, which means:
- ✅ You're **compute-bound** (not memory bandwidth limited)
- ⚠️ Small models may not fully saturate large GPUs
- ⚠️ Overhead from kernel launches, compilation, and data pipeline becomes significant

## Recommended Optimizations (Prioritized)

### 1. Increase Batch Size (HIGHEST IMPACT - Expected: +20-30% MFU)

**Current**: `per_device_batch_size: 128`
**Your note**: "70% GPU memory was idle"

**Action**: Increase batch size to saturate GPU memory

```yaml
# Test incrementally:
per_device_batch_size: 192  # +50% batch size
# or
per_device_batch_size: 256  # +100% batch size
# or
per_device_batch_size: 384  # +200% batch size (test until OOM)
```

**How to find optimal batch size**:
```bash
# Monitor GPU memory during training
nvidia-smi dmon -s mu -c 100

# Look for:
# - Memory utilization: 85-95% is ideal
# - GPU utilization: should stay high (>90%)
```

**Learning rate adjustment** (when increasing batch size):
```yaml
# Use sqrt scaling rule
# If batch_size increases 2x: lr increases sqrt(2) = 1.414x
# Examples:
per_device_batch_size: 192 → learning_rate: 1.225e-4  # sqrt(192/128) = 1.225
per_device_batch_size: 256 → learning_rate: 1.414e-4  # sqrt(256/128) = 1.414
per_device_batch_size: 384 → learning_rate: 1.732e-4  # sqrt(384/128) = 1.732
```

**Expected impact**:
- 2x batch size → 1.4-1.6x throughput (not linear due to overhead)
- Memory utilization: 70% → 85-95%
- **MFU improvement: 30% → 38-42%**

---

### 2. Enable XLA GPU Optimizations (Expected: +5-10% MFU)

Add these JAX/XLA flags for A100-specific optimizations:

**Create new file**: `scripts/train_optimized.sh`
```bash
#!/bin/bash

# A100-optimized XLA flags
export XLA_FLAGS="
  --xla_gpu_enable_latency_hiding_scheduler=true
  --xla_gpu_enable_triton_softmax_fusion=true
  --xla_gpu_triton_gemm_any=true
  --xla_gpu_enable_async_collectives=true
  --xla_gpu_enable_highest_priority_async_stream=true
"

# Run training
python scripts/train.py "$@"
```

**What these do**:
- `latency_hiding_scheduler`: Overlaps computation with data loading
- `triton_softmax_fusion`: Fuses softmax operations into single kernels
- `triton_gemm_any`: Uses optimized Triton kernels for matrix multiplies
- `async_collectives`: Enables asynchronous operations (future-proofing for multi-GPU)

**Expected impact**: **MFU improvement: +3-8%**

---

### 3. Optimize Data Pipeline (Expected: +2-5% MFU)

**Current**:
```yaml
grain_worker_count: 16
grain_per_worker_buffer_size: 8
```

**Optimizations**:

#### A. Enable grain debugging to identify bottlenecks
```yaml
grain_debug_mode: true  # Log execution metrics every 60s
grain_visualization_dir: "/tmp/grain_viz"  # Generate pipeline graph
```

#### B. Increase prefetch buffer (if data loading is slow)
```yaml
grain_per_worker_buffer_size: 16  # up from 8
grain_ram_budget_mb: 32768  # up from 16384 (if RAM available)
```

#### C. Profile to check if data is bottleneck
```bash
# During training, check if GPU is waiting:
nvidia-smi dmon -s u -c 100
# If GPU util drops below 90%, data pipeline may be slow
```

**Expected impact**: **MFU improvement: +2-5%** (if data-bound)

---

### 4. Compiler Optimizations (Expected: +3-5% MFU)

**Enable PGLE (Profile Guided Latency Estimator)**:
```bash
# In your training script
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_pgle_profile_file_or_directory_path=/tmp/pgle_profiles"
```

This profiles actual runtime and optimizes scheduling on subsequent runs.

**Expected impact**: **MFU improvement: +3-5%** (after warmup)

---

### 5. Mixed Precision Optimization (Already done, but verify)

**Current**: `dtype: 'bfloat16'` ✅

Verify no accidental upcasting:
```bash
# Check for FP32 operations in logs
grep -i "float32\|fp32" <training_logs>
```

---

### 6. Remove Dropout During Final Runs (Expected: +1-2% MFU)

**Current**: `dropout_rate: 0.1`

Dropout adds random operations. For production runs:
```yaml
dropout_rate: 0.0  # Remove for slight speedup (only if not affecting convergence)
```

**Expected impact**: **MFU improvement: +1-2%**

---

### 7. Gradient Accumulation (For even larger effective batches)

If you hit OOM with large batch sizes, simulate larger batches:

```yaml
gradient_accumulation_steps: 2  # Simulates 2x batch size
# Effective batch = 256 * 2 = 512
# But memory usage stays at 256
```

---

## Quick Wins Summary

| Optimization | Difficulty | Expected MFU Gain | New MFU |
|--------------|-----------|-------------------|---------|
| **Increase batch to 256** | Easy | +8-12% | **38-42%** |
| **Add XLA flags** | Easy | +3-8% | **41-50%** |
| **Optimize data pipeline** | Medium | +2-5% | **43-55%** |
| **Enable PGLE** | Easy | +3-5% | **46-60%** |
| **Remove dropout** | Easy | +1-2% | **47-62%** |

**Best case scenario**: 30% → **50-60% MFU** 🎯

---

## Implementation Plan

### Phase 1: Immediate (5 minutes)
1. Increase `per_device_batch_size: 256`
2. Adjust `learning_rate: 1.414e-4` (sqrt scaling)
3. Test training for 10 steps
4. **Expected: 30% → 38-42% MFU**

### Phase 2: XLA Optimization (10 minutes)
1. Create `scripts/train_optimized.sh` with XLA flags
2. Run training
3. **Expected: 38-42% → 41-50% MFU**

### Phase 3: Data Pipeline (if needed, 20 minutes)
1. Enable `grain_debug_mode: true`
2. Check if data loading is bottleneck
3. Tune buffer sizes if needed
4. **Expected: +2-5% MFU**

### Phase 4: Advanced (optional, 30 minutes)
1. Enable PGLE for automatic optimization
2. Profile with `nvidia-smi` and `nsys`
3. **Expected: +3-5% MFU**

---

## Monitoring

Track these metrics after each change:

```bash
# From training logs, watch for:
# - TFLOP/s/device: should increase from ~94 to 125-156
# - Tokens/s/device: should increase proportionally
# - total_weights: should stay ~120k-125k (packing working)
# - Step time: should decrease

# GPU monitoring:
nvidia-smi dmon -s umt -c 100
# Look for:
# - GPU util (%): should be >90%
# - Memory (%): should be 85-95%
# - Temp: should stay <85°C
```

---

## Expected Final Performance

**Before all optimizations**:
```
MFU: 30%
TFLOPS: 93.6
Tokens/s: ~180k
Step time: ~0.4s
```

**After batch size + XLA optimizations**:
```
MFU: 40-50%
TFLOPS: 125-156
Tokens/s: 250k-320k
Step time: ~0.25-0.30s
```

**Training time for 10k steps**:
- Before: 1.1 hours
- After: 0.7-0.8 hours (30-40% faster)

---

## References

- [JAX GPU Performance Tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
- [XLA GPU Architecture](https://openxla.org/xla/gpu_architecture)
- [MaxText Optimization Guide](https://maxtext.readthedocs.io/en/latest/guides/optimization/benchmark_and_performance.html)
- [NVIDIA MaxText Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/maxtext-llama2-70b-dgxc-benchmarking-c)
