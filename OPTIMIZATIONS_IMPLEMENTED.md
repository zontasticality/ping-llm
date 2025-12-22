# Optimizations Implemented - 2025-12-20

**Goal**: Improve from 30% MFU → 40-50% MFU on A100-80GB

## Changes Summary

### 1. Batch Size Increase (Expected: +8-12% MFU)

**File**: `src/MaxText/configs/latency_network.yml`

```diff
- per_device_batch_size: 128
+ per_device_batch_size: 256  # 2x increase

- learning_rate: 1.0e-4
+ learning_rate: 1.414e-4  # sqrt(256/128) = 1.414x scaling
```

**Rationale**:
- Previous monitoring showed "70% GPU memory was idle"
- Doubling batch size will saturate GPU memory (target: 85-95% utilization)
- Learning rate scaled using sqrt rule to maintain training dynamics

**Expected Impact**:
- Memory usage: 70% → 85-95%
- **MFU: 30% → 38-42%** (primary bottleneck was underutilization)
- Tokens/s: ~180k → 250k-300k

---

### 2. XLA GPU Optimizations (Expected: +5-10% MFU)

**File**: `scripts/train/modal_wrapper.py`

Added A100-specific XLA compiler flags:

```python
"XLA_FLAGS": " ".join([
    # Existing flags (kept):
    "--xla_gpu_force_compilation_parallelism=1",
    "--xla_cpu_multi_thread_eigen=false",

    # NEW A100 optimizations:
    "--xla_gpu_enable_latency_hiding_scheduler=true",      # Overlap compute/IO
    "--xla_gpu_enable_triton_softmax_fusion=true",         # Fuse softmax kernels
    "--xla_gpu_triton_gemm_any=true",                      # Optimized matrix ops
    "--xla_gpu_enable_async_collectives=true",             # Async operations
    "--xla_gpu_enable_highest_priority_async_stream=true", # Priority scheduling

    # Profile-guided optimization (learns from runtime):
    "--xla_gpu_pgle_profile_file_or_directory_path=/mnt/pgle_profiles",
]),
```

**What Each Flag Does**:

1. **`latency_hiding_scheduler`**: Overlaps computation with data loading, reducing GPU idle time
2. **`triton_softmax_fusion`**: Combines multiple softmax operations into single GPU kernels (fewer launches)
3. **`triton_gemm_any`**: Uses highly optimized Triton kernels for matrix multiplications
4. **`async_collectives`**: Enables asynchronous operations (future-proofs for multi-GPU)
5. **`highest_priority_async_stream`**: Prioritizes async operations to reduce latency
6. **`pgle_profile_file_or_directory_path`**: Profiles actual runtime and recompiles for better scheduling

**Expected Impact**:
- Kernel fusion: Fewer kernel launches, better occupancy
- Latency hiding: Less GPU idle time waiting for data
- **MFU: +5-10%** (on top of batch size improvement)
- First run: ~+3-5% (still profiling)
- After warmup: ~+5-10% (PGLE optimizations kick in)

---

### 3. Data Pipeline Optimizations (Expected: +2-5% MFU)

**File**: `src/MaxText/configs/latency_network.yml`

```diff
- grain_per_worker_buffer_size: 8
+ grain_per_worker_buffer_size: 16  # 2x prefetch buffer

- grain_per_worker_buffer_size_eval: 8
+ grain_per_worker_buffer_size_eval: 16

- grain_ram_budget_mb: 16384  # 16GB
+ grain_ram_budget_mb: 32768  # 32GB (if available on Modal)
```

**Rationale**:
- Larger buffers allow more aggressive prefetching
- Prevents GPU starvation during batch composition
- Modal A100 instances likely have sufficient RAM

**Expected Impact**:
- Reduced data loading stalls
- **MFU: +2-5%** (if data pipeline was bottleneck)
- Higher benefit when batch size is large (more data needed per step)

---

## Combined Expected Performance

### Before Optimizations
```
MFU:                30%
TFLOPS:             93.6
Tokens/s/device:    ~180k
Step time:          ~0.4s
Memory util:        ~70%
GPU util:           Variable
```

### After Optimizations
```
MFU:                40-50%  (38-42% from batch + 5-10% from XLA)
TFLOPS:             125-156
Tokens/s/device:    250k-320k
Step time:          ~0.25-0.30s
Memory util:        85-95%
GPU util:           >90% sustained

Training speedup:   1.33-1.67x faster
10k steps time:     2.75 hours → 1.65-2.1 hours
```

### Breakdown by Optimization
| Optimization | MFU Gain | Cumulative MFU |
|--------------|----------|----------------|
| Starting point | - | 30% |
| Batch size 2x | +8-12% | 38-42% |
| XLA flags | +5-10% | 43-52% |
| Data pipeline | +2-5% | 45-57% |
| **Best case** | **+15-27%** | **45-57%** |
| **Conservative** | **+10-15%** | **40-45%** |

---

## Verification Steps

### 1. Check Training Logs

Watch for these metrics in the first 10 steps:

```bash
# Expected improvements:
completed step: X, seconds: 0.25-0.30, TFLOP/s/device: 125-156, Tokens/s/device: 250000-320000
```

Compare to baseline:
```bash
# Baseline (30% MFU):
seconds: 0.4, TFLOP/s/device: 94, Tokens/s/device: 180000
```

### 2. Monitor GPU Utilization

```bash
# Run this in a separate terminal during training:
nvidia-smi dmon -s umt -c 100

# Look for:
# GPU util: Should be >90% (was variable)
# Memory: Should be 85-95% (was 70%)
# Temp: Should stay <85°C
```

### 3. Check XLA Compilation

First run will be slower due to:
- XLA compiling with new flags
- PGLE profiling (creates `/mnt/pgle_profiles/`)

Second run should be faster as PGLE optimizations apply.

---

## Monitoring During Training

### Key Metrics to Track

1. **Step time**: Should decrease from ~0.4s to ~0.25-0.30s
2. **TFLOP/s**: Should increase from ~94 to 125-156
3. **Tokens/s**: Should increase from ~180k to 250k-320k
4. **total_weights**: Should stay ~120k-125k (packing still working)

### Signs of Success

✅ **Step time decreased by 25-40%**
✅ **TFLOP/s increased to 125-156 range**
✅ **GPU memory usage 85-95%**
✅ **Sustained GPU utilization >90%**

### Potential Issues

⚠️ **OOM (Out of Memory)**
- If batch size 256 causes OOM, reduce to 192 or 224
- Adjust learning rate accordingly: `lr = 1.0e-4 * sqrt(batch/128)`

⚠️ **No improvement in MFU**
- Check if data pipeline is bottleneck (enable `grain_debug_mode: true`)
- Verify XLA flags are applied: Check Modal logs for compilation messages

⚠️ **Training diverges (NaN loss)**
- Learning rate may need adjustment
- Try reducing to `learning_rate: 1.225e-4` (sqrt(192/128))

---

## Rollback Instructions

If optimizations cause issues, revert with:

```bash
git checkout HEAD -- src/MaxText/configs/latency_network.yml
git checkout HEAD -- scripts/train/modal_wrapper.py
```

Or manually:

### Revert Batch Size
```yaml
per_device_batch_size: 128
learning_rate: 1.0e-4
```

### Revert XLA Flags
```python
"XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=1 --xla_cpu_multi_thread_eigen=false"
```

### Revert Data Pipeline
```yaml
grain_per_worker_buffer_size: 8
grain_per_worker_buffer_size_eval: 8
grain_ram_budget_mb: 16384
```

---

## Next Steps After Testing

### If MFU reaches 40-45% (good!)
- Training is well-optimized for this model size
- Further gains require model architecture changes (larger model)

### If MFU reaches 45-50% (excellent!)
- Near-optimal for small model on A100
- Consider testing larger models to fully utilize GPU

### If MFU is still <40% (investigate)
1. Enable grain debug mode to profile data pipeline
2. Check for slow compilation (first run vs second run)
3. Profile with `nvidia-smi` to identify GPU idle time
4. Consider reducing dropout to 0.0 for slight gain

---

## Files Modified

1. `src/MaxText/configs/latency_network.yml` - Batch size, learning rate, data pipeline
2. `scripts/train/modal_wrapper.py` - XLA optimization flags
3. `OPTIMIZATIONS_IMPLEMENTED.md` - This documentation

## References

- [JAX GPU Performance Tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
- [XLA GPU Architecture](https://openxla.org/xla/gpu_architecture)
- [NVIDIA MaxText Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/maxtext-llama2-70b-dgxc-benchmarking-c)
- OPTIMIZATION_NEXT_STEPS.md - Detailed optimization guide
