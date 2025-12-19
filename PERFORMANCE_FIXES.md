# Performance Optimizations Applied

## ✅ **Fixes Implemented (Priority 2 & 3)**

Based on grain debug output showing **85.47% wait time on data source**, implemented critical optimizations:

### **Fix 1: Corrected Pipeline Ordering** (Priority 2)
**File**: `probe_chunk_pipeline.py:82-95`

**Before** (WRONG):
```python
dataset = dataset.random_map(sampler, seed=shuffle_seed)
dataset = dataset.batch(batch_size, drop_remainder=True)  # ✗ Batch first
dataset = dataset.to_iter_dataset(...)  # Then iter
```

**After** (CORRECT):
```python
dataset = dataset.random_map(sampler, seed=shuffle_seed)
dataset = dataset.to_iter_dataset(...)  # ✓ Iter first
dataset = dataset.batch(batch_size, drop_remainder=True)  # Then batch
```

**Why**: Batching before `to_iter_dataset()` prevents parallelization of individual elements.  
**Expected improvement**: 10-20% throughput increase

---

### **Fix 2: Added Multiprocessing** (Priority 3)
**File**: `probe_chunk_pipeline.py:97-106`

**Added**:
```python
if use_multiprocessing:
    multiprocessing_options = pick_performance_config(
        ds=dataset,
        ram_budget_mb=ram_budget_mb,
        max_workers=None,  # Auto-tune based on CPU cores
        max_buffer_size=None,  # Auto-tune
    ).multiprocessing_options
    
    dataset = dataset.mp_prefetch(multiprocessing_options)
```

**Why**: Enables parallel processing across multiple worker processes for CPU-intensive tokenization.  
**Expected improvement**: 30-50% throughput increase

---

### **Fix 3: Increased Workers** (Based on debug output)
**File**: `latency_network.yml:75-79`

**Before**:
```yaml
grain_worker_count: 4
grain_per_worker_buffer_size: 4
```

**After**:
```yaml
grain_worker_count: 16  # For B200 with 8 CPUs
grain_per_worker_buffer_size: 8
grain_ram_budget_mb: 16384  # 16GB for mp_prefetch
```

**Why**: Debug showed source bottleneck (85% wait) - more workers reduce I/O contention.  
**Expected improvement**: 20-40% throughput increase

---

## 📊 **Expected Results**

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| **Tokens/sec** | 52K | 200-400K (4-8x) |
| **Source wait %** | 85.47% | 20-40% |
| **ProbeRowSampler %** | 12.39% | 30-50% (now the bottleneck) |
| **Step time** | ~2.5s | ~0.3-0.6s |

---

## 🔄 **To Apply These Fixes**

### Option 1: Re-run on Modal (Automatic)
Your code is already updated. Just re-run:
```bash
modal run scripts/train/modal_train_with_wandb_sync.py::run \
  --run-name optimized_test \
  --steps 1000 \
  --batch-size 128 \
  --wandb-project ping-llm-optimized
```

The fixes will apply automatically.

### Option 2: Stop Current Run & Restart
If you have a training run currently active:
```bash
# Stop current run
modal app stop ping-llm-maxtext-wandb-sync

# Start new optimized run
modal run scripts/train/modal_train_with_wandb_sync.py::run \
  --run-name optimized_test \
  --steps 1000
```

---

## 📈 **What to Monitor**

After restarting, watch for:

1. **New log messages**:
   ```
   [NETWORK BACKEND] Optimizations: parallel tokenization, mp_prefetch enabled
   ```

2. **Improved grain debug** (after 60s):
   ```
   SourceMapDataset: 20-40% wait time (was 85%)  ← Should be much lower
   RandomMapDataset: 30-50% wait time (becomes bottleneck)
   ```

3. **Higher tokens/sec**:
   ```
   Tokens/s/device: 200000-400000 (was 52K)
   ```

---

## 🎯 **Next Optimizations (If Still Slow)**

If after these fixes you're still below 200K tokens/sec:

### **Priority 4: Optimize RNG Type Checking**
**File**: `_probe_chunk_datasource.py`

Currently checks `hasattr(rng, 'integers')` repeatedly in hot path.

**Fix**: Cache RNG type in `__init__`:
```python
def __init__(self, ...):
    self.is_numpy_rng = hasattr(rng, 'integers')  # Cache once
```

**Expected**: 5-10% improvement in tokenization stage

### **Priority 5: Profile Tokenization**
If `ProbeRowSampler` becomes >60% bottleneck:

1. Profile `encode_measurement()` function
2. Consider vectorization or batching
3. Check for redundant string operations

---

## 🐛 **If Something Breaks**

### Rollback to Previous Version
```python
# In probe_chunk_pipeline.py, set:
use_multiprocessing=False  # Disable mp_prefetch

# In latency_network.yml, reduce:
grain_worker_count: 4
grain_per_worker_buffer_size: 4
```

### Common Issues

**"Out of memory"**:
```yaml
grain_ram_budget_mb: 8192  # Reduce from 16384
grain_worker_count: 8  # Reduce from 16
```

**"Too many open files"**:
```bash
ulimit -n 4096  # Increase file descriptor limit
```

---

## 📝 **Changes Summary**

| File | Change | Lines |
|------|--------|-------|
| `probe_chunk_pipeline.py` | Fixed ordering + mp_prefetch | 82-106 |
| `_network_data_processing.py` | Pass multiprocessing params | 109-124, 217-230 |
| `latency_network.yml` | Increased workers | 75-79 |

All changes follow MaxText best practices and grain documentation.
