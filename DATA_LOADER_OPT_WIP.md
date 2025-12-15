# Data Loading Optimization - Work in Progress

## Problem Statement
- Training with batch_size=256 achieving only **15% MFU** (Model Flops Utilization)
- Suspected data starvation: GPU waiting for data instead of computing
- System: 16 CPU cores, 60GB RAM, 1x A100 GPU

## Current Configuration
```yaml
grain_worker_count: 12  # 12 workers out of 16 cores
grain_per_worker_buffer_size: 32  # 384 batches buffered total
```

## Data Pipeline Status

### ✓ What's Working
1. **Eager Loading**: All 1.8GB parquet files preloaded into memory at startup
   - Code: `eager_load=True` in `_network_grain_integration.py:115`
   - Cache size set to `len(data_files)` (all 180 train files)
   - Confirmed by `[EAGER LOAD]` print statements showing successful preload

2. **Multicore Setup**: 16 cores allocated (verified with `os.sched_getaffinity(0)`)

### ❌ What's Broken
**mp_prefetch() Implementation Issue**

Location: `network_grain_datasource.py:447-460`

**Problem**: Calling `mp_prefetch()` on wrong dataset type
```python
dataset = dataset.batch(batch_size, drop_remainder=True)  # Returns BatchMapDataset
dataset = dataset.mp_prefetch(multiprocessing_options)    # ❌ BatchMapDataset has no mp_prefetch!
```

**Error**: `AttributeError: 'BatchMapDataset' object has no attribute 'mp_prefetch'`

### Root Cause Analysis

From grain source code investigation:
- `mp_prefetch()` exists on **MapDataset** class only
- Must be called BEFORE `to_iter_dataset()`
- Order matters: `mp_prefetch()` → `to_iter_dataset()`

**MaxText Reference** (`_grain_data_processing.py:270-284`):
```python
dataset = dataset.batch(batch_size, batch_fn=batch_fn)
dataset = dataset.map(ShiftData(...))  # Still MapDataset
multiprocessing_options = grain.MultiprocessingOptions(...)
dataset = dataset.mp_prefetch(multiprocessing_options)  # ✓ Works on MapDataset
return dataset
```

## Investigation Findings

### Grain API Discovery
1. **mp_prefetch signature** (from `.venv/lib/python3.12/site-packages/grain/_src/python/dataset/dataset.py:1294`):
   ```python
   def mp_prefetch(
       self,
       options: MultiprocessingOptions | None = None,
       worker_init_fn: Callable[[int, int], None] | None = None,
       sequential_slice: bool = False,
   ) -> IterDataset[T]:
   ```

2. **Available on**: `MapDataset` instances only
3. **Returns**: `IterDataset` (already iterable, no need for `to_iter_dataset()`)

### Dataset Type Chain
```
source() → SourceMapDataset (has mp_prefetch ✓)
  ↓
map() → MapDataset (has mp_prefetch ✓)
  ↓
batch() → BatchMapDataset (NO mp_prefetch ❌)
  ↓
to_iter_dataset() → IterDataset
```

## Next Steps

### Option 1: Call mp_prefetch BEFORE batch()
```python
dataset = dataset.map(tokenizer)
# Apply mp_prefetch here (before batch)
if num_workers > 0:
    multiprocessing_options = grain.MultiprocessingOptions(...)
    dataset = dataset.mp_prefetch(multiprocessing_options)  # Returns IterDataset
    # Now batch on the IterDataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
else:
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.to_iter_dataset()
```

### Option 2: Use to_iter_dataset with ReadOptions (current approach)
Keep using `ReadOptions` with `num_threads` instead of `mp_prefetch`:
```python
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.to_iter_dataset(
    read_options=grain.ReadOptions(
        num_threads=num_workers,
        prefetch_buffer_size=prefetch_buffer_size,
    )
)
```

### Option 3: Remove batch-level operations after mp_prefetch
Move all transformations before batching, then apply mp_prefetch on MapDataset.

## Performance Tuning Targets

### Buffer Size Calculation
- Workers: 12
- Buffer per worker: 32
- **Total buffered batches: 384**
- Batch size: 256
- **Memory for buffers**: ~384 batches × ~10MB/batch = ~4GB buffered data

### Recommendations from Google Developers Blog
Source: https://developers.googleblog.com/en/building-high-performance-data-pipelines-with-grain-and-arrayrecord/

> "The optimal number depends on the CPU cores available on your machine and the complexity of your map function. If you notice your accelerator is often idle waiting for data, increasing this value can significantly improve throughput."

### Current Hypothesis
With **eager loading + 12 workers + 32 buffer**, data should not be bottleneck IF:
1. mp_prefetch is correctly implemented
2. Workers can actually access preloaded memory (thread-safe cache confirmed)
3. No GIL contention (multiprocessing should bypass this)

## Files Modified
1. `network_grain_datasource.py:447-460` - Added mp_prefetch (BROKEN)
2. `src/MaxText/configs/latency_network.yml:61-62` - Updated worker config
3. `src/MaxText/input_pipeline/_network_grain_integration.py:115` - Set eager_load=True

## Test Commands
```bash
# Test data loading with current config
python << 'EOF'
import glob
from network_grain_datasource import create_grain_pipeline

train_files = sorted(glob.glob("data/sharded/train/*.parquet"))[:10]
dataset = create_grain_pipeline(
    parquet_files=train_files,
    batch_size=32,
    num_workers=12,
    cache_size=10,
    eager_load=True,
    prefetch_buffer_size=32,
)
iterator = iter(dataset)
batch = next(iterator)
print(f"✓ Batch shape: {batch['inputs'].shape}")
EOF
```

## References
- MaxText implementation: `src/MaxText/input_pipeline/_grain_data_processing.py:270-284`
- Grain source: `.venv/lib/python3.12/site-packages/grain/_src/python/dataset/dataset.py:1294`
- Google blog: https://developers.googleblog.com/en/building-high-performance-data-pipelines-with-grain-and-arrayrecord/
