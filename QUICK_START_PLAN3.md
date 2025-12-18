# PLAN_3 Quick Start Guide

## TL;DR - Get Started in 30 Seconds

### For Modal (200M measurements)

```bash
# Run the streaming preprocessor (memory-safe, fastest)
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "/mnt/data/training_data.parquet" \
  --output /mnt/data/probe_rows \
  --workers 8 \
  --memory-limit-gb 28
```

**Expected:** 20-25 minutes, output: ~7M rows

---

### For Local Testing

```bash
# Test on small dataset
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/sharded/test/*.parquet" \
  --output data/probe_rows \
  --workers 4
```

**Expected:** A few seconds to minutes depending on data size

---

## Which Script Do I Use?

### 🟢 Use `create_probe_rows_parallel_streaming.py` if:
- You have **> 50M measurements** ✅
- You're running on **Modal** ✅
- You're getting **OOM errors** ✅
- You have **< 16GB RAM** ✅
- You want the **fastest performance** ✅

### 🟡 Use `create_probe_rows_parallel.py` if:
- You have 1M - 50M measurements
- You have 16GB+ RAM available
- You want simple parallel processing

### 🔴 Use `create_probe_rows.py` if:
- You're debugging issues
- You have < 1M measurements
- You want to understand the code

---

## Common Commands

### Auto-detect settings (easiest)
```bash
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/**/*.parquet" \
  --output data/probe_rows
```

### Specify workers and memory
```bash
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/**/*.parquet" \
  --output data/probe_rows \
  --workers 8 \
  --memory-limit-gb 24
```

### Low memory mode (2GB)
```bash
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/**/*.parquet" \
  --output data/probe_rows \
  --workers 2 \
  --memory-limit-gb 2
```

---

## Inspect Output

```bash
# View statistics
python scripts/data/inspect_probe_rows.py \
  data/probe_rows/train.arrayrecord \
  --samples 3
```

---

## Test Data Loading

```bash
# Test the full PLAN_3 pipeline
python scripts/test_plan3_pipeline.py \
  --arrayrecord data/probe_rows/train.arrayrecord
```

---

## Troubleshooting

### OOM Error
```
Error: Out of Memory Error: failed to allocate data
```

**Solution:** Lower memory limit
```bash
--memory-limit-gb 8  # or lower
```

### Too Slow
```
Taking hours to process
```

**Solution:** Add more workers
```bash
--workers 16  # use more CPU cores
```

### Disk Space Error
```
Error: No space left on device
```

**Solution:** Point to larger disk
```bash
export TMPDIR=/path/to/large/disk
```

---

## Performance Expectations

| Dataset Size | Workers | Memory | Time |
|--------------|---------|--------|------|
| 500K meas | 4 | 2GB | 5s |
| 10M meas | 4 | 8GB | 2 min |
| 50M meas | 8 | 16GB | 8 min |
| 200M meas | 8 | 32GB | 20-25 min |

---

## Next Steps

After preprocessing:

1. **Inspect output:**
   ```bash
   python scripts/data/inspect_probe_rows.py data/probe_rows/train.arrayrecord
   ```

2. **Test pipeline:**
   ```bash
   python scripts/test_plan3_pipeline.py --arrayrecord data/probe_rows/train.arrayrecord
   ```

3. **Start training:**
   ```python
   from MaxText.input_pipeline._network_grain_integration import create_probe_chunk_dataset

   dataset = create_probe_chunk_dataset(
       data_file_pattern="data/probe_rows/train.arrayrecord",
       batch_size=32,
   )
   ```

---

## Full Documentation

- **PREPROCESSING_SCRIPT_GUIDE.md** - Detailed decision guide
- **PREPROCESSING_PERFORMANCE_COMPARISON.md** - Performance benchmarks
- **PLAN_3_IMPLEMENTATION_SUMMARY.md** - Complete implementation details

---

## Questions?

**Q: Which script is fastest?**
A: `create_probe_rows_parallel_streaming.py` (23x speedup, 108K meas/sec)

**Q: Which script for Modal?**
A: `create_probe_rows_parallel_streaming.py` (only one that works with 200M)

**Q: Do I need to delete old scripts?**
A: No, keep all three for different use cases

**Q: What if I get OOM?**
A: Use streaming version with lower `--memory-limit-gb`

**Q: How much faster is this?**
A: 23x faster than sequential, solves OOM issues

---

**Ready to preprocess? Run the streaming version! 🚀**
