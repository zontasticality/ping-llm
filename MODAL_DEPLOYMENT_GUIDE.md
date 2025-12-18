# Running PLAN_3 Preprocessing on Modal

## Quick Start (30 seconds)

```bash
# Deploy and run
modal run scripts/data/modal_run_streaming.py
```

That's it! Takes 20-25 minutes for 200M measurements.

---

## Step-by-Step Instructions

### Prerequisites

1. **Modal account set up**
   ```bash
   pip install modal
   modal token new
   ```

2. **Volume exists with data**
   ```bash
   modal volume ls
   # Should show: ping-llm-data
   ```

3. **Data is uploaded**
   ```
   /mnt/data/training_data.parquet should exist
   ```

---

### Method 1: One-Line Deployment (Easiest)

```bash
cd /home/zyansheep/Projects/ping-llm
modal run scripts/data/modal_run_streaming.py
```

**What happens:**
- ✅ Deploys script to Modal
- ✅ Runs with 8 workers, 32GB RAM
- ✅ Processes data with 28GB memory limit
- ✅ Commits volume automatically
- ✅ Takes ~20-25 minutes

**Expected output:**
```
Starting preprocessing on Modal...
===========================================
MODAL: Streaming Probe Row Preprocessing
===========================================
Input: /mnt/data/training_data.parquet
Output: /mnt/data/probe_rows
Workers: 8
Memory limit: 28.0GB

Step 1: Streaming GROUP BY to disk...
  Grouped 1,234,567 probes in 180s
  Total measurements: 200,000,000

Step 2: Splitting train/test...
  Train: 1,111,110 probes
  Test: 123,457 probes

Step 3: Parallel processing (8 workers)...
  Processing 111 train batches...
  Train: 900s

  Processing 12 test batches...
  Test: 120s

Step 4: Merging...
  Merged 6,789,012 rows

===========================================
COMPLETE!
===========================================
Time: 1234.5s (20.6m)
Throughput: 162,000 meas/sec

Train: 6,111,110 rows, 180,000,000 measurements
Test: 677,902 rows, 20,000,000 measurements

Output: /mnt/data/probe_rows

✓ Complete! Output: /mnt/data/probe_rows
```

---

### Method 2: Custom Parameters

Edit `scripts/data/modal_run_streaming.py` to change defaults:

```python
@app.function(
    cpu=16.0,        # More cores → faster
    memory=65536,    # 64GB RAM
)
def preprocess(
    workers: int = 16,              # More workers
    memory_limit_gb: float = 60.0,  # More memory for DuckDB
):
```

Then run:
```bash
modal run scripts/data/modal_run_streaming.py
```

---

### Method 3: Interactive Shell

For debugging or custom workflows:

```bash
# Start interactive session
modal shell scripts/data/modal_run_streaming.py

# Inside Modal container
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "/mnt/data/training_data.parquet" \
  --output /mnt/data/probe_rows \
  --workers 8 \
  --memory-limit-gb 28
```

---

## Configuration Options

### CPU & Memory Sizing

| Dataset Size | CPU | Memory | Workers | Time | Cost |
|--------------|-----|--------|---------|------|------|
| 50M meas | 4 | 16GB | 4 | 40 min | ~$0.40 |
| 100M meas | 8 | 32GB | 8 | 25 min | ~$0.80 |
| 200M meas | 8 | 32GB | 8 | 20-25 min | ~$0.80 |
| 200M meas | 16 | 64GB | 16 | 15-18 min | ~$1.60 |

**Recommended:** 8 CPU, 32GB RAM (best price/performance)

### Memory Limit Guidelines

```python
# Formula: memory_limit_gb = total_ram_gb - overhead

Examples:
16GB RAM  → memory_limit_gb = 12-14  (leave 2-4GB)
32GB RAM  → memory_limit_gb = 28     (leave 4GB)
64GB RAM  → memory_limit_gb = 60     (leave 4GB)
```

---

## Checking Results

### Option 1: Modal Volume Browse

```bash
# List output files
modal volume ls ping-llm-data data/probe_rows

# Should show:
# data/probe_rows/train.arrayrecord
# data/probe_rows/test.arrayrecord
```

### Option 2: Download and Inspect

```bash
# Download a sample
modal volume get ping-llm-data data/probe_rows/train.arrayrecord ./

# Inspect locally
python scripts/data/inspect_probe_rows.py ./train.arrayrecord
```

### Option 3: Run Inspection on Modal

Add to `modal_run_streaming.py`:

```python
@app.function(
    image=image,
    volumes={"/mnt": volume},
)
def inspect():
    import subprocess
    subprocess.run([
        "python", "-c",
        """
import array_record.python.array_record_module as array_record_module
reader = array_record_module.ArrayRecordReader('/mnt/data/probe_rows/train.arrayrecord')
print(f'Train rows: {reader.num_records():,}')
reader = array_record_module.ArrayRecordReader('/mnt/data/probe_rows/test.arrayrecord')
print(f'Test rows: {reader.num_records():,}')
        """
    ])
```

Then run:
```bash
modal run scripts/data/modal_run_streaming.py::inspect
```

---

## Monitoring Progress

### View Logs in Real-Time

```bash
# In another terminal while running
modal app logs probe-rows-streaming
```

### Check Container Status

```bash
modal app list
```

---

## Troubleshooting

### Error: Out of Memory

**Symptom:**
```
_duckdb.OutOfMemoryException: Out of Memory Error
```

**Solution:** Lower memory limit
```python
def preprocess(
    memory_limit_gb: float = 20.0,  # Reduce from 28
):
```

### Error: Volume not found

**Symptom:**
```
Error: Volume 'ping-llm-data' not found
```

**Solution:** Create volume
```bash
modal volume create ping-llm-data
```

### Error: Input file not found

**Symptom:**
```
No files found matching pattern: /mnt/data/training_data.parquet
```

**Solution:** Check file location
```bash
modal volume ls ping-llm-data data/

# If in different location, update input_pattern
```

### Error: Timeout

**Symptom:**
```
Function timed out after 3600s
```

**Solution:** Increase timeout
```python
@app.function(
    timeout=3600 * 8,  # 8 hours
)
```

### Slow Performance

**Check:**
1. Worker count matches CPU count
2. Memory limit is high enough (but not too high)
3. Data is not fragmented across many small files

**Optimize:**
```python
@app.function(
    cpu=16.0,        # More cores
    memory=65536,    # 64GB
)
def preprocess(
    workers: int = 16,
    memory_limit_gb: float = 60.0,
):
```

---

## Cost Optimization

### Reduce Cost

1. **Use fewer cores** (slower but cheaper)
   ```python
   cpu=4.0, memory=16384, workers=4
   # ~$0.40 for 200M, takes ~40 minutes
   ```

2. **Spot instances** (if available)
   ```python
   @app.function(
       image=image,
       allow_concurrent_inputs=10,
       # Modal may add spot instance support
   )
   ```

3. **Process incrementally**
   - Split data into chunks
   - Process one chunk at a time
   - Merge results

### Monitor Costs

```bash
# Check Modal usage
modal profile current
```

---

## Advanced: Multi-File Processing

If data is split across multiple files:

```python
@app.local_entrypoint()
def main():
    # Process multiple files
    files = [
        "/mnt/data/shard_001.parquet",
        "/mnt/data/shard_002.parquet",
        # ...
    ]

    for i, file in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}: {file}")
        preprocess.remote(
            input_pattern=file,
            output_dir=f"/mnt/data/probe_rows_part_{i}",
        )

    # Then merge all parts (separate script needed)
```

---

## Validation After Preprocessing

### Quick Check

```bash
modal run scripts/data/modal_run_streaming.py::inspect
```

### Full Validation

Download and run locally:
```bash
# Download train set
modal volume get ping-llm-data \
  data/probe_rows/train.arrayrecord \
  ./train.arrayrecord

# Inspect
python scripts/data/inspect_probe_rows.py ./train.arrayrecord --samples 5

# Test pipeline
python scripts/test_plan3_pipeline.py --arrayrecord ./train.arrayrecord
```

---

## Next Steps After Preprocessing

1. **Verify output:**
   ```bash
   modal run scripts/data/modal_run_streaming.py::inspect
   ```

2. **Start training:**
   - Update MaxText config to point to `/mnt/data/probe_rows/train.arrayrecord`
   - Launch training job on Modal

3. **Monitor training:**
   ```bash
   modal app logs your-training-app
   ```

---

## Summary

**Easiest way to run:**
```bash
modal run scripts/data/modal_run_streaming.py
```

**Expected:**
- ✅ Time: 20-25 minutes
- ✅ Cost: ~$0.80
- ✅ Output: ~7M rows from 200M measurements
- ✅ No OOM errors
- ✅ Automatic volume commit

**Configuration:**
- 8 CPU cores
- 32GB RAM
- 28GB memory limit for DuckDB
- 8 parallel workers

**That's it!** Your data will be preprocessed and ready for training.

---

## Questions?

- **How long?** 20-25 minutes for 200M measurements
- **How much?** ~$0.80 with recommended config
- **Any errors?** See Troubleshooting section above
- **Different data location?** Edit `input_pattern` in script
- **Need more speed?** Increase CPU/memory (costs more)
- **Want cheaper?** Reduce to 4 CPU (takes ~40 min)

**Ready to preprocess? Run the one-line command!** 🚀
