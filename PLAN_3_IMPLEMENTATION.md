# PLAN_3 Implementation Guide

File-by-file changes needed to implement DATA_LOADING_PLAN_3_CLEAN.md

## Files to Modify

### 1. `src/MaxText/input_pipeline/_probe_chunk_datasource.py`
**Status:** MAJOR REWRITE NEEDED

**Current state (PLAN_1):**
- Reads pre-tokenized chunks from ArrayRecord
- Schema: {src_id, bucket_start_time, part_id, tokens (bytes), meas_offsets (bytes), n_tokens, n_measurements}
- ProbeChunkCropper does measurement-boundary aligned cropping on pre-tokenized data

**Required changes for PLAN_3:**
- Rename class: `ProbeChunkDataSource` → `ProbeRowDataSource`
- New schema: {src_id, measurements (bytes), n_measurements, metadata}
  - measurements: PyArrow RecordBatch serialized as IPC bytes
    - Fields: event_time (timestamp), dst_addr (str), ip_version (int8), rtt (float32)
- Remove: `_deserialize_tokens()`, `_deserialize_offsets()`
- Add: `_deserialize_measurements()` - unpack PyArrow IPC to list of measurement dicts

**ProbeChunkCropper → ProbeRowSampler:**
- Complete redesign per PLAN_3 sampling logic
- Calculate K = min(ceil(n_measurements / 30), 16)
- For each context generation:
  - Small row path (n < 34): tokenize all
  - Large row path (n >= 34):
    - Sample window size (log-uniform)
    - Sample offset
    - Sample measurements from window
    - Sort by timestamp
  - Select timestamp mode (40/30/30)
  - Tokenize measurements using `tokenization.encode_measurement()`
    - Random field order
    - Timestamp encoding per mode
  - Pad to 1024 tokens

**Key additions:**
- Import `tokenization.encode_measurement()` from project root
- Implement log-uniform sampling
- Implement timestamp mode selection and masking
- Field order randomization during tokenization

**Line count estimate:** ~400-500 lines (currently 329)


### 2. `src/MaxText/input_pipeline/probe_chunk_pipeline.py`
**Status:** MINOR UPDATES

**Current state:**
- Wrapper function `build_probe_chunk_dataset()`
- Delegates to ProbeChunkDataSource + ProbeChunkCropper

**Required changes:**
- Update docstring: PLAN_1 → PLAN_3
- Update import: ProbeChunkDataSource → ProbeRowDataSource (if renamed)
- Update import: ProbeChunkCropper → ProbeRowSampler (if renamed)
- Update comments to reflect big-row sampling

**Line count estimate:** ~90 lines (currently 90, minimal changes)


### 3. `src/MaxText/input_pipeline/_network_grain_integration.py`
**Status:** MINOR UPDATES

**Current state:**
- Integration layer with two functions:
  - `create_network_measurement_dataset()` - PLAN_2 (obsolete, uses deleted file)
  - `create_probe_chunk_dataset()` - PLAN_1 wrapper

**Required changes:**
- **Remove** `create_network_measurement_dataset()` entirely (lines 31-128)
  - Depends on deleted `network_grain_datasource.py`
  - No longer needed
- Update `create_probe_chunk_dataset()` docstring: PLAN_1 → PLAN_3
- Update log messages: DATA_LOADING_PLAN_1 → DATA_LOADING_PLAN_3
- Update prerequisite note: point to new preprocessing script

**Line count estimate:** ~100 lines (currently 212, after removing PLAN_2 function)


### 4. `tokenization.py`
**Status:** NO CHANGES (reusable as-is!)

**Current state:**
- `encode_measurement()` function (line 480)
- Takes measurement dict, returns token list
- Supports delta timestamps, field randomization
- Already has all features we need

**Usage in PLAN_3:**
```python
from tokenization import encode_measurement

# In ProbeRowSampler.random_map():
for measurement in sampled_measurements:
    tokens = encode_measurement(
        measurement,
        prev_timestamp=prev_ts,  # for delta encoding
        randomize_field_order=True,  # PLAN_3 feature
        include_timestamp=(timestamp_mode decision)
    )
```

**No changes needed!**


## New Files to Create

### 5. `scripts/data/create_probe_rows.py`
**Status:** NEW FILE

**Purpose:** Preprocessing script to create PLAN_3 ArrayRecord files

**Functionality:**
```python
# High-level flow:
1. Read input Parquet (sharded or single file)
2. Group by src_addr, order by event_time
3. For each probe group:
   a. Serialize measurements to compact binary (PyArrow IPC)
   b. If size > 8MB:
      - Write current row
      - Start new row with remaining measurements
   c. Track metadata (n_measurements, time_span)
4. Write to ArrayRecord:
   - train/train_shard_*.arrayrecord
   - test/test_shard_*.arrayrecord
5. Split train/test by probe (90/10)
```

**Schema to write:**
```python
{
    'src_id': int64,
    'measurements': binary,  # PyArrow IPC RecordBatch
    'n_measurements': int32,
    'time_span_seconds': float64,
    'first_timestamp': timestamp('us'),
    'last_timestamp': timestamp('us'),
}
```

**Measurements RecordBatch schema:**
```python
{
    'event_time': timestamp('us'),
    'dst_addr': string,
    'ip_version': int8,
    'rtt': float32,
}
```

**Key implementation notes:**
- Use DuckDB for group-by-src_addr + order-by
- Stream processing (don't load all probes into memory)
- Track row sizes in bytes (PyArrow IPC size)
- 8MB threshold = 8 * 1024 * 1024 bytes
- Use ArrayRecordWriter from array_record module

**Estimated line count:** ~300-400 lines

**CLI interface:**
```bash
python scripts/data/create_probe_rows.py \
  --input data/sharded/train/*.parquet \
  --output data/probe_rows \
  --max-row-size 8388608 \
  --train-ratio 0.9 \
  --workers 4
```


### 6. `scripts/data/modal_create_probe_rows.py` (OPTIONAL)
**Status:** NEW FILE (optional, for Modal users)

**Purpose:** Modal wrapper for create_probe_rows.py

**Functionality:**
- Same as create_probe_rows.py but runs on Modal
- Mount Volume at /mnt
- Input: /mnt/data/training_data.parquet
- Output: /mnt/data/probe_rows/

**Estimated line count:** ~150-200 lines

**Can be deferred** - local script is sufficient for now


### 7. `scripts/data/inspect_probe_rows.py` (OPTIONAL, helpful for debugging)
**Status:** NEW FILE (utility)

**Purpose:** Inspect PLAN_3 ArrayRecord files

**Functionality:**
```python
# Print stats about probe rows:
- Total rows
- Row size distribution
- Measurements per row distribution
- Sample a few rows and pretty-print measurements
```

**Estimated line count:** ~100 lines


## Files That Can Be Deleted (Already Done!)

✅ All obsolete files deleted in previous cleanup:
- PLAN_2, DATA_LOADING_PLAN_1.md, etc.
- network_grain_datasource.py (PLAN_2)
- scripts/data/probe_chunk_preprocess.py (PLAN_1)
- tests/test_probe_chunks.py (PLAN_1)


## Summary of Work

### Critical Path (Must implement):
1. ✅ `tokenization.py` - no changes needed
2. 🔨 `scripts/data/create_probe_rows.py` - NEW, ~300-400 lines
3. 🔨 `src/MaxText/input_pipeline/_probe_chunk_datasource.py` - MAJOR REWRITE, ~400-500 lines
4. 🔨 `src/MaxText/input_pipeline/probe_chunk_pipeline.py` - MINOR UPDATES, ~10 line changes
5. 🔨 `src/MaxText/input_pipeline/_network_grain_integration.py` - REMOVE PLAN_2, update docs, ~50 line changes

### Nice to have:
6. 🆕 `scripts/data/modal_create_probe_rows.py` - NEW, ~150-200 lines (Modal users)
7. 🆕 `scripts/data/inspect_probe_rows.py` - NEW, ~100 lines (debugging)

### Testing/validation:
8. Update `scripts/data/analyze_padding.py` - should work with PLAN_3 (validate <5% padding)
9. Update `scripts/data/modal_sample_probe_chunks.py` - may need updates for new schema
10. Create simple smoke test: read row → sample contexts → verify output shape

## Estimated Total Work:
- **Lines to write/rewrite:** ~1000-1200 lines
- **Implementation time:** 1-2 days
- **Testing/validation:** 0.5-1 day
- **Total:** 1.5-3 days


## Implementation Order Recommendation:

### Phase 1: Preprocessing (Day 1)
1. Write `scripts/data/create_probe_rows.py`
2. Test on small Parquet sample
3. Validate ArrayRecord output (row sizes, measurement deserialization)

### Phase 2: Data Loading (Day 1-2)
4. Rewrite `_probe_chunk_datasource.py` → ProbeRowDataSource + ProbeRowSampler
5. Update `probe_chunk_pipeline.py` (minimal)
6. Update `_network_grain_integration.py` (minimal)

### Phase 3: Integration Testing (Day 2)
7. Run preprocessing on full dataset
8. Test data loading pipeline (single batch)
9. Run padding analysis (expect <5%)
10. Run short training run (10-100 steps)

### Phase 4: Validation (Day 2-3)
11. Verify K contexts per row
12. Verify timestamp mode distribution (40/30/30)
13. Verify multi-scale window sampling
14. Performance profiling (tokens/sec)
