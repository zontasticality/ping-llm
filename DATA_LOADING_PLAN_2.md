# Probe-Centric Data Loading with Runtime Multi-Scale Sampling

**Status:** Recommended approach (supersedes DATA_LOADING_PLAN_1)
**Key improvement:** Eliminates padding waste while enabling multi-scale temporal learning

---

## 1. Core Insight

**The Problem with PLAN_1:**
- Fixed 5-minute time buckets create arbitrary boundaries
- Pre-tokenized chunks lead to 50-90% padding waste
- No multi-scale temporal learning (model only sees one time resolution)
- Storage overhead from duplicated pre-tokenized data

**The Solution (PLAN_2):**
- Store **entire probe history** (all measurements) in one ArrayRecord row per probe
- Sample windows **at runtime** with variable temporal strides
- Tokenize **on-the-fly** (~3ms overhead, negligible vs GPU time)
- Achieve **minimal padding** (<5% typical) and **infinite data augmentation**

---

## 2. Why This Works: Storage vs Compute Trade-off

### ArrayRecord Row Access is Fast
```
Reading one ArrayRecord row (entire probe):
- Single sequential disk read: ~2MB @ 500MB/s = 4ms
- Deserialize measurements: ~1-2ms
- Sample window with stride: ~0.1ms
- Tokenize 64 measurements: ~0.6ms
Total overhead: ~3-5ms per training example
```

Compare to:
- GPU forward pass: 20-100ms per batch
- Data loading overhead: <5% of training time with 16+ workers

### Why Not Parquet?
```
Parquet (columnar compression):
- Accessing individual rows scattered across columns: 50-100ms
- Must decompress column chunks for each access
- Designed for aggregations, not random row access

ArrayRecord (row-based):
- Each row is independent binary blob
- Random access = single seek + sequential read
- Designed for ML training with shuffling
```

**Key insight:** We read **entire rows** (entire probe histories), not individual measurements. ArrayRecord is perfect for this.

---

## 3. Multi-Scale Temporal Sampling

The model must learn network dynamics at **all time scales** (1 second to 1 day apart). Storing only one time resolution (e.g., 5-minute windows) limits learning.

### Stride-Based Sampling

**Idea:** Sample measurements with different strides to create windows spanning different time periods.

```
Probe with 10,000 measurements over 1 month (1 ping/sec avg):

Stride 1 (dense):
  [M100, M101, M102, ..., M163]  → 64 measurements spanning ~1 minute

Stride 4 (medium):
  [M1000, M1004, M1008, ..., M1252]  → 64 measurements spanning ~4 minutes

Stride 16 (sparse):
  [M5000, M5016, M5032, ..., M6008]  → 64 measurements spanning ~17 minutes

Stride 128 (very sparse):
  [M0, M128, M256, ..., M8064]  → 64 measurements spanning ~2.2 hours
```

**Each window has ~1024 tokens but represents different temporal resolutions.**

### Benefits

1. **Model learns all time scales** - from short-term bursts to long-term trends
2. **Data augmentation** - same probe data generates diverse training examples
3. **Realistic inference** - probes naturally vary in ping frequency
4. **No probe diversity assumption** - even one probe yields multi-scale data

---

## 4. Storage Format: Probe-Level ArrayRecord

### Schema (One Row = One Probe)

```python
probe_record = {
    'src_id': int64,              # Numeric probe identifier
    'n_measurements': int32,       # Total measurements for this probe
    'time_span_seconds': float64,  # Total time span (first to last measurement)
    'measurements': binary,        # Serialized PyArrow RecordBatch of measurements
}

# measurements RecordBatch contains:
measurements_schema = {
    'event_time': timestamp('us'),  # Measurement timestamp
    'dst_addr': string,             # Destination IP
    'ip_version': int8,             # 4 or 6
    'rtt': float32,                 # RTT in ms (negative = failed)
}
```

### File Layout

```
data/
  probes/
    train/
      train_shard_00000.arrayrecord  # ~1000 probes per file
      train_shard_00001.arrayrecord
      ...
    test/
      test_shard_00000.arrayrecord
      ...
```

**Sharding rationale:**
- Each file contains multiple probes (rows)
- Grain can efficiently shuffle and parallel-load across files
- Enables distributed training (different workers load different shards)

### Storage Size Estimate

```
Probe with 10,000 measurements:
  10,000 × 30 bytes/measurement (raw) = 300 KB
  Compressed (Snappy/LZ4): ~150 KB

Dataset with 1,000 probes:
  1,000 × 150 KB = 150 MB per shard

Typical dataset (50,000 probes):
  50 shards × 150 MB = 7.5 GB total
```

**Compare to PLAN_1:**
- Pre-tokenized chunks: ~50 GB (10x larger due to duplication)
- Multi-scale windows: Would be 100+ GB if pre-computed

---

## 5. Runtime Sampling Pipeline

### Grain Pipeline (Conceptual)

```python
class ProbeWindowSampler(grain.RandomMapTransform):
    """Sample multi-scale windows from probe measurements at runtime."""

    def __init__(self, crop_size=1024, avg_tokens_per_meas=20):
        self.crop_size = crop_size
        self.target_measurements = crop_size // avg_tokens_per_meas  # ~51 measurements

    def random_map(self, probe_record, rng):
        # 1. Deserialize measurements (~1-2ms)
        measurements = deserialize_measurements(probe_record['measurements'])

        # 2. Choose random stride (multi-scale sampling)
        max_stride = max(1, len(measurements) // self.target_measurements)
        stride = sample_geometric_stride(rng, max_stride)

        # 3. Choose random window start
        strided_length = (len(measurements) + stride - 1) // stride
        if strided_length <= self.target_measurements:
            start_idx = 0
        else:
            max_start_strided = strided_length - self.target_measurements
            start_strided = rng.randint(0, max_start_strided)
            start_idx = start_strided * stride

        # 4. Extract window with stride (~0.1ms)
        window = measurements[start_idx::stride][:self.target_measurements]

        # 5. Tokenize (~0.6ms)
        tokens = tokenize_measurements_with_delta_timestamps(window)

        # 6. Apply timestamp masking (training mode)
        tokens = apply_timestamp_masking(tokens, rng, mode_probs=(0.4, 0.3, 0.3))

        # 7. Pad to crop_size and format for MaxText
        return format_for_training(tokens, crop_size=self.crop_size)
```

### Stride Sampling Strategy

**Geometric progression:** Strides sample from {1, 2, 4, 8, 16, 32, 64, 128, ...}

```python
def sample_geometric_stride(rng, max_stride):
    """Sample stride from geometric distribution."""
    if max_stride <= 1:
        return 1

    # Find largest power of 2 <= max_stride
    max_power = int(np.log2(max_stride))

    # Sample uniformly from log space
    power = rng.randint(0, max_power + 1)
    stride = 2 ** power

    return min(stride, max_stride)
```

**This gives:**
- Dense windows (stride 1-2): Common
- Medium windows (stride 4-16): Common
- Sparse windows (stride 32+): Less common but still sampled

---

## 6. Timestamp Masking (Training-Time Augmentation)

**Goal:** Model must handle varying timestamp availability (realistic for federated deployment).

### Three Training Modes (40/30/30 split)

1. **Full timestamps (40%):** All measurements include timestamps
   - Teaches model to use temporal information when available
   - Delta encoding naturally appears (2 tokens vs 9 for absolute)

2. **No timestamps (30%):** Remove all timestamp tokens
   - Model must infer temporal patterns from measurement content
   - Learns RTT patterns, destination clustering, etc.

3. **Mixed timestamps (30%):** Random subset of measurements have timestamps
   - Model learns to combine explicit and implicit temporal cues
   - Most realistic for partial observability scenarios

### Implementation

```python
def apply_timestamp_masking(tokens, rng, mode_probs=(0.4, 0.3, 0.3)):
    """Apply timestamp masking augmentation."""
    mode = rng.choices(['full', 'none', 'mixed'], weights=mode_probs)[0]

    if mode == 'full':
        return tokens  # Keep all timestamps

    elif mode == 'none':
        return remove_all_timestamp_tokens(tokens)

    else:  # mode == 'mixed'
        # Keep timestamps with 50% probability per measurement
        return selectively_remove_timestamps(tokens, keep_prob=0.5, rng=rng)
```

**Note:** This happens **after** tokenization, so we can identify and remove timestamp tokens (IDs 5, 6, 7 in vocabulary).

---

## 7. Padding Analysis

### Minimal Padding Compared to PLAN_1

**PLAN_1 (5-minute buckets):**
```
Bucket too small: 200 tokens → pad 824 tokens (80% waste)
Bucket moderate: 600 tokens → pad 424 tokens (41% waste)
Average padding: 50-70% across dataset
```

**PLAN_2 (runtime sampling):**
```
Sparse probe (100 measurements total):
  Tokenize all → ~2000 tokens
  Crop to 1024 → 0% padding (exact fit)

Dense probe (10,000 measurements):
  Sample 51 measurements with stride → ~1020 tokens
  Pad to 1024 → 4 tokens padding (0.4% waste)

Average padding: <5% across dataset
```

**Why the difference?**
- PLAN_1: Fixed buckets → variable token counts → forced padding
- PLAN_2: Variable sampling → target exact crop_size → minimal padding

---

## 8. Preprocessing Pipeline

### Input: Sharded Parquet (existing)
```
data/sharded/
  train/
    shard_0000.parquet
    shard_0001.parquet
    ...
```

### Output: Probe-Level ArrayRecord
```
data/probes/
  train/
    train_shard_00000.arrayrecord
    ...
  test/
    test_shard_00000.arrayrecord
    ...
```

### Processing Steps

```python
# probe_level_preprocess.py

def preprocess_parquet_to_probe_arrayrecord(
    input_pattern: str,
    output_dir: str,
    train_ratio: float = 0.9,
    probes_per_shard: int = 1000,
):
    """Convert measurement Parquet to probe-level ArrayRecord."""

    # 1. Read all parquet files and group by src_addr
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE probe_groups AS
        SELECT
            src_addr,
            ROW_NUMBER() OVER (ORDER BY src_addr) - 1 as src_id,
            LIST(STRUCT_PACK(
                event_time := event_time,
                dst_addr := dst_addr,
                ip_version := ip_version,
                rtt := rtt
            ) ORDER BY event_time) as measurements
        FROM read_parquet('{input_pattern}')
        GROUP BY src_addr
        ORDER BY src_addr;
    """)

    # 2. Split into train/test by probe
    n_probes = con.execute("SELECT COUNT(*) FROM probe_groups").fetchone()[0]
    train_cutoff = int(n_probes * train_ratio)

    # 3. Write to sharded ArrayRecord files
    write_probe_arrayrecords(
        con,
        output_dir,
        train_cutoff,
        probes_per_shard,
    )
```

### Serialization Format

```python
def serialize_probe_record(src_id: int, measurements: list) -> bytes:
    """Serialize one probe's measurements to ArrayRecord entry."""

    # Convert measurements to PyArrow Table
    meas_table = pa.Table.from_pylist([{
        'event_time': m['event_time'],
        'dst_addr': m['dst_addr'],
        'ip_version': m['ip_version'],
        'rtt': m['rtt'],
    } for m in measurements])

    # Serialize table to IPC format (fast, zero-copy deserialization)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, meas_table.schema)
    writer.write_table(meas_table)
    writer.close()
    meas_bytes = sink.getvalue().to_pybytes()

    # Create probe record
    record = {
        'src_id': src_id,
        'n_measurements': len(measurements),
        'time_span_seconds': (measurements[-1]['event_time'] - measurements[0]['event_time']).total_seconds(),
        'measurements': meas_bytes,
    }

    # Serialize to single-row PyArrow RecordBatch
    return record_to_ipc_bytes(record)
```

---

## 9. Training Pipeline

### Grain DataSource

```python
class ProbeDataSource(grain.RandomAccessDataSource):
    """Read probe-level ArrayRecord files."""

    def __init__(self, arrayrecord_files: list[str]):
        self.files = arrayrecord_files
        self.readers = [ArrayRecordReader(f) for f in files]

        # Build index: (file_idx, row_idx) for each probe
        self.probe_index = []
        for file_idx, reader in enumerate(self.readers):
            for row_idx in range(reader.num_records()):
                self.probe_index.append((file_idx, row_idx))

    def __len__(self):
        return len(self.probe_index)

    def __getitem__(self, index):
        """Get one probe's data."""
        file_idx, row_idx = self.probe_index[index]
        record_bytes = self.readers[file_idx].read([row_idx])[0]
        return deserialize_probe_record(record_bytes)
```

### Full Pipeline

```python
def create_probe_training_pipeline(
    arrayrecord_pattern: str,
    batch_size: int = 256,
    crop_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 16,
):
    """Create Grain pipeline for probe-level training."""

    # 1. Find ArrayRecord files
    files = glob.glob(arrayrecord_pattern)

    # 2. Create data source (one element = one probe)
    source = ProbeDataSource(files)

    # 3. Wrap in MapDataset
    dataset = grain.MapDataset.source(source)

    # 4. Shuffle probes (client-first sampling inherent)
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    # 5. Sample windows with multi-scale strides + tokenize
    dataset = dataset.random_map(
        ProbeWindowSampler(crop_size=crop_size),
        seed=42
    )

    # 6. Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # 7. Convert to IterDataset with parallel workers
    dataset = dataset.to_iter_dataset(
        read_options=grain.ReadOptions(
            num_threads=num_workers,
            prefetch_buffer_size=2,
        )
    )

    return dataset
```

---

## 10. Client-First Sampling (Federated Learning)

**Inherently probe-centric:** Each dataset element = one probe's data.

### Natural Properties

1. **Shuffle at probe level** - Grain's `.shuffle()` operates on probes
2. **No index needed** - One row = one probe = one client
3. **Balanced by default** - Each probe sampled equally (adjustable with weights)

### Simulating Federated Training

```python
def sample_federated_batch(dataset_iter, probes_per_round: int):
    """Simulate federated learning round."""

    # Each batch is already from different probes (shuffled)
    batch = next(dataset_iter)

    # batch['inputs'].shape = (256, 1024)
    # Each of 256 examples is from a different probe's window

    return batch
```

**Key difference from PLAN_1:**
- PLAN_1: Need explicit src_id index to sample probes first
- PLAN_2: Probes are already the dataset elements

---

## 11. Advantages Summary

### vs PLAN_1 (5-minute chunks)

| Aspect | PLAN_1 | PLAN_2 |
|--------|--------|--------|
| **Padding waste** | 50-90% | <5% |
| **Storage size** | 50 GB | 7.5 GB |
| **Multi-scale learning** | No (fixed 5-min) | Yes (geometric strides) |
| **Data augmentation** | Limited | Infinite (random windows) |
| **Flexibility** | Fixed (must regenerate) | High (change strides at runtime) |
| **Preprocessing complexity** | High (split logic, part_id) | Low (group by probe) |
| **Training overhead** | ~0.5ms/example | ~3ms/example |

### vs Original PLAN_2 (Parquet runtime)

| Aspect | Parquet Runtime | ArrayRecord Runtime |
|--------|-----------------|---------------------|
| **Row access speed** | 50-100ms | 3-5ms |
| **GPU utilization** | 20-40% | 80-95% |
| **Tokens/sec (batch=256)** | 50-80k | 500k-1M |
| **Storage** | 2.1 GB | 7.5 GB |

**Trade-off:** 3.5x storage for 10-20x training speed = worth it!

---

## 12. Implementation Roadmap

### Phase 1: Preprocessing (1-2 days)
- [ ] Write `probe_level_preprocess.py`
- [ ] Handle train/test split by probe
- [ ] Serialize probe records with PyArrow IPC
- [ ] Write to sharded ArrayRecord files
- [ ] Validate: check row counts, sample probe inspection

### Phase 2: Grain Pipeline (1-2 days)
- [ ] Implement `ProbeDataSource` for ArrayRecord reading
- [ ] Implement `ProbeWindowSampler` with geometric stride sampling
- [ ] Implement timestamp masking (3 modes: full/none/mixed)
- [ ] Integrate with existing MaxText pipeline
- [ ] Validate: check token distributions, padding stats

### Phase 3: Testing & Optimization (1 day)
- [ ] Benchmark throughput (target: >500k tok/sec)
- [ ] Analyze padding distribution (target: <5% mean)
- [ ] Verify multi-scale coverage (check stride distribution)
- [ ] Profile worker CPU usage (should be <20% with 16 workers)

**Total: 4-5 days**

---

## 13. Open Questions

1. **Stride distribution:** Should very sparse strides (128+) be weighted differently?
2. **Probe size limits:** Should we split extremely dense probes (>100k measurements) across multiple rows?
3. **Timestamp masking probabilities:** Are 40/30/30 optimal, or should we tune empirically?
4. **Worker count:** What's the optimal num_workers for your hardware? (Start with 16-32)

---

## 14. Migration from PLAN_1

If you already have PLAN_1 ArrayRecords:

### Option A: Regenerate from source Parquet (recommended)
- Run new preprocessing script on original Parquet shards
- Cleaner, smaller storage footprint
- Takes 1-2 hours for typical dataset

### Option B: Convert PLAN_1 chunks to PLAN_2 format
- Read all chunks for each probe
- Detokenize back to measurements (if possible)
- Regroup and write probe-level records
- More complex, not recommended unless Parquet unavailable

---

## 15. Expected Results

### Training Performance
- **Throughput:** 500k-1M tokens/sec (batch=256, small model)
- **GPU utilization:** 80-95% (up from 20-40% with Parquet)
- **Data loading overhead:** <5% of step time

### Data Efficiency
- **Padding waste:** <5% mean (down from 50-90%)
- **Storage:** ~7.5 GB (down from 50+ GB for pre-tokenized chunks)
- **Effective data augmentation:** ~100x (each probe generates many windows)

### Model Quality
- **Multi-scale learning:** Model sees 1-second to 1-day gaps
- **Robust to timestamp sparsity:** Trained with 30% no-timestamp examples
- **Federated-ready:** Probe-centric by design

---

## 16. Conclusion

**PLAN_2 achieves the ideal balance:**
- Fast ArrayRecord random access (row-level reads)
- Runtime flexibility (dynamic windowing, infinite augmentation)
- Minimal storage (no pre-tokenization duplication)
- Multi-scale temporal learning (geometric stride sampling)
- Probe-centric design (federated learning compatible)

**The key insight:** Tokenization overhead (~3ms) is negligible compared to GPU compute (~20-100ms), so we optimize for storage efficiency and training flexibility rather than pre-computing everything.
