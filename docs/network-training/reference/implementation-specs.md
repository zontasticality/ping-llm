# PLAN_3 Detailed File Specifications

Technical specifications for each file change.

---

## 1. `src/MaxText/input_pipeline/_probe_chunk_datasource.py`

### New Class: `ProbeRowDataSource`

**Purpose:** Read big probe rows from ArrayRecord (PLAN_3 format)

**ArrayRecord Schema (what's stored on disk):**
```python
{
    'src_id': int64,                    # Probe identifier
    'measurements': binary,              # PyArrow IPC serialized RecordBatch
    'n_measurements': int32,             # Count
    'time_span_seconds': float64,       # last_time - first_time
    'first_timestamp': timestamp('us'), # For metadata/debugging
    'last_timestamp': timestamp('us'),  # For metadata/debugging
}
```

**Measurements RecordBatch Schema (inside binary blob):**
```python
pyarrow.schema([
    ('event_time', pa.timestamp('us')),
    ('dst_addr', pa.string()),
    ('ip_version', pa.int8()),
    ('rtt', pa.float32()),
])
```

**Class Interface:**
```python
class ProbeRowDataSource(grain.RandomAccessDataSource):
    def __init__(self, arrayrecord_path: str):
        # Read ArrayRecord, store length

    def __len__(self) -> int:
        # Return number of rows

    def __getitem__(self, index: int) -> dict:
        # Return:
        {
            'src_id': int,
            'measurements': List[dict],  # Deserialized
            'n_measurements': int,
            'metadata': {
                'time_span_seconds': float,
                'first_timestamp': timestamp,
                'last_timestamp': timestamp,
            }
        }

    def _read_row(self, index: int) -> dict:
        # Read from ArrayRecord, deserialize PyArrow IPC

    def _deserialize_measurements(self, measurements_bytes: bytes) -> List[dict]:
        # PyArrow IPC → list of dicts
        # Each dict: {event_time, dst_addr, ip_version, rtt}
```


### New Class: `ProbeRowSampler`

**Purpose:** Generate K training contexts per row with PLAN_3 sampling

**Class Interface:**
```python
class ProbeRowSampler(grain.RandomMapTransform):
    def __init__(
        self,
        crop_size: int = 1024,
        avg_tokens_per_measurement: int = 30,
        max_contexts_per_row: int = 16,
        mode_weights: tuple = (0.40, 0.30, 0.30),
        seed: Optional[int] = None,
    ):
        self.crop_size = crop_size
        self.avg_tokens_per_meas = avg_tokens_per_measurement
        self.max_K = max_contexts_per_row
        self.mode_weights = mode_weights
        self.seed = seed

    def random_map(self, row: dict, rng: random.Random) -> dict:
        """
        Generate ONE context from row.

        Note: Grain calls this K times per row via flatmap or repeat.
        Alternative: Use grain.FlatMap to yield K contexts per row.

        Flow:
        1. Calculate K = min(ceil(n_measurements / 30), 16)
        2. Choose path: small_row vs large_row
        3. Sample measurement buffer
        4. Choose timestamp mode
        5. Tokenize with randomization
        6. Pad and format for MaxText
        """
        measurements = row['measurements']
        n = len(measurements)

        # Calculate how many measurements we need for ~crop_size tokens
        target_measurements = self.crop_size // self.avg_tokens_per_meas

        # Branch: small row vs large row
        if n < target_measurements:
            # Small row: use all measurements
            meas_buffer = measurements
        else:
            # Large row: sample window
            meas_buffer = self._sample_large_row(
                measurements, target_measurements, rng
            )

        # Select timestamp mode
        mode = self._select_timestamp_mode(rng)

        # Tokenize measurements
        tokens = self._tokenize_measurements(
            meas_buffer, mode, rng
        )

        # Pad and format
        return self._format_output(tokens)

    def _sample_large_row(
        self, measurements: List[dict], target: int, rng: random.Random
    ) -> List[dict]:
        """
        Sample measurements from large row.

        1. Sample window_size ~ log-uniform[1, len(measurements)]
        2. Sample offset ~ uniform[0, len - window_size]
        3. Randomly pick items from [offset:offset+window_size]
        4. Sort selected items by timestamp
        5. Return enough to reach ~target count
        """
        n = len(measurements)

        # Log-uniform window size
        # log-uniform means: log(size) ~ uniform
        # Equivalent to: size ~ exp(uniform(log(1), log(n)))
        log_min = 0  # log(1) = 0
        log_max = np.log(n)
        log_size = rng.uniform(log_min, log_max)
        window_size = min(n, int(np.exp(log_size)))

        # Random offset
        if window_size >= n:
            offset = 0
        else:
            offset = rng.randint(0, n - window_size)

        # Select window
        window = measurements[offset:offset+window_size]

        # Randomly subsample from window
        # Keep sampling until we have enough tokens
        # (heuristic: target measurements, may need adjustment)
        if len(window) <= target:
            selected = window
        else:
            # Random sample without replacement
            selected = rng.sample(window, target)

        # Sort by timestamp
        selected.sort(key=lambda m: m['event_time'])

        return selected

    def _select_timestamp_mode(self, rng: random.Random) -> str:
        """Select timestamp mode: full, partial, or none."""
        r = rng.random()
        if r < self.mode_weights[0]:
            return 'full'
        elif r < self.mode_weights[0] + self.mode_weights[1]:
            return 'partial'
        else:
            return 'none'

    def _tokenize_measurements(
        self,
        measurements: List[dict],
        mode: str,
        rng: random.Random,
    ) -> List[int]:
        """
        Tokenize measurements with timestamp mode and field randomization.

        Per PLAN_3:
        - full: include all timestamps, delta encode
        - partial: extract 10-90% of measurements, remove their timestamps,
                   randomize order of non-timestamped measurements
        - none: remove all timestamps, randomize measurement order
        """
        from tokenization import encode_measurement

        tokens = []
        prev_timestamp = None

        if mode == 'full':
            # Include all timestamps
            for meas in measurements:
                meas_tokens = encode_measurement(
                    meas,
                    prev_timestamp=prev_timestamp,
                    randomize_field_order=True,
                    include_timestamp=True,
                )
                tokens.extend(meas_tokens)
                prev_timestamp = meas['event_time']

        elif mode == 'partial':
            # Extract random percentage
            extract_pct = rng.uniform(0.1, 0.9)
            n_extract = int(len(measurements) * extract_pct)

            # Random sample to extract
            extract_indices = set(rng.sample(
                range(len(measurements)), n_extract
            ))

            # Build two lists: timestamped and non-timestamped
            timestamped = []
            non_timestamped = []

            for i, meas in enumerate(measurements):
                if i in extract_indices:
                    non_timestamped.append(meas)
                else:
                    timestamped.append(meas)

            # Randomize non-timestamped
            rng.shuffle(non_timestamped)

            # Merge: timestamped stay in order, non-timestamped shuffled in
            # Strategy: interleave randomly
            all_meas = []
            ts_idx = 0
            nts_idx = 0

            for _ in range(len(measurements)):
                if ts_idx >= len(timestamped):
                    all_meas.append(('nts', non_timestamped[nts_idx]))
                    nts_idx += 1
                elif nts_idx >= len(non_timestamped):
                    all_meas.append(('ts', timestamped[ts_idx]))
                    ts_idx += 1
                else:
                    # Random choice
                    if rng.random() < 0.5:
                        all_meas.append(('ts', timestamped[ts_idx]))
                        ts_idx += 1
                    else:
                        all_meas.append(('nts', non_timestamped[nts_idx]))
                        nts_idx += 1

            # Tokenize
            for typ, meas in all_meas:
                include_ts = (typ == 'ts')
                meas_tokens = encode_measurement(
                    meas,
                    prev_timestamp=prev_timestamp if include_ts else None,
                    randomize_field_order=True,
                    include_timestamp=include_ts,
                )
                tokens.extend(meas_tokens)
                if include_ts:
                    prev_timestamp = meas['event_time']

        else:  # mode == 'none'
            # No timestamps, randomize order
            shuffled = measurements.copy()
            rng.shuffle(shuffled)

            for meas in shuffled:
                meas_tokens = encode_measurement(
                    meas,
                    prev_timestamp=None,
                    randomize_field_order=True,
                    include_timestamp=False,
                )
                tokens.extend(meas_tokens)

        return tokens

    def _format_output(self, tokens: List[int]) -> dict:
        """Pad to crop_size and format for MaxText."""
        tokens = np.array(tokens, dtype=np.int32)

        # Truncate if too long
        if len(tokens) > self.crop_size:
            tokens = tokens[:self.crop_size]

        original_length = len(tokens)

        # Pad if too short
        if len(tokens) < self.crop_size:
            padding = np.zeros(
                self.crop_size - len(tokens), dtype=np.int32
            )
            tokens = np.concatenate([tokens, padding])

        # Segmentation mask
        segmentation = np.ones(self.crop_size, dtype=np.int32)
        segmentation[original_length:] = 0

        # Position IDs
        positions = np.arange(self.crop_size, dtype=np.int32)

        return {
            "inputs": tokens,
            "inputs_segmentation": segmentation,
            "inputs_position": positions,
            "targets": tokens,
            "targets_segmentation": segmentation,
            "targets_position": positions,
        }
```

**Critical Design Decision: K contexts per row**

Option A: Sample 1 context per random_map call, let Grain handle K
```python
# In pipeline construction:
dataset = dataset.random_map(sampler, seed=seed)
# Grain calls random_map once per row
# Problem: Only get 1 context per row!
```

Option B: Use FlatMap to yield K contexts
```python
class ProbeRowSampler(grain.RandomMapTransform):
    def random_map(self, row, rng):
        # Return list of K contexts
        K = min(ceil(row['n_measurements'] / 30), 16)
        contexts = []
        for _ in range(K):
            context = self._generate_one_context(row, rng)
            contexts.append(context)
        return contexts

# Then:
dataset = dataset.random_map(sampler)
dataset = dataset.flat_map(lambda x: x)  # Flatten list of contexts
```

Option C: Custom Iterator (cleanest for PLAN_3)
```python
# Wrap dataset to repeat each row K times
class MultiContextDataset(grain.IterDataset):
    def __init__(self, source_dataset, sampler):
        self.source = source_dataset
        self.sampler = sampler

    def __iter__(self):
        for row in self.source:
            K = min(ceil(row['n_measurements'] / 30), 16)
            for _ in range(K):
                yield self.sampler.sample_one(row)
```

**Recommendation: Use Option B (FlatMap)** - most compatible with Grain API.

---

## 2. `scripts/data/create_probe_rows.py`

### Purpose
Convert sharded Parquet → PLAN_3 ArrayRecord (big rows)

### High-Level Algorithm

```python
def main():
    # 1. Read all parquet files
    con = duckdb.connect()
    con.execute("""
        CREATE VIEW all_measurements AS
        SELECT * FROM read_parquet('data/sharded/**/*.parquet')
    """)

    # 2. Get list of unique src_addr (for train/test split)
    src_addrs = con.execute("""
        SELECT DISTINCT src_addr FROM all_measurements
        ORDER BY src_addr
    """).fetchall()

    # 3. Split into train/test by probe (90/10)
    n_train = int(len(src_addrs) * 0.9)
    train_addrs = set(src_addrs[:n_train])

    # 4. Process each probe
    #    (Can parallelize across workers)
    train_writer = ArrayRecordWriter('data/probe_rows/train.arrayrecord')
    test_writer = ArrayRecordWriter('data/probe_rows/test.arrayrecord')

    for src_addr in src_addrs:
        writer = train_writer if src_addr in train_addrs else test_writer

        # Fetch probe's measurements
        measurements = con.execute("""
            SELECT event_time, dst_addr, ip_version, rtt
            FROM all_measurements
            WHERE src_addr = ?
            ORDER BY event_time
        """, [src_addr]).fetchall()

        # Write to ArrayRecord (with 8MB splitting)
        write_probe_to_arrayrecord(
            writer, src_addr, measurements, max_size_mb=8
        )

    train_writer.close()
    test_writer.close()


def write_probe_to_arrayrecord(
    writer, src_addr, measurements, max_size_mb=8
):
    """Write probe measurements, splitting at max_size_mb."""
    max_bytes = max_size_mb * 1024 * 1024

    # Convert measurements to PyArrow Table
    table = pa.Table.from_pylist([
        {
            'event_time': m[0],
            'dst_addr': m[1],
            'ip_version': m[2],
            'rtt': m[3],
        }
        for m in measurements
    ])

    # Serialize to IPC and check size
    buffer = serialize_table_to_ipc(table)

    if len(buffer) <= max_bytes:
        # Write single row
        record = create_record(src_addr, buffer, len(measurements), ...)
        writer.write(record)
    else:
        # Split into multiple rows
        split_and_write(writer, src_addr, table, max_bytes)


def serialize_table_to_ipc(table: pa.Table) -> bytes:
    """Serialize PyArrow Table to IPC format."""
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


def create_record(
    src_id: int,
    measurements_bytes: bytes,
    n_measurements: int,
    first_ts,
    last_ts,
) -> bytes:
    """Create ArrayRecord entry (single-row RecordBatch)."""
    record_dict = {
        'src_id': [src_id],
        'measurements': [measurements_bytes],
        'n_measurements': [n_measurements],
        'time_span_seconds': [(last_ts - first_ts).total_seconds()],
        'first_timestamp': [first_ts],
        'last_timestamp': [last_ts],
    }

    batch = pa.RecordBatch.from_pydict(record_dict)

    # Serialize to IPC
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()

    return sink.getvalue().to_pybytes()
```

### CLI Interface
```bash
python scripts/data/create_probe_rows.py \
  --input "data/sharded/train/*.parquet" \
  --output data/probe_rows \
  --max-row-size-mb 8 \
  --train-ratio 0.9
```

---

## 3. Minor Updates

### `src/MaxText/input_pipeline/probe_chunk_pipeline.py`
```python
# Update line 44:
-    Construct the probe-chunk Grain pipeline (DATA_LOADING_PLAN_1).
+    Construct the probe-row Grain pipeline (DATA_LOADING_PLAN_3).

# Update imports if classes renamed
```

### `src/MaxText/input_pipeline/_network_grain_integration.py`
```python
# Delete lines 31-128 (PLAN_2 function)

# Update line 145:
-    Create probe-centric chunk dataset (DATA_LOADING_PLAN_1).
+    Create probe-centric row dataset (DATA_LOADING_PLAN_3).

# Update line 147-151 description:
    This is the RECOMMENDED approach for production training. It provides:
-    - 50-100x faster I/O (one chunk read vs 64 measurements)
-    - Pre-tokenized data (removes tokenization from training hot path)
+    - Minimal padding (<5% vs 50-90%)
+    - Multi-scale temporal learning (log-uniform window sampling)
+    - Runtime tokenization with data augmentation
     - Probe-centric design (perfect for federated/decentralized deployment)
     - ArrayRecord format (optimized for random access)

# Update line 181-183 log messages:
-    max_logging.log(f"[DATA_LOADING_PLAN_1] ...")
+    max_logging.log(f"[DATA_LOADING_PLAN_3] ...")
```

---

## Testing Checklist

After implementation:

1. **Preprocessing validation:**
   - [ ] Run on small Parquet sample (100 probes)
   - [ ] Verify row count matches probe count (or splits if >8MB)
   - [ ] Inspect first row: deserialize measurements, check schema
   - [ ] Check row sizes (should be <8MB each)

2. **Data loading validation:**
   - [ ] Load single row, verify measurements deserialized correctly
   - [ ] Sample one context, verify output shape (1024,)
   - [ ] Sample K contexts from one row, verify K calculation
   - [ ] Check timestamp mode distribution over 1000 samples (~40/30/30)

3. **Padding analysis:**
   - [ ] Run analyze_padding.py on PLAN_3 data
   - [ ] Verify mean padding <5% (target from plan)

4. **Integration test:**
   - [ ] Create small dataset (1000 rows)
   - [ ] Load in MaxText training loop
   - [ ] Train for 10 steps
   - [ ] Verify loss is finite, gradients update

5. **Performance test:**
   - [ ] Measure tokens/sec throughput
   - [ ] Compare to PLAN_1 baseline
   - [ ] Profile bottlenecks (tokenization, I/O, etc.)
