# PLAN

This document sketches the plan for building and training a MaxText-based model on latency/IP measurements stored in Parquet, using a custom byte + role-token scheme. It is meant to be iterated on as the project evolves.

---

## 1. Goals

1. Learn a generative model over network measurements, where each measurement consists of:

   * Source IP (IPv4/IPv6)
   * Destination IP (IPv4/IPv6)
   * Latency statistics (avg RTT, individual RTTs)
   * Throughput and packet counts
   * Timestamp and probe ID
2. Represent each measurement as a sequence of discrete tokens:

   * Role / delimiter tokens (e.g., `MeasurementStart`, `SrcIpStart`, `DestIpStart`, `LatencyStart`, `TimestampStart`, etc.)
   * Unified byte tokens `Byte0` .. `Byte255` covering IPs, timestamps, latencies, and other numeric fields.
3. Randomize the order of field blocks within a measurement so the model must learn the joint distribution, not just fixed positions.
4. Train an efficient decoder-only Transformer using MaxText + Grain (Parquet input).
5. Eventually support:

   * Partial IP conditioning (e.g., XXX.YYY.ZZZ.???)
   * Distributional queries (e.g., distribution over latency for given IP ranges).

---

## 2. Data & Tokenization

### 2.1 Parquet schema (actual)

**Current dataset:** `data/training_data.parquet`
- **Size:** 1.1 GB, 100,005,691 rows
- **Date range:** 2025-06-24 → 2025-07-22 (28 days)
- **IP distribution:** IPv4 57.7M (57.7%) / IPv6 42.3M (42.3%)
- **Measurements:** 66,275 distinct msm_id

Columns (per measurement row):

* `msm_id: int64`           — Measurement identifier
* `event_time: timestamp`   — Event timestamp with timezone
* `src_addr: string`        — Source IP address (text format, e.g., "192.0.2.1" or "2001:db8::1")
* `dst_addr: string`        — Destination IP address (text format)
* `ip_version: int64`       — IP version (4 or 6)
* `rtt: double`             — Round-trip time in milliseconds
* `size: int64`             — Packet size in bytes
* `packet_error_count: int64` — Number of packet errors

**Value ranges (observed):**
* `rtt`: min -1.0 (sentinel for failed probes), max 302,281.66, p50 50ms, p90 239ms, p99 343ms
* `size`: mostly 64 bytes, max 2000
* `packet_error_count`: mostly 0, max 16

**Note:** IPs are stored as strings and must be parsed to bytes during tokenization using `ip_version`.

### 2.2 Token schema

Define a small, fixed vocabulary:

* Special / role tokens (enum-style):

  * `MeasurementStart`      — Start of a measurement
  * `SrcIpStart`            — Start of source IP block
  * `DestIpStart`           — Start of destination IP block
  * `Ipv4Start`             — IPv4 address follows (4 bytes)
  * `Ipv6Start`             — IPv6 address follows (16 bytes)
  * `RttStart`              — RTT value follows (8 bytes, float64)
  * `SizeStart`             — Packet size follows (2 bytes, uint16)
  * `ErrorCountStart`       — Packet error count follows (1 byte, uint8)
  * `TimestampStart`        — Event timestamp follows (8 bytes, int64)
  * `MsmIdStart`            — Measurement ID follows (8 bytes, int64)

* Byte tokens:

  * `Byte0`, `Byte1`, ..., `Byte255`

Proposed ID mapping:

* IDs 0–9: role tokens (10 total)
* IDs 10–265: byte tokens (256 total)
* `vocab_size = 266`

### 2.3 Row → token sequence

For each row:

1. **Parse and encode IP addresses from strings**

   * Use `ip_version` to determine IPv4 or IPv6
   * Parse `src_addr` and `dst_addr` strings to bytes:
     * IPv4: parse to 4 bytes (e.g., "192.0.2.1" → `[192, 0, 2, 1]`)
     * IPv6: parse to 16 bytes (e.g., "2001:db8::1" → expand and convert)
   * Encode as:
     * Source: `SrcIpStart` + `Ipv4Start`/`Ipv6Start` + byte tokens
     * Destination: `DestIpStart` + `Ipv4Start`/`Ipv6Start` + byte tokens

2. **Encode RTT (round-trip time)**

   * **Handling sentinel values:** `rtt = -1.0` indicates failed probe
   * **Clipping outliers:** Observed max is 302,281ms; consider clipping at p99.9 (~500-1000ms) or using log-scale encoding
   * Options:
     * **Raw IEEE 754:** `RttStart` + 8 bytes from `rtt: f64` (big-endian)
     * **Fixed-point:** Convert to microseconds, encode as uint64 (better for learning?)
   * **Decision needed:** Raw vs. fixed-point vs. log-scale (see section 8)

3. **Encode packet size**

   * Most values are 64 bytes; max observed is 2000
   * `SizeStart` + 2 bytes as `uint16` (big-endian)
   * Values >65535 are not expected based on observations

4. **Encode packet error count**

   * Observed range: 0–16
   * `ErrorCountStart` + 1 byte as `uint8`

5. **Encode timestamp**

   * Convert `event_time` timestamp to Unix epoch (int64, seconds or milliseconds)
   * `TimestampStart` + 8 bytes (big-endian)

6. **Encode measurement ID**

   * `MsmIdStart` + 8 bytes from `msm_id: i64` (big-endian)

7. **Build field blocks**

   * `SrcIpStart` + IP encoding
   * `DestIpStart` + IP encoding
   * `RttStart` + RTT encoding
   * `SizeStart` + size encoding
   * `ErrorCountStart` + error encoding
   * `TimestampStart` + timestamp encoding
   * `MsmIdStart` + measurement ID encoding

8. **Randomize field block order per measurement**

   * Use a deterministic RNG seeded from `(msm_id, event_time)` so randomization is reproducible
   * This ensures the model learns the joint distribution, not positional patterns

9. **Final measurement sequence**

   * `[MeasurementStart] + shuffled_blocks`
   * Approximate token count per measurement:
     * IPv4 src + dst: ~12 tokens (2 role + 2×4 bytes + 2 IP-type markers)
     * IPv6 src + dst: ~36 tokens (2 role + 2×16 bytes + 2 IP-type markers)
     * Other fields: ~30 tokens (6 role tokens + 8+2+1+8+8 = 27 bytes)
     * **Total per measurement:** ~42–66 tokens depending on IP version

### 2.4 Pre-tokenized vs on-the-fly

Two options:

1. **On-the-fly tokenization** during Grain loading (simpler to start):

   * Grain reads raw Parquet rows.
   * A custom map transform applies `encode_measurement(row)` to produce `tokens`.

2. **Pre-tokenized Parquet** (better throughput later):

   * Preprocessing job creates a derived Parquet dataset with an additional column:

     * `tokens: list<uint16 or uint32>`
     * `length: uint16` (optional)
   * Grain reads the `tokens` column directly.

Plan: **start on-the-fly**, switch to pre-tokenized when profiling shows CPU bottlenecks.

---

## 3. Grain + Parquet Integration

### 3.1 Local Grain smoke test (laptop)

1. Install `grain` and its Parquet dependencies.
2. Write a small script `local_grain_smoke.py` that:

   * Opens a single Parquet file via `ParquetIterDataset`.
   * Applies a `Map` transform that calls `encode_measurement(row)`.
   * Prints a few tokenized examples for sanity.

Pseudo-code outline:

```python
import grain

from tokenization import encode_measurement

class EncodeRow(grain.transforms.Map):
    def map(self, element):
        tokens = encode_measurement(element)
        return {"tokens": tokens, "length": len(tokens)}

if __name__ == "__main__":
    ds = grain.experimental.ParquetIterDataset("./sample_measurements.parquet")
    ds = ds.map(EncodeRow())

    it = iter(ds)
    for _ in range(5):
        print(next(it))
```

3. Confirm:

   * No schema errors.
   * Token IDs are within `[0, vocab_size)`.
   * Field randomization behaves as expected.

### 3.2 Dataset sharding strategy

**Current state:** Single file `data/training_data.parquet` with 100M rows.

**Action needed:** Shard into multiple files for better shuffling and parallelism:

* **Target:** 100–500 files (~200k–1M rows/file)
* **Benefits:**
  * Better shuffle quality during training
  * Improved I/O parallelism with Grain workers
  * Easier train/eval split management
* **Implementation:**
  * Create `scripts/shard_parquet.py` to split the single file
  * Stratify by time and/or msm_id to ensure balanced shards
  * Consider 80/10/10 split for train/val/test
* **File organization:**
  ```
  data/
    training_data.parquet          # original 100M rows
    sharded/
      train/
        shard_0000.parquet
        shard_0001.parquet
        ...
      val/
        shard_0000.parquet
        ...
      test/
        shard_0000.parquet
        ...
  ```

**Note:** Until sharding is implemented, local Grain testing can use the single file with limited workers.

### 3.3 Packing measurements into training sequences

* Each measurement is relatively short (dozens of tokens).
* Plan to **pack multiple measurements per training sequence** to improve utilization.
* Options:

  * Simple concatenation with `MeasurementStart` tokens as natural boundaries.
  * Later: explore smarter packing/length-based bucketing.

---

## 4. MaxText Integration

### 4.1 Repository layout

Proposed structure:

```text
project_root/
  PLAN.md               # this document
  tokenization.py       # encode_measurement and helpers
  data/
    raw_parquet/        # original measurements
    tokenized_parquet/  # optional derived dataset later
  maxtext/
    # MaxText clone as submodule or subtree
    src/MaxText/configs/
      latency_parquet.yml
      latency_model_100m.yml
  scripts/
    local_grain_smoke.py
    pretokenize_to_parquet.py
    slurm_train_maxtext.sh
```

### 4.2 MaxText configs

Two config layers:

1. **Model config** (`latency_model_100m.yml`)

   * Define a small-ish decoder-only model (~100M parameters):

     * `vocab_size: 266` (10 role tokens + 256 byte tokens)
     * `num_decoder_layers: 12`
     * `emb_dim: 768`
     * `num_query_heads: 12` (head_dim 64)
     * `mlp_dim: 3072`
     * `max_length`: TBD (based on how many measurements to pack per sequence)
2. **Run/config overlay** (`latency_parquet.yml`)

   * Data and runtime specifics:

```yaml
# latency_parquet.yml

# Data
dataset_type: grain
grain_file_type: parquet
grain_train_files: "data/sharded/train/*.parquet"
grain_eval_files: "data/sharded/val/*.parquet"
grain_worker_count: 4

# Model / vocab
vocab_size: 266

# Training
steps: 200000        # placeholder, to be tuned
eval_interval: 0     # set >0 when eval dataset ready
enable_checkpointing: true

# Hardware defaults can be overridden via CLI (cpu/gpu)
```

### 4.3 Local MaxText smoke test

On laptop (CPU):

```bash
export DECOUPLE_GCLOUD=TRUE   # avoid GCP deps during local dev

python -m MaxText.train maxtext/src/MaxText/configs/latency_parquet.yml \
  run_name=local_smoke \
  base_output_directory=/tmp/maxtext_out \
  hardware=cpu \
  per_device_batch_size=1 \
  steps=20
```

Check that:

* Training loop starts.
* Grain loader is invoked without errors.
* Loss is finite and decreasing over a few steps.

---

## 5. SLURM Integration

### 5.1 Single-node multi-GPU job script

Create `scripts/slurm_train_maxtext.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=latency-maxtext
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load cuda            # if your cluster uses modules
source ~/venvs/maxtext/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

DATA_DIR=/scratch/$USER/latency_parquet
OUT_DIR=/scratch/$USER/maxtext_runs

python -m MaxText.train maxtext/src/MaxText/configs/latency_parquet.yml \
  run_name=slurm_latency_${SLURM_JOB_ID} \
  base_output_directory=$OUT_DIR \
  hardware=gpu \
  per_device_batch_size=4 \
  grain_train_files="$DATA_DIR/train/*.parquet" \
  grain_eval_files="$DATA_DIR/eval/*.parquet" \
  grain_worker_count=8
```

Plan:

* Start with 1–4 GPUs on a single node.
* Only consider multi-node once single-node jobs are stable and fast.

### 5.2 Git-based deployment

Workflow:

1. Develop and test tokenization + Grain code locally.
2. Commit to Git.
3. Push to remote accessible by SLURM cluster.
4. On the cluster, pull the latest commit and submit `slurm_train_maxtext.sh`.

---

## 6. Monitoring & Experiment Tracking

### 6.1 TensorBoard (default)

* Use MaxText’s built-in TensorBoard logging.
* Point TensorBoard at `base_output_directory` for live monitoring.

### 6.2 Optional: Weights & Biases integration

Two options:

1. **Sync TensorBoard logs to W&B** after or during runs:

   * `wandb sync /path/to/tensorboard/logdir`
2. **Add a small `wandb.init(...)` hook** in the MaxText training entrypoint with `sync_tensorboard=True` and control it via env vars (`ENABLE_WANDB`, `WANDB_PROJECT`, etc.).

Plan: start with TensorBoard; add W&B once the training job is stable.

---

## 7. Iteration Plan

1. **Phase 0 — Missing scaffolding** ⚠️ *Prerequisites before Phase 1*

   * **Create `tokenization.py`:**
     * Implement IP string parsing (IPv4/IPv6) to bytes
     * Implement `encode_measurement(row)` with all field encodings
     * Add deterministic RNG for field block shuffling
     * Define token ID mappings (role tokens + byte tokens)
   * **Create `scripts/shard_parquet.py`:**
     * Split `data/training_data.parquet` into train/val/test shards
     * Implement stratified splitting by time and/or msm_id
   * **Create Grain smoke test script (`scripts/local_grain_smoke.py`):**
     * Test Parquet loading with actual schema
     * Validate tokenization output on sample rows
   * **Create MaxText config files:**
     * `maxtext/src/MaxText/configs/latency_model_100m.yml`
     * `maxtext/src/MaxText/configs/latency_parquet.yml`
   * **Fix `duckdb_sample.py`:**
     * Update date range expectations (2025-06-24 → 2025-07-22)
     * Remove references to Nov 10/22 dates

2. **Phase 1 — Data & tokenization validation**

   * Run `local_grain_smoke.py` with actual data
   * Validate token ID ranges: all in [0, 266)
   * Verify IP parsing correctness (spot-check IPv4/IPv6)
   * Test randomized field order reproducibility
   * Validate RTT encoding handles -1.0 sentinel
   * Check vocab bounds on observed value ranges

3. **Phase 2 — Local MaxText sanity**

   * Minimal CPU run with tiny model and few steps
   * Confirm no shape or vocab issues
   * Verify Grain integration works end-to-end

4. **Phase 3 — SLURM single-node training**

   * Run on 1–4 GPUs, moderate batch size
   * Tune `grain_worker_count`, batch size, and learning rate
   * Monitor throughput and GPU utilization

5. **Phase 4 — Pre-tokenization & optimization**

   * Add `scripts/pretokenize_to_parquet.py` to create `tokens` column
   * Point Grain to pre-tokenized dataset
   * Re-profile throughput and training speed

6. **Phase 5 — Modeling improvements**

   * Adjust model size (20M → 100M → 300M) based on dataset size and validation metrics
   * Experiment with RTT encoding variations (raw/fixed-point/log-scale)
   * Explore partial-IP conditioning and query mechanisms

7. **Phase 6 — Inference and downstream usage**

   * Export trained checkpoint
   * Build small inference scripts for:
     * Density estimation / likelihood evaluation
     * Sampling latencies given partial IPs
   * Explore on-device deployment for smaller distilled models

---

## 8. Open Questions / TODOs

### 8.1 RTT encoding strategy

**Critical decision:** How to encode RTT values with observed range [-1.0, 302,281.66]?

* **Option A: Raw IEEE 754 float64**
  * Pro: Simple, no information loss
  * Con: Model must learn floating-point bit patterns; extreme outliers (302k ms) may hurt learning
* **Option B: Fixed-point (microseconds as uint64)**
  * Pro: More interpretable for model; natural integer encoding
  * Con: Still very wide range (requires 8 bytes); outliers remain
* **Option C: Log-scale encoding**
  * Pro: Compresses range; aligns with perceptual differences in latency
  * Con: More complex; requires special handling of -1.0 sentinel
* **Option D: Clipped + fixed-point**
  * Pro: Clip at p99.9 (~500-1000ms), encode failed probes as special value
  * Con: Loses information about extreme outliers

**Action:** Prototype all options in Phase 1; measure token distribution and preliminary loss.

### 8.2 Timestamp encoding

* Convert `event_time` to Unix epoch: seconds or milliseconds?
* Consider: Does the model benefit from sub-second precision? (Likely no)
* **Recommendation:** Seconds since epoch (int64) for simplicity

### 8.3 Packing strategy

* Each measurement: ~42–66 tokens (IPv4 vs IPv6)
* Target sequence length: 512? 1024? 2048?
* How many measurements to pack per sequence?
  * 512 tokens → ~8–12 measurements
  * 1024 tokens → ~16–24 measurements
* Consider: Benefits of longer context vs. training efficiency

### 8.4 Model size targets

* Dataset: 100M rows × ~50 avg tokens/row = ~5B tokens
* Industry rule-of-thumb: ~10–20 tokens per parameter for good performance
* Suggested models:
  * **Small:** 20M params (~200M tokens)
  * **Medium:** 100M params (~1–2B tokens)
  * **Large:** 300M params (~3–6B tokens, may require multiple epochs)

### 8.5 Evaluation metrics

* **NLL (negative log-likelihood):** Standard autoregressive loss
* **Field-specific accuracy:**
  * RTT prediction: MAE, MSE, calibration
  * IP prediction: Exact match for byte sequences
* **Conditional queries:**
  * P(RTT | src_ip, dst_ip): Does the model learn geographic/network patterns?
  * P(dst_ip | src_ip, RTT): Can it predict likely destinations?

### 8.6 When to pre-tokenize?

* **Start:** On-the-fly tokenization for Phase 1–2 (flexibility during experimentation)
* **Switch to pre-tokenization:** After RTT encoding and packing strategy are finalized (Phase 4)
* **Benefit:** 5–10× faster data loading once tokenization stabilizes

This PLAN.md should be updated as implementation details and design decisions become clearer.
