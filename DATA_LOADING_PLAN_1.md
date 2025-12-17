# RIPE Atlas Sequence Chunking & Training Pipeline Guide

## 1. Goal and Context

**Objective:**
Train a transformer-style language model over *tokenized network measurements* (RTT, failures, timestamps, IPs) so that, given a sequence of prior measurements, the model predicts the next token / partial measurement.

**Deployment constraint (clarified):**
The intended deployment is **end-user / probe-centric**. Each model instance runs locally on a device (a RIPE Atlas probe or similar end-user node), trains locally on its *own outgoing measurements*, and exchanges gradients with peers (federated / pairwise / gossip-style learning).

**Critical implication:**
At inference and local training time, a device only observes:

* its **own measurement history** (outgoing probes),
* to **many different destinations** (anchors and others),
* over time.

It does *not* observe:

* measurements originating from other probes,
* global network-wide traffic,
* full inbound measurement streams to a destination.

Therefore, the preprocessing and training data distribution must be **probe-centric** (per source device), not anchor-centric, and must avoid conditioning the model on context that would not exist locally.

---

## 2. Why We Do Sequence Chunking

Raw RIPE Atlas data is stored as **one row per measurement**:

* event_time
* src_addr
* dst_addr
* ip_version
* rtt

However, the model operates on **sequences** of measurements. If each training example were built by dynamically gathering hundreds of neighboring rows, training would be I/O-bound and unscalable.

**Sequence chunking** solves this by:

* grouping measurements into *pre-built, ordered sequences*,
* storing each sequence as a single record,
* enabling fast random access + random cropping during training.

Each chunk becomes the unit of shuffling, sampling, and parallelism.

---

## 3. Empirical Data Reality (Probe-Centric, Data-Driven Design)

We initially assumed that measurements might be extremely dense and that a small number of anchors dominated the data. DuckDB statistics over the full training set (~200M rows over ~30 days) corrected this assumption.

### Key findings from DuckDB stats:

* **Total rows:** ~200M
* **Distinct source addresses (`src_addr`):** ~45k (probes)
* **Distinct destinations (`dst_addr`):** ~130k (anchors + other targets)
* **IPv6 share:** ~42%
* **Failure rate:** ~17%

Importantly, the `src_addr` field corresponds to **probes (end-user devices)**, not anchors. Anchors appear primarily as **destinations**. The large cardinality and sparsity of `src_addr` is therefore expected and correct for a probe-centric view.

### Burstiness (top 200 busiest probes, sampled days):

* Median rate: **~1 measurement / second**
* p90: **~2 measurements / second**
* p99: **~3 measurements / second**
* Worst observed bursts: **~10 measurements / second**

**Conclusion:**
From the perspective of a single probe, the data stream is mostly sparse with occasional short bursts. This validates a probe-centric design and shows that aggressive downsampling is unnecessary for 1024-token contexts.

---

## 4. Token Economics (Why Bucket Sizes Matter)

Given the tokenization scheme:

* Typical IPv4 measurement: ~16 tokens
* Typical IPv6 measurement: ~40 tokens
* Weighted average: ~26–30 tokens / measurement

At 1–2 measurements/sec, this yields:

* ~30–60 tokens/sec
* ~1,800–3,600 tokens/min

Thus:

* A 1-minute window already contains multiple 1024-token training samples.
* A 5-minute window typically contains ~10k tokens.

---

## 5. Chosen Chunking Strategy (Probe-Centric)

### Chunking key

Measurements are grouped by:

* **source device** (`src_addr` or numeric `src_id`),
* fixed **time bucket**.

This ensures:

* each chunk corresponds to data a *single device could actually see*,
* perfect alignment with local training and inference,
* realistic client heterogeneity for federated-style optimization,
* no leakage of global or cross-device context.

Anchors naturally appear inside sequences as **destination tokens** (`DstIp`) and do not require special handling.

### Bucket size

**5-minute buckets per probe**

Rationale:

* Typical probe: ~75 measurements/min → ~375 measurements/5 min
* Tokenized size: ~10k–12k tokens (IPv4/IPv6 mix)
* Supports many random 1024-token crops with minimal I/O waste

### Record size cap

To guarantee predictable I/O and avoid pathological bursts:

* Impose a hard cap of **~50k–100k tokens per record**
* If a 5-minute bucket exceeds the cap, split it into `part_id = 0,1,…`

Splitting is rare but ensures worst-case safety.

---

## 6. Preprocessing Pipeline (Probe-Centric)

### Step 1: Partition and sort

For each training split (train/test):

1. Read Parquet rows (streaming / out-of-core)
2. Compute `bucket_start_time = floor(event_time / 300s)`
3. Partition by `(src_addr, bucket_start_time)`
4. Within each partition, sort rows by `event_time`

This yields time-ordered, per-probe buckets that match local device views.

### Step 2: Tokenization and chunk assembly

For each `(probe, bucket)` group:

* Tokenize measurements sequentially using the fixed schema
* Concatenate tokens into a single token array
* Record `meas_offsets`: token indices where `<MeasurementStart>` occurs

**Timestamp presence as a training variable (important):**

The tokenizer allows timestamps to be *optional* per measurement. This is intentional and should be preserved in preprocessing.

At **chunk creation time**, measurements should retain their true timestamps (absolute + deltas) so that the chunk encodes the full temporal information.

At **training time**, timestamps may be *selectively removed or masked* to simulate realistic partial observability:

* Some training windows include full timestamp information
* Some include timestamps only on a subset of measurements
* Some include no timestamps at all

This variability teaches the model to:

* use timestamps when available,
* fall back to implicit ordering when not,
* remain robust to missing or sparse timing metadata.

Importantly, timestamp removal is **not** a preprocessing-time operation; it is a **training-time augmentation** applied after random cropping.

**Chunk splitting and timestamp delta chaining (implementation detail):**

If a bucket is split into multiple records (`part_id = 0,1,…`) due to a size cap, delta timestamps must not reference a measurement in a previous part.

**Rule:** At every split boundary (start of each `part_id`), the first measurement must include an **absolute timestamp** (`<TimestampAbs>`). Subsequent measurements in that part may again use deltas.

This ensures each record is self-contained before any training-time timestamp masking is applied.

### Step 3: Persist to Parquet / ArrayRecord

Each row contains:

* `src_id` (probe identifier)
* `bucket_start_time`
* `bucket_duration_s = 300`
* `part_id`
* `tokens` (binary or uint16 array, with full timestamps present)
* `meas_offsets` (int32 array)

---

## 7. Parquet / ArrayRecord Layout

### Partitioning scheme

```
split=train/
  src_id=000123/
    date=2025-07-02/
      part-000.parquet
```

This layout:

* mirrors per-device data locality,
* enables massive parallel reads,
* allows client-aware or client-balanced sampling,
* cleanly supports federated-style training simulations.

### Record schema (one row = one chunk)

Each row should contain:

* `src_id` (probe identifier)
* `bucket_start_time`
* `bucket_duration_s = 300`
* `part_id` (0 unless split)
* `tokens` (binary blob or uint16 array)
* `meas_offsets` (int32 array)
* optional: `n_tokens`, `n_measurements` (for sampling/debug)

### Storage scale (rough estimate)

* ~200M measurements → ~5–6B tokens total
* 5-minute buckets → millions of probe-local chunks
* Average chunk size: ~10–20KB compressed

---

## 8. Training-Time Sampling Logic (Federated-Compatible)

### Dataset unit

One dataset element = **one probe-local chunk record**.

### Client-first sampling (required)

To approximate local training and to control skew:

1. Sample a **probe/client** first
2. Sample a chunk (time bucket / part) from that probe
3. Apply random cropping to obtain a candidate token window
4. Apply **timestamp masking / removal augmentation** (see below)
5. Emit one or more training examples
6. Mix updates across probes

### Timestamp masking / removal (training-time augmentation)

After cropping a token window (e.g., 1024 tokens), optionally modify timestamp fields:

Possible modes (sampled per window or per batch):

* **Full timestamps**: keep all timestamp tokens
* **Sparse timestamps**: keep timestamps only on a random subset of measurements
* **First-only timestamps**: keep timestamp only on the first measurement
* **No timestamps**: remove all timestamp tokens entirely

When timestamps are removed:

* measurement order remains explicit via `<MeasurementStart>` tokens
* the model must infer temporal structure implicitly from ordering and content

This augmentation should be stochastic and configurable (e.g., probability per mode). It should not affect `meas_offsets` alignment beyond removing timestamp subspans inside measurements.

### Training example generation

For each augmented window:

* input: tokens[0 : 1024]
* target: tokens[1 : 1025]

This yields a mixture of temporally explicit and temporally implicit contexts during training.

---

## 9. Why No Downsampling (Yet)

Given the **probe-centric burstiness statistics**:

* 5-minute probe-local buckets are already compact
* 1024-token windows are easily sampled without excessive I/O
* Downsampling would remove genuine local temporal structure

Downsampling should only be added if:

* context length increases substantially (>8k tokens), or
* probes begin producing sustained high-rate streams

---

## 10. If Long-Horizon Context Is Added Later

If longer temporal context is desired **per probe** (hours or days):

* Keep the 5-minute raw probe-local stream unchanged
* Add a separate *downsampled probe-local stream*
* Use **time-stratified random sampling per minute**:

  * keep ≤M measurements per minute
  * always keep failures and extreme RTTs

This preserves realism while controlling size.

---

## 11. Implementation Guidance (Preprocessing + Training)

### Preprocessing: what to use and why

**Fastest practical approach:** use a columnar engine that can read many Parquet shards efficiently and do out-of-core partitioning/sorting.

Recommended options:

* **DuckDB**: excellent for Parquet scans and can spill to disk; good when working from a single machine and you already used it for stats.
* **Spark / Beam**: better when you need distributed processing (multi-machine) or if the dataset grows beyond what a single machine can comfortably partition/sort.

**Which input to use (shards vs full Parquet):**

* Prefer the **existing sharded train/test Parquets**. They support parallel reads and keep train/test separation clean.
* Avoid rebuilding from the single monolithic Parquet unless you need to re-shard anyway.

**Conceptual implementation plan (no code):**

1. Read train shards
2. Compute `bucket_start_time` per row
3. Partition rows by `(src_id, bucket_start_time)` so each group is isolated
4. Sort each group by `event_time`
5. Tokenize sequentially; build `tokens` + `meas_offsets`
6. Enforce `max_tokens_per_record`; when exceeded, flush record with incremented `part_id`
7. At each flush boundary, reset timestamp encoding so the first measurement of the next part uses `<TimestampAbs>`
8. Write chunk records to Parquet (or directly to ArrayRecord)

### Training with Grain: which APIs are needed

The training pipeline should treat **one chunk record** as the dataset element and then generate 1024-token examples by random cropping.

Core Grain components:

* **ArrayRecordDataSource**: random access over chunk records (recommended for high-throughput shuffling)
* **MapDataset** transforms:

  * `.shuffle(...)` for global chunk-level shuffle
  * `.map(...)` to decode a record into `{tokens, meas_offsets, metadata}`
  * `.random_map(...)` (RandomMap) to implement random cropping and client-aware sampling logic
* **ReadOptions** to configure `num_threads` and `prefetch_buffer_size`
* Optional: **packing** utilities if you want to reduce padding waste with variable-length crops

**Client-first sampling in Grain (conceptual):**

* Build (or materialize) an index mapping `src_id -> list of record indices` (or `src_id -> list of files/row groups`)
* Implement a sampler that:

  1. samples a `src_id` (uniform or tempered)
  2. samples one record for that `src_id`
  3. applies random cropping to emit a 1024-token training example

If an explicit `src_id -> indices` mapping is too heavy to build in-memory, approximate client-first sampling by:

* writing chunk files partitioned by `src_id`, then
* sampling `src_id` by choosing a partition first, then a record inside it.

### Estimated engineering cost

| Stage                                                            | Estimate  |
| ---------------------------------------------------------------- | --------- |
| Update preprocessing to enforce split timestamp resets           | < 0.5 day |
| Implement chunk writer (partition/sort/tokenize/cap/split)       | 1–2 days  |
| Write/read Parquet (or ArrayRecord) + validate decode invariants | < 1 day   |
| Implement Grain input pipeline + client-first sampling           | 1–2 days  |

Total: **~3–5 days** for a robust, scalable probe-centric pipeline.

---

## 12. Summary

* Training must be **probe-centric** to match local/federated deployment.
* Use **client-first sampling** to avoid data skew and busy-probe dominance.
* Use **5-minute buckets** with a record cap; split rarely and reset timestamps at split boundaries.
* Store each chunk as a row with `tokens` + `meas_offsets` to enable fast random cropping to 1024-token examples.

This design is data-driven, scalable, and faithful to the intended local-training scenario.
