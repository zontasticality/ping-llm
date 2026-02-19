# PLAN

Active project plan for ping-llm: training a decoder-only Transformer on RIPE Atlas network latency measurements using MaxText.

---

## Project Goal

Train a generative model over network measurements that learns the joint distribution of:
- Source/destination IP addresses (IPv4/IPv6)
- Round-trip times (RTT)
- Temporal patterns (timestamps, delta encoding)

The model should support:
- Conditional generation (predict RTT given IP pair, predict likely destinations given source)
- Distributional queries (latency distributions for IP ranges)
- Partial-IP conditioning (e.g., subnet-level predictions)

---

## Current State

### What Works
- **Data pipeline** (PLAN_3): Probe-centric big-row ArrayRecord format with runtime tokenization
  - `data/parquet_ping/*.parquet` → `data/probe_rows/{train,test}.arrayrecord`
  - Multi-scale temporal sampling (log-uniform window sizes)
  - 3 timestamp modes (full/partial/none) for data augmentation
  - <5% padding waste
- **MaxText integration**: Custom `network` dataset backend with ~20 lines across 3 upstream files
- **Training**: Runs on Modal (A100/B200) and locally (CPU)
- **Evaluation scripts**: `eval_paper_metrics.py`, `eval_next_token_predictions.py`, `eval_ordering_likelihood.py`, `eval_live_ping.py`
- **Checkpointing**: Auto-save every 200 steps + on interrupt

### Architecture
- **Model**: ~95M param decoder-only Transformer
  - 267 vocab (11 role tokens + 256 byte tokens)
  - 20 layers, 640 emb dim, 10 heads, 2048 MLP dim
  - RoPE positional encoding, flash attention
  - 1024 max sequence length
- **Tokenization**: Custom byte-level scheme
  - Role tokens: MEASUREMENT_START, SRC_IPV4/IPV6, DST_IPV4/IPV6, TIMESTAMP_ABS/DELTA1/DELTA4, RTT_START, FAILED
  - RTT: 5-bit exponent + 11-bit mantissa (< 0.1% relative error)
  - Timestamps: Delta-encoded (95%+ fit in 1 byte)
  - Field order randomization for joint distribution learning

### Key Files
```
src/MaxText/configs/latency_network.yml          # Training config
src/MaxText/input_pipeline/
  _network_data_processing.py                     # Backend interface (make_network_{train,eval}_iterator)
  probe_chunk_pipeline.py                         # Dataset builder (grain pipeline construction)
  _probe_chunk_datasource.py                      # Core: ProbeRowDataSource + ProbeRowSampler
  network_tokenization.py                         # Tokenization (encode_measurement, IP/RTT encoding)
src/MaxText/input_pipeline/input_pipeline_interface.py  # Backend registration (+3 lines)
src/MaxText/configs/types.py                      # Config types (+12 lines: NetworkDataset)
src/MaxText/train.py                              # Training loop (+25 lines: interrupt handler, eval fix)
```

### Upstream Divergence (as of 2026-02-16)
- Fork point: `7ebcc9a39` (PR #2783)
- **450 commits behind** upstream `google/maxtext`
- Upstream has **restructured** `input_pipeline/` and **deleted** `configs/types.py`
- Our modifications are minimal but will require manual re-integration after upstream sync

---

## Design Decisions (Rationale)

### Why probe-centric big rows (not per-measurement rows)?
Per-measurement rows caused 50-90% padding waste because measurements tokenize to 14-47 tokens but sequences are 1024. Big rows group all measurements from one source IP, allowing the sampler to fill sequences efficiently with <5% waste.

### Why shift-before-batch (not grain's shift-after-batch)?
The network pipeline uses `FlatMapTransform` to generate K contexts per row. Each context is independently padded and shifted inside the sampler. This is architecturally incompatible with grain's `FirstFitPackIterDataset` (which expects unshifted variable-length sequences), but simpler for our use case where the sampler already fills sequences to near-capacity.

### Why `packing: False`?
The network backend already minimizes padding via windowed sampling. Grain's packing requires variable-length unshifted inputs, which our pipeline doesn't produce. Setting `packing: True` was a no-op (the flag was silently ignored since the pipeline didn't implement it).

---

## Known Issues

1. **Upstream sync needed**: 450 commits behind; `types.py` deleted upstream, `input_pipeline` restructured
2. **No multi-host data sharding**: All hosts read identical data (fine for single-GPU, blocks scaling)
3. **Eval iterator resets each interval**: `for` loop calls `__iter__()` → grain worker pool restart overhead
4. **Eval always repeats infinitely**: No option for finite eval passes; relies on `eval_steps` limit
5. **TensorFlow still required**: `multihost_dataloading.py` imports TF unconditionally
6. **Scripts have stale imports**: `verify_tokenization.py`, `smoke_test_maxtext.py`, `test_tokenization_standalone.py` import from old `tokenization` module path

---

## Performance Targets

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| MFU | 30% | ~30% | 40-50% |
| TFLOPS | 94 | ~94 | 125-156 |
| Tokens/s/device | 180k | ~180k | 250-320k |
| Training time (10k steps) | ~1.1h | ~1.1h | ~0.7h |
| Padding waste | 50-90% | <5% | <5% |
