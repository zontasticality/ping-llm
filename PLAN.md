# PLAN

Active project plan for ping-llm: training a decoder-only Transformer on RIPE Atlas network latency measurements using PyTorch.

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

### Stack
- **Framework**: Pure PyTorch (migrated from MaxText/JAX in Feb 2026)
- **Optimizer**: Muon (weight matrices) + AdamW (embeddings) — nanochat pattern
- **LR schedule**: Warmup-Stable-Decay (WSD)
- **Deployment**: Modal (A100 GPU), SLURM (multi-GPU)
- **Data**: grain pipeline with ArrayRecord, probe-centric big-row format

### Architecture
- **Model**: ~95M param decoder-only Transformer (`src/ping_llm/model.py`)
  - 267 vocab (11 role tokens + 256 byte tokens)
  - 24 layers, 1536 emb dim, 12 heads, 128 head dim
  - RoPE positional encoding, logit softcap at 15
  - ReLU² activation, parameterless RMSNorm
  - 1024 max sequence length
- **Tokenization**: Custom byte-level scheme (`src/ping_llm/data/tokenization.py`)
  - Role tokens: MEASUREMENT_START, SRC_IPV4/IPV6, DST_IPV4/IPV6, TIMESTAMP_ABS/DELTA1/DELTA4, RTT_START, FAILED
  - RTT: 5-bit exponent + 11-bit mantissa (< 0.1% relative error)
  - Timestamps: Delta-encoded (95%+ fit in 1 byte)
  - Field order randomization for joint distribution learning

### What Works
- **Data pipeline**: Probe-centric big-row ArrayRecord format with runtime tokenization
  - Multi-scale temporal sampling (log-uniform window sizes)
  - 3 timestamp modes (full/partial/none) for data augmentation
  - <5% padding waste
  - Sharded ArrayRecord support (4 train shards on Modal volume)
- **Training**: Runs on Modal A100 (smoke-tested 10 steps, BS=8, `--no-compile --no-multiprocessing`)
- **Gradient accumulation**: Supports effective BS=256 via `--gradient-accumulation-steps`
- **Evaluation scripts**: `eval_paper_metrics.py`, `eval_next_token_predictions.py`, `eval_ordering_likelihood.py`, `eval_live_ping.py` (migrated to PyTorch)
- **Checkpointing**: Auto-save every 200 steps + on SIGINT
- **Wandb integration**: Loss, LR, tokens/sec, eval loss

### Key Files
```
src/ping_llm/
  model.py              # GPT model (RoPE, RMSNorm, softcap)
  train.py              # Training loop (Muon+AdamW, WSD schedule, grad accum)
  config.py             # ModelConfig + TrainConfig dataclasses, CLI parsing
  muon.py               # Muon optimizer
  inference.py          # Inference utilities
  data/
    tokenization.py     # Byte-level tokenization (IP, RTT, timestamps)
    datasource.py       # ProbeRowDataSource + ProbeRowSampler (grain)
    pipeline.py         # Grain pipeline builder (sharding, mp_prefetch)
    loader.py           # create_loader() — pipeline → PyTorch tensors

scripts/train/
  modal_wrapper.py      # Modal deployment wrapper

scripts/eval_*.py       # Evaluation scripts
scripts/data/           # Data preparation and inspection tools
```

---

## Design Decisions

### Why PyTorch over MaxText/JAX?
MaxText was 450 commits behind upstream with breaking restructuring (deleted `types.py`, reorganized `input_pipeline`). Our modifications were ~40 lines across upstream files but required constant rebasing. PyTorch gives us full control, simpler debugging, and access to the Muon optimizer ecosystem.

### Why probe-centric big rows (not per-measurement rows)?
Per-measurement rows caused 50-90% padding waste because measurements tokenize to 14-47 tokens but sequences are 1024. Big rows group all measurements from one source IP, allowing the sampler to fill sequences efficiently with <5% waste.

### Why Muon + AdamW split?
Following nanochat pattern: Muon handles 2D weight matrices (attention, MLP projections) with orthogonal momentum updates. AdamW handles 1D params (embeddings, unembeddings). This consistently outperforms pure AdamW at the same compute budget.

### Why gradient accumulation instead of larger batch size?
Without `torch.compile`, RoPE intermediates aren't fused, causing OOM at BS>=32 on A100-40GB. Gradient accumulation achieves effective BS=256 with micro-batches of BS=32.

---

## Known Issues

1. **`torch.compile` not yet validated**: Smoke tests use `--no-compile`; need to verify compile works for production runs
2. **Eval scripts may have stale imports**: Some scripts in `scripts/` may still reference old MaxText paths
3. **No multi-host data sharding**: Single-GPU only (Modal A100); SLURM multi-GPU scripts exist but untested with new code
4. **Eval iterator resets each interval**: Creates new data loader per eval — some overhead
