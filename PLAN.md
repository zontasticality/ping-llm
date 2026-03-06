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
- **Deployment**: Modal (single/multi-GPU), SLURM scripts (untested)
- **Data**: grain pipeline with ArrayRecord, probe-centric big-row format

### Model Sizes

Two configurations:

| Config | Layers | Embd | Heads | Params | Use Case |
|--------|--------|------|-------|--------|----------|
| Small (95M) | 20 | 640 | 10 | ~95M | Validation on 1×A100-40GB |
| Full (680M) | 24 | 1536 | 12 | ~680M | Production on 8×H100 |

Default config is the small model for cheap iteration. Scale up via CLI args.

### Architecture
- 267 vocab (11 role tokens + 256 byte tokens)
- RoPE positional encoding, logit softcap at 15
- ReLU² activation, parameterless RMSNorm
- 1024 max sequence length
- **Tokenization**: Custom byte-level scheme (`src/ping_llm/data/tokenization.py`)
  - Role tokens: MEASUREMENT_START, SRC_IPV4/IPV6, DST_IPV4/IPV6, TIMESTAMP_ABS/DELTA1/DELTA4, RTT_START, FAILED
  - RTT: 5-bit exponent + 11-bit mantissa (< 0.1% relative error)
  - Timestamps: Delta-encoded (95%+ fit in 1 byte)
  - Field order randomization for joint distribution learning

### What Works
- **Data pipeline**: Probe-centric big-row ArrayRecord with runtime tokenization
  - Multi-scale temporal sampling (log-uniform window sizes)
  - 3 timestamp modes (full/partial/none) for data augmentation
  - <5% padding waste, sharded ArrayRecord (4 train shards on Modal volume)
- **Training on Modal**: Smoke-tested on A100-40GB (BS=8 compiled + no-compile)
- **torch.compile**: Works, ~23 min compile (cached across runs via TORCHINDUCTOR_FX_GRAPH_CACHE)
- **Gradient accumulation**: Effective BS=256 via `--gradient-accumulation-steps`
- **Streaming output**: PYTHONUNBUFFERED=1, per-step progress logging
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

## Training Strategy

### Phase 1: Validate at 95M on 1×A100-40GB (~$25)
- Small model (20 layers, 640 embd, ~95M params)
- BS=32 compiled should fit in 40GB
- Full training run to convergence (14k steps)
- Validate: loss curve, checkpointing, eval, wandb
- **Goal**: Confirm the entire pipeline works end-to-end

### Phase 2: Scale to 680M on 8×H100 (~$50-90)
- Add DDP support to train.py
- Switch Modal to `gpu="H100:8"`
- Full model (24 layers, 1536 embd, ~680M params)
- Device BS=16-32 per GPU, minimal or no grad accum
- ~2h training time (comparable to nanochat d24)

### Cost Comparison

| Setup | Model | BS | Time | Cost |
|-------|-------|----|------|------|
| 1×A100-40GB compiled | 95M | 32 | ~8h | ~$17 |
| 1×A100-40GB compiled | 680M | 8×32acc | ~800h | ~$1700 |
| 8×H100 | 680M | 16×8gpu | ~2h | ~$90 |

The 680M model is only viable on multi-GPU.

---

## Design Decisions

### Why PyTorch over MaxText/JAX?
MaxText was 450 commits behind upstream with breaking restructuring. PyTorch gives full control, simpler debugging, and access to the Muon optimizer ecosystem.

### Why probe-centric big rows (not per-measurement rows)?
Per-measurement rows caused 50-90% padding waste because measurements tokenize to 14-47 tokens but sequences are 1024. Big rows group all measurements from one source IP, allowing the sampler to fill sequences efficiently with <5% waste.

### Why Muon + AdamW split?
Following nanochat pattern: Muon handles 2D weight matrices with orthogonal momentum updates. AdamW handles embeddings. Consistently outperforms pure AdamW at the same compute budget.

### Why two model sizes?
The 680M model OOMs at BS>=32 on A100-40GB even with torch.compile. A 95M model fits comfortably and allows cheap validation of the full pipeline. Scale up to 680M only on multi-GPU (8×H100).

---

## Known Issues

1. **No DDP/multi-GPU support yet**: train.py is single-GPU only; needed for 680M training
2. **Eval scripts may have stale imports**: Some scripts in `scripts/` may reference old MaxText paths
3. **Eval iterator resets each interval**: Creates new data loader per eval — some overhead
4. **Modal CLI hangs after completion**: Double `outputs_vol.commit()` issue (fixed in uncommitted changes)
