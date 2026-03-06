# MaxText vs Nanochat: Framework Comparison for ping-llm

*Research conducted 2026-02-25. Source discussion: [karpathy/nanochat#481](https://github.com/karpathy/nanochat/discussions/481)*

---

## Context

ping-llm currently uses a **MaxText fork** (450 commits behind upstream) to train a ~95M parameter decoder-only Transformer on RIPE Atlas network latency measurements. The question is whether pivoting to nanochat would be beneficial.

### Current ping-llm Architecture (MaxText)
```
~95M params | 267 vocab | 20 layers | 640 emb dim | 10 heads | 64 head dim
MLP dim 2048 (3.2x ratio) | 1024 seq len | RoPE | Flash Attention | bfloat16
AdamW (β1=0.9, β2=0.999, wd=0.01) | cosine LR schedule | per_device_batch=256
```

### Nanochat #481 Architecture (Speedrun Recipe)
```
~1.38B params | 50304 vocab | 24 layers | 1536 emb dim | 12 heads | 128 head dim
ReLU² activation | RoPE (θ=10k) | RMSNorm (no learnable params) | 2048 seq len
Split optimizer: AdamW (embeddings) + Muon (matrices) | WSD schedule (50% warmdown)
Logit softcapping ±15 | SSSL sliding window (3×1024 + 1×2048 repeating)
Value embeddings w/ gating | Per-layer residual scalars | Depth-dependent weight decay
```

---

## Feature-by-Feature Comparison: MaxText Defaults vs Nanochat Recipe

### What MaxText already supports (config-level):

| Feature | Nanochat #481 | MaxText | Status |
|---------|--------------|---------|--------|
| RoPE (θ=10k) | ✅ | ✅ default | **Match** |
| Sequence length | 2048 | 2048 default (ping-llm uses 1024) | **Match** |
| Head dim 128 | ✅ | ✅ default | Match (ping-llm uses 64) |
| Separate embed/unembed | ✅ | ✅ `logits_via_embedding: false` | **Match** |
| AdamW β1/β2/ε | 0.9/0.95/1e-8 | 0.9/0.95/1e-8 default | Match (ping-llm uses β2=0.999) |
| Muon optimizer | ✅ split design | ✅ `opt_type: "muon"` w/ auto param split | **Partial** |
| WSD schedule | 50% linear warmdown | ✅ `lr_schedule_type: 'wsd'` | **Configurable** |
| Logit softcapping | ±15 | ✅ `final_logits_soft_cap` param | **Configurable** |
| QK normalization | ✅ post-rotation | ✅ `use_qk_norm: True` flag | **Configurable** (placement may differ) |
| Flash Attention | FA3 (H100-specific) | Pallas/Splash (TPU), cudnn_flash_te (GPU) | **Partial** |
| Gradient clipping | Not specified | ✅ `gradient_clipping_threshold: 1.0` | **Available** |
| bfloat16 | ✅ | ✅ default | **Match** |

### What MaxText is MISSING (would need code changes):

| Feature | Nanochat #481 | MaxText | Difficulty to Add |
|---------|--------------|---------|-------------------|
| **ReLU² activation** | `F.relu(x).square()` | Not in activation registry (only silu, gelu, relu, sigmoid) | Easy (~5 lines in `linears.py`) |
| **RMSNorm w/o learnable params** | No gamma/beta | Always has learnable `self.scale` param | Easy (add config flag) |
| **SSSL sliding window** | 3×1024 + 1×2048 repeating per-layer | Only global fixed `sliding_window_size` | Medium (per-layer attention config) |
| **Value embeddings w/ gating** | At alternating layers | Not in any decoder block | Hard (new architecture component) |
| **Per-layer residual scalars** | `x = λ_resid * x + λ_x0 * x0` | Standard `x = x + sublayer(x)` only | Medium (modify decoder layer) |
| **Per-parameter-group LRs** | 4+ different LRs (embed=0.3, output=0.004, scalars=0.005, x0=0.5) | Single global LR schedule | Hard (optimizer rework) |
| **Depth-dependent weight decay** | `0.2 * (12/depth)²` | Flat `adam_weight_decay` only | Medium (optimizer wrapper) |
| **NorMuon / Cautious WD** | Enhanced Muon variants | Not in optax's Muon | Hard (custom optimizer code) |
| **Nesterov momentum warmup** | 0.85→0.95 over 300 steps | Fixed `muon_beta: 0.95` | Medium |
| **BOS-aligned BestFit-Crop packing** | ✅ | Not in data pipeline | Medium (dataloader change) |

### Muon Details

MaxText's Muon implementation (via `optax.contrib._muon`) does auto-split parameters:
- Embeddings, scale/bias, logits_dense → **AdamW**
- Weight matrices (attention projections, MLP) → **Muon** with configurable dimension numbers

However, it uses a **single global learning rate** for both groups and lacks the nanochat recipe's per-group LR scheduling, Nesterov warmup, NorMuon variance reduction, and cautious weight decay.

Source: `maxtext/src/maxtext/utils/muon_utils.py:transform_logic()` — returns `None` (→ AdamW) for paths containing "scale", "bias", "embedding", "logits_dense".

---

## Nanochat as an Alternative Framework

### What nanochat IS

Nanochat (formerly nanogpt-speedrun) is a **minimalist PyTorch training codebase** (~1-2 files) focused on training GPT-2 scale models as fast as possible. The discussion #481 documents the current record-holding training recipe. Key characteristics:

- **Pure PyTorch** — no framework abstractions, direct CUDA/H100 optimization
- **Single-file training loop** — maximally hackable
- **8×H100 target** — every optimization is hardware-specific
- **FineWeb-edu dataset** — standard NLP benchmark data
- **~1.38B param fixed architecture** — not configurable, hand-tuned

### What nanochat is NOT

- Not a general-purpose training framework
- Not designed for custom tokenizers or datasets
- Not designed for TPU (it's deeply CUDA/H100-specific: Flash Attention 3, tensor core layouts)
- Not designed for arbitrary model scales (the recipe is tuned for exactly ~1.38B)
- Has no data pipeline abstraction (expects pre-tokenized data in a specific format)
- Has no checkpointing framework, no eval framework, no config system

---

## Pivot Analysis for ping-llm

### Arguments FOR pivoting to nanochat

1. **Simplicity**: ~1-2 files vs MaxText's hundreds. Much easier to understand, debug, and modify.
2. **No upstream sync problem**: ping-llm is 450 commits behind MaxText with painful structural divergence (deleted `types.py`, restructured `input_pipeline/`). Nanochat has no framework churn.
3. **Modern training recipe**: The nanochat recipe incorporates cutting-edge techniques (Muon, ReLU², value embeddings, per-layer scaling) that would genuinely improve model quality.
4. **PyTorch ecosystem**: Easier GPU debugging, broader library compatibility, no JAX/XLA quirks.
5. **Custom model flexibility**: Since ping-llm already needs a non-standard architecture (267 vocab, custom tokenization, probe-centric data), a hackable single-file approach may be more natural than fitting into MaxText's abstractions.

### Arguments AGAINST pivoting to nanochat

1. **Scale mismatch**: The nanochat recipe is for ~1.38B params. ping-llm is ~95M. The hyperparameters (LR schedules, weight decay scaling, Muon config) are tuned for that specific scale and will NOT transfer. You'd need to re-derive the training recipe for 95M.
2. **Hardware mismatch**: Nanochat is optimized for 8×H100 with Flash Attention 3. ping-llm trains on Modal A100s/B200s. FA3 is H100-only. Many CUDA-specific optimizations won't apply.
3. **Re-implementation cost**: You'd need to rewrite:
   - The entire data pipeline (probe-centric ArrayRecord → tokenization → batching)
   - Checkpointing (currently using Orbax via MaxText)
   - Evaluation framework
   - Config management
   - Modal deployment integration
4. **No TPU path**: If you ever want TPU training (which MaxText is built for), nanochat is a dead end.
5. **The recipe IS the codebase**: Nanochat's value is the specific training recipe, not the framework. You can adopt the recipe's ideas (Muon, WSD schedule, ReLU²) without adopting the codebase.

### The Middle Path: Cherry-Pick Techniques into MaxText (or a New Framework)

Rather than a full pivot, consider adopting specific nanochat innovations:

| Technique | Effort in MaxText | Expected Impact |
|-----------|-------------------|-----------------|
| WSD schedule (50% warmdown) | Config-only: `lr_schedule_type: 'wsd'` | Medium — better end-of-training convergence |
| Muon optimizer | Config-only: `opt_type: "muon"` | High — well-documented speedup for weight matrices |
| Logit softcapping | Config-only: `final_logits_soft_cap: 15.0` | Low-medium — training stability |
| QK normalization | Config-only: `use_qk_norm: True` | Low-medium — attention stability |
| ReLU² activation | ~5 lines in `linears.py` | Unknown at 95M scale |
| β2=0.95 (from 0.999) | Config-only | Likely beneficial (standard modern practice) |

These config-only changes could be applied **immediately** without any code modifications to upstream MaxText, and would capture the most transferable insights from the nanochat recipe.

### What NOT to adopt (at 95M scale)

- **Value embeddings with gating** — significant engineering for unclear benefit at this scale
- **Per-layer residual scalars** — adds complexity; primarily helps at >>1B scale
- **Depth-dependent weight decay** — the scaling formula `(12/depth)²` was tuned for 24 layers at 1.38B
- **Per-parameter-group LRs** — hard to implement in MaxText and the specific values won't transfer
- **SSSL sliding window** — at seq_len=1024 with 95M params, full attention is likely fine

---

## Upstream Sync vs Framework Switch: Cost Comparison

### Option A: Sync MaxText fork (Phase 3 from NEXT_STEPS.md)
- **Effort**: ~1-2 days (manual re-integration of ~40 lines across 4 files)
- **Risk**: Moderate — upstream interfaces may have changed enough to break `_network_data_processing.py`
- **Benefit**: Access to upstream improvements, continued MaxText ecosystem
- **Ongoing cost**: Will need periodic re-syncs as MaxText evolves

### Option B: Pivot to nanochat
- **Effort**: ~1-2 weeks (rewrite data pipeline, checkpointing, eval, deployment)
- **Risk**: High — hyperparameter re-derivation needed for 95M scale on A100
- **Benefit**: Simpler codebase, no upstream sync, modern training recipe baked in
- **Ongoing cost**: Low — stable, minimal codebase

### Option C: Pivot to a different framework (e.g., nanoGPT, litgpt, or custom PyTorch)
- **Effort**: ~1 week (less than nanochat since you wouldn't need to adapt the speedrun-specific code)
- **Risk**: Moderate — standard PyTorch training loop is well-understood
- **Benefit**: Full control, no upstream dependency, adopt nanochat ideas selectively
- **Ongoing cost**: You own all the code

### [MY SYNTHESIS] Recommendation

The upstream sync (Option A) is the lowest-risk path for the immediate term. The MaxText fork works, the modifications are minimal (~40 lines), and the sync is a known-scope task.

However, if the upstream sync proves painful or if MaxText's framework constraints become limiting for future experiments, a **custom PyTorch training loop** (Option C, not nanochat specifically) would be the natural next step. You'd cherry-pick nanochat's best ideas (Muon, WSD, possibly ReLU²) while keeping your existing data pipeline concepts.

Full nanochat pivot (Option B) is the worst option because you'd inherit an opinionated codebase designed for a completely different scale, hardware, and task — then need to modify everything anyway.

---

## Immediate Actionable Config Changes

These can be applied to `latency_network.yml` right now with zero code changes:

```yaml
# Adopt nanochat-style training improvements:
adam_b2: 0.95          # Modern standard (currently 0.999, which is outdated for LLMs)
lr_schedule_type: 'wsd'  # WSD schedule instead of cosine
wsd_decay_steps_fraction: 0.5  # 50% warmdown like nanochat
wsd_decay_style: 'linear'
learning_rate_final_fraction: 0.0  # Decay fully to zero
# opt_type: 'muon'     # Requires MaxText >= current upstream (may not work on stale fork)
```

**Caveat**: These values are derived from a 1.38B model training on NLP data. They should be tested with short runs before committing to a full training run at 95M on network data. The transferability of these hyperparameters across scales and domains is uncertain.
