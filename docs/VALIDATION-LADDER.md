# Validation Ladder: Progressive Scaling for ping-llm

> **For AI assistants**: When asked to "run validation", "check the pipeline", or launch training,
> consult this document. Always start at the lowest tier that hasn't passed yet. Never skip tiers.
> Record results in `docs/RUN-STATUS.md`. Always use `modal run --detach` for Modal runs.

## Philosophy

Validate cheap before burning expensive GPU hours. Each tier validates one additional layer
of the stack. Fixing a bug at Tier 0 costs $0; discovering it at Tier 5 wastes $17+.

**Rule: never advance to Tier N+1 with a failing Tier N.**

## Summary

| Tier | Name | Validates | GPU | Time | Cost |
|------|------|-----------|-----|------|------|
| 0 | Local sanity | Imports, config, tokenization | None | <30s | $0 |
| 1 | Data pipeline | grain throughput, volume I/O | A100-40GB | 2-5 min | ~$0.15 |
| 2 | Forward pass | Memory fit, loss at init | A100-40GB | 3-5 min | ~$0.15 |
| 3 | Compile warmup | torch.compile, FX cache | A100-40GB | 25-40 min | ~$1.20 |
| 4 | Short training | Loss curve, optimizer, grads | A100-40GB | 10-20 min | ~$0.50 |
| 5 | Full 95M run | Convergence over 14k steps | A100-40GB | ~8h | ~$17 |
| 6 | DDP smoke test | Multi-GPU comms | H100:2 | 10-15 min | ~$1.30 |
| 7 | Full 680M run | Production training | H100:8 | ~2h | ~$90 |

---

## Tier 0: Local Sanity

**Purpose**: Catch broken imports, wrong defaults, and syntax errors before touching Modal.

**Prerequisites**: Local venv with `ping_llm` installed.

**Commands**:
```bash
# Check config defaults
python -c "
from ping_llm.config import ModelConfig, TrainConfig
m = ModelConfig(); t = TrainConfig()
print(f'Model: {m.num_params/1e6:.1f}M params')
print(f'  n_layer={m.n_layer}, n_embd={m.n_embd}, n_head={m.n_head}, head_dim={m.head_dim}')
print(f'Train: BS={t.batch_size}, steps={t.total_steps}, compile={t.compile}')
"

# Check tokenization round-trip
python -c "
from ping_llm.data.tokenization import encode_measurement
meas = {'event_time': 1700000000.0, 'src_addr': '192.168.1.1', 'dst_addr': '10.0.0.1', 'rtt': 42.5}
tokens = encode_measurement(meas, prev_timestamp=None, include_timestamp=True, shuffle_seed=42)
print(f'Tokenized {len(tokens)} tokens: {tokens[:20]}...')
assert len(tokens) > 0, 'Tokenization produced empty output'
print('OK')
"
```

**Pass criteria**:
1. Model prints ~95M params with expected dimensions
2. Tokenization produces non-empty token list, no errors

**Failure modes**:
- `ImportError` -> broken install or circular import after refactor
- Wrong param count -> config defaults were changed accidentally

---

## Tier 1: Data Pipeline Throughput

**Purpose**: Isolate and measure the grain data pipeline independent of model compute.
This is the most common bottleneck — fix it before anything else.

**Prerequisites**: Tier 0 passes. Data volumes mounted (`ping-llm-data`).

**Command**:
```bash
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-pipeline \
  --steps 10 --batch-size 32 --no-compile
```

**Expected duration**: 2-5 min (no compile overhead).

**Expected cost**: ~$0.15

**Pass criteria**:
1. All preflight data checks pass
2. `tok/s > 50,000` at step 10 (after pipeline warmup)
3. Step time < 1s per step (after first step)
4. No `TimeoutError` or worker deadlock messages

**Failure modes**:
- `tok/s < 5,000` -> Data pipeline is the bottleneck. Check:
  - `grain_workers` count vs available CPUs (Modal `cpu=8`, don't use >8 workers)
  - `mp_prefetch` worker count (should match or be less than CPU count)
  - `num_threads` inside `to_iter_dataset` (redundant with mp_prefetch, set to 1-2)
  - ArrayRecord I/O from Modal volume (network storage is slower than local SSD)
- `tok/s 5,000-50,000` -> Pipeline is slow but not catastrophically broken.
  Tokenization in `ProbeRowSampler` may be the bottleneck. Consider pre-tokenization.
- Deadlock / hang -> `pick_performance_config` auto-tuning with spawn mode.
  Use fixed `MultiprocessingOptions` instead.

**Key metric**: `tok/s` at step 10 is the single most important number.
At BS=32, seq_len=1024, each step is 32,768 tokens. 50k tok/s = ~0.65s/step.

---

## Tier 2: Forward Pass (no compile)

**Purpose**: Verify the model fits in GPU memory at target batch size and that
the loss at random initialization is in the expected range.

**Prerequisites**: Tier 1 passes (data pipeline is healthy).

**Command**:
```bash
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-fwd-95m \
  --steps 5 --batch-size 32 --no-compile
```

**Expected duration**: 3-5 min.

**Expected cost**: ~$0.15

**Pass criteria**:
1. No CUDA OOM errors
2. `gpu/peak_allocated_gb < 35` (A100-40GB has 40GB)
3. Loss at step 1 between **4.5 and 6.5**
   - Theoretical: ln(267) = 5.59 for uniform predictions over 267 vocab tokens
   - Padding masking shifts this slightly
4. All 5 steps complete

**Failure modes**:
- `CUDA out of memory` -> Reduce batch size or check model config
- Loss > 8.0 at step 1 -> Loss computation or masking is broken
- Loss < 3.0 at step 1 -> Data leakage or degenerate targets
- Loss = NaN -> Numerical issue in model (check softcap, dtypes)

---

## Tier 3: Compile Warmup

**Purpose**: Validate that `torch.compile` succeeds for this model architecture
and populates the FX graph cache. This is a one-time cost per model arch + batch size.

**Prerequisites**: Tier 2 passes. **Must use `--detach`** — compile takes 25-40 min.

**Command**:
```bash
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-compile-95m \
  --steps 5 --batch-size 32
```

**Expected duration**: 25-40 min (dominated by first-step compilation).

**Expected cost**: ~$1.20

**Pass criteria**:
1. Step 1 completes (logged as `step 1/5 loss=X.XXXX (Xs)`)
2. Compile time < 45 min (shown in step 1 elapsed time)
3. Loss at step 1 matches Tier 2 within 0.5 (compile should not change numerics significantly)
4. FX graph cache populated: new entries in `torch_cache/fxgraph/` on `ping-llm` volume
5. tok/s after compile should be **higher** than Tier 2 (compile speedup)

**Failure modes**:
- Container killed before step 1 -> Didn't use `--detach`, or compile exceeded timeout
- Repeated "compiling" messages -> Graph breaks causing recompilation every step.
  Check for Python-level control flow in the model's `forward()` method.
- No speedup vs Tier 2 -> Compile produced suboptimal code; check for unsupported ops

**Cache behavior**: After this tier passes, subsequent runs with the same model config
+ batch size will skip compilation (~1 min startup instead of ~30 min). Cache is stored
on the `ping-llm` Modal volume at `/mnt/outputs/torch_cache/`. Cache keys depend on:
- Model architecture (layer count, dims, activation)
- Input tensor shapes (batch_size, seq_len)
- PyTorch version
- GPU architecture (SM80 for A100, SM90 for H100)

Changing any of these = cache miss = full recompilation.

---

## Tier 4: Short Training (200 steps)

**Purpose**: Validate that training dynamics are healthy — loss decreases,
gradients are stable, memory doesn't leak.

**Prerequisites**: Tier 3 passes (or Tier 2 if running with `--no-compile`).

**Command**:
```bash
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-short-95m \
  --steps 200 --batch-size 32
```

**Expected duration**: 10-20 min (compile cached from Tier 3).

**Expected cost**: ~$0.50

**Pass criteria**:
1. Loss at step 200 is meaningfully lower than step 1 (e.g., 5.5 -> < 4.5)
2. Loss curve is monotonically decreasing (small fluctuations OK, large spikes not OK)
3. `train/grad_norm` stays below 5.0 (clipped at 1.0, but pre-clip norm is logged)
4. `gpu/peak_allocated_gb` remains stable across steps (no memory leak)
5. tok/s is stable after initial steps
6. Checkpoint saved at step 200
7. Eval loss at step 100 and 200 is in the same ballpark as train loss

**Failure modes**:
- Loss not decreasing -> LR too low, optimizer bug, or data returning constant batches
- Loss spikes -> Gradient explosion. Check `train/grad_norm` for spikes > 10x median.
  Reduce learning rate or increase warmup.
- Memory climbing -> Leak in data pipeline, wandb, or gradient accumulation
- tok/s degrading over time -> Data pipeline falling behind (prefetch buffer draining)

**Note on WSD schedule**: With `total_steps=200`, the warmup (1%) is just 2 steps and
the warmdown (50%) starts at step 100. The loss curve shape will look different from a
14k-step run. This is fine — the goal is to verify the machinery works, not to match
the final schedule shape.

---

## Tier 5: Full 95M Training (14k steps)

**Purpose**: Full validation run. Prove the model converges on the real data
with the production schedule.

**Prerequisites**: Tier 4 passes.

**Command**:
```bash
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name 95m-full \
  --steps 14000 --batch-size 32
```

**Expected duration**: ~8h.

**Expected cost**: ~$17 (A100-40GB at $2.10/hr).

**Pass criteria**:
1. Training completes all 14,000 steps
2. Final train loss < 3.5 (calibrate after first successful run)
3. Eval loss tracks train loss (gap < 1.0)
4. Loss curve shows WSD shape: warmup ramp, stable plateau, decay drop
5. Checkpoints saved every 200 steps, resume works from `latest.pt`
6. Total tokens: 14000 * 32 * 1024 = ~460M tokens processed
7. Persistent log saved to volume: `logs/95m-full.log`

**Failure modes**:
- Loss plateaus early -> LR too high, data too easy, or dataset exhausted (repeat not working)
- Eval diverges from train -> Overfitting or train/eval data contamination
- Container killed -> Check `modal app list` for state. If "stopped" unexpectedly,
  check logs for OOM, timeout, or CLI cancellation issues.

---

## Tier 6: DDP Smoke Test (Future — Phase 2)

> **Status**: Requires DDP implementation in `train.py`. Placeholder only.

**Purpose**: Validate multi-GPU training works and gradients synchronize correctly.

**Command** (estimated):
```bash
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-ddp \
  --steps 20 --batch-size 32 --gpu "H100:2"
```

**Pass criteria**:
1. Loss at step 1 matches single-GPU Tier 2/3 loss (within 0.1)
2. No NCCL errors
3. Both GPUs show utilization
4. tok/s scales ~1.8-2x vs single GPU

---

## Tier 7: Full 680M Training (Future — Phase 3)

> **Status**: Requires DDP (Phase 2) + Tier 6 passing. Placeholder only.

**Command** (estimated):
```bash
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name 680m-full \
  --steps 14000 --batch-size 32 \
  --n-layer 24 --n-embd 1536 --n-head 12 --head-dim 128 \
  --gpu "H100:8"
```

**Expected cost**: ~$90 (8x H100 at $3.95/hr for ~2h + overhead).

**Pass criteria**: Lower final loss than Tier 5 (larger model should do better on same data).

---

## Interpreting Results

### Loss Values
- **Random init**: ln(267) = 5.59 for uniform predictions. Step-1 loss of 4.5-6.5 is normal.
- **Good training**: Steady decrease. 95M model should reach < 3.5 over 14k steps (calibrate after first run).
- **WSD curve shape**: Gentle decline during stable phase, sharp drop during warmdown (last 50% of steps).

### Throughput (tok/s)
- **Eager (no compile), 95M, BS=32**: Baseline. Expect 50k-100k tok/s if pipeline is healthy.
- **Compiled, 95M, BS=32**: Should be 1.5-3x faster than eager.
- **If tok/s < 5k**: Data pipeline is the bottleneck, not the GPU. Fix before scaling.

### GPU Memory
- **95M, BS=32, compiled on A100-40GB**: Expect ~15-25 GB peak (comfortable).
- **680M, BS=32 on A100-40GB**: Will OOM. Known limitation, requires H100 or smaller BS.

### Gradient Norms
- **Normal**: 0.1-2.0 (after clip at 1.0; pre-clip value is logged).
- **Warning**: Spike > 5x median = potential instability.
- **Critical**: Sustained > 10 = likely diverging, reduce LR.

### Common Failure Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | BS too large or wrong model config | Reduce `--batch-size` |
| `tok/s < 5,000` | Data pipeline bottleneck | Fix grain workers, see Tier 1 |
| Loss = NaN | Numerical explosion | Reduce LR, check softcap/dtypes |
| Container killed | No `--detach`, or compile timeout | Always use `--detach` |
| No wandb data | Missing secret or `--wandb-mode disabled` | Check `wandb-secret` in Modal |
| Loss stuck flat | LR too low or broken optimizer | Check schedule params |
| Step 1 takes >45 min | Compile (expected for new arch) | Wait; will be cached after |

---

## Decision Tree

```
After running Tier N:
  1. Did ALL pass criteria pass?
     YES -> Record results in RUN-STATUS.md, advance to Tier N+1
     NO  -> Go to step 2
  2. Is the failure cosmetic (e.g., wandb glitch, log formatting)?
     YES -> Document it, advance anyway
     NO  -> Go to step 3
  3. Fix the issue, re-run Tier N from scratch.
     Do NOT advance with a substantive failure.
```

---

## Quick Reference: All Commands

```bash
# Tier 0: Local sanity (no GPU)
python -c "from ping_llm.config import ModelConfig; print(f'{ModelConfig().num_params/1e6:.1f}M')"

# Tier 1: Data pipeline
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-pipeline --steps 10 --batch-size 32 --no-compile

# Tier 2: Forward pass
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-fwd-95m --steps 5 --batch-size 32 --no-compile

# Tier 3: Compile warmup
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-compile-95m --steps 5 --batch-size 32

# Tier 4: Short training
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name val-short-95m --steps 200 --batch-size 32

# Tier 5: Full 95M
modal run --detach scripts/train/modal_wrapper.py::run \
  --run-name 95m-full --steps 14000 --batch-size 32

# Tier 6: DDP smoke (future)
# modal run --detach scripts/train/modal_wrapper.py::run \
#   --run-name val-ddp --steps 20 --batch-size 32 --gpu "H100:2"

# Tier 7: Full 680M (future)
# modal run --detach scripts/train/modal_wrapper.py::run \
#   --run-name 680m-full --steps 14000 --batch-size 32 \
#   --n-layer 24 --n-embd 1536 --n-head 12 --head-dim 128 --gpu "H100:8"
```
