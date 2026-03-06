# NEXT STEPS

Prioritized steps to get ping-llm training for real.

---

## Phase 1: 95M Validation Run on 1×A100-40GB (immediate, ~$25)

### 1.1 Set model defaults to 95M
Change `ModelConfig` defaults to the small model:
- n_layer=20, n_embd=640, n_head=10, head_dim=64

### 1.2 Smoke test compiled at BS=32
```bash
modal run scripts/train/modal_wrapper.py::run \
  --steps 20 --run-name smoke-95m --batch-size 32
```
**Validates**: 95M model fits in A100-40GB at BS=32 compiled, loss decreases.

### 1.3 Full training run (14k steps)
```bash
modal run scripts/train/modal_wrapper.py::run \
  --run-name 95m-full --batch-size 32 --total-steps 14000
```
~8h, ~$17. Monitor on wandb. Should see clean WSD loss curve.

### 1.4 Run evaluation
```bash
python scripts/eval_paper_metrics.py \
  --checkpoint <path> --output-dir outputs/paper_metrics_95m/
```

---

## Phase 2: Add DDP for Multi-GPU (before scaling to 680M)

### 2.1 Add DDP to train.py
- `torch.distributed.init_process_group()`
- Wrap model in `DistributedDataParallel`
- Shard data across ranks
- Only save checkpoints / log on rank 0

### 2.2 Update modal_wrapper.py for multi-GPU
- Change `gpu="A100"` to configurable (e.g. `gpu="H100:8"`)
- Launch training with `torchrun --nproc_per_node=8`
- Increase CPU count for 8-GPU instance

### 2.3 Smoke test DDP on 2×A100
```bash
# Quick DDP validation before burning $90 on 8×H100
modal run scripts/train/modal_wrapper.py::run \
  --steps 20 --run-name smoke-ddp --batch-size 32 --gpu "A100:2"
```

---

## Phase 3: 680M Production Run on 8×H100 (~$90)

### 3.1 Full training
```bash
modal run scripts/train/modal_wrapper.py::run \
  --run-name 680m-full --batch-size 16 \
  --n-layer 24 --n-embd 1536 --n-head 12 --head-dim 128 \
  --total-steps 14000 --gpu "H100:8"
```
~2h, ~$90. Device BS=16 per GPU, 8 GPUs = 128 per step, 2 accum = 256 effective.

### 3.2 Monitor + evaluate
- Watch wandb for loss curve health
- Run eval suite on final checkpoint
- Compare 95M vs 680M results

---

## Phase 4: Evaluation & Analysis

### 4.1 Key metrics
- RTT prediction accuracy (MAE, calibration)
- Timestamp prediction (delta accuracy)
- IP prediction (byte-level accuracy)
- Conditional generation: P(RTT | src_ip, dst_ip)
- Live ping comparison

### 4.2 Ablations (if results warrant)
- Model size: 95M vs 680M
- Timestamp mode: full vs mixed
- Window size: fixed vs log-uniform

---

## Future Work
- **FP8 training**: H100 supports FP8, ~2× matmul speedup (nanochat-style)
- **Flash Attention 3**: H100-only, another 30-40% speedup
- **Data scaling**: More RIPE Atlas data, additional probe rows
- **Larger models**: Scale beyond 680M if results justify it
