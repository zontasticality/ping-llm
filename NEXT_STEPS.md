# NEXT STEPS

Prioritized steps to get ping-llm training for real.

---

## Phase 1: Validate Fixes & First Real Run (immediate)

### 1.1 Smoke test — regression (BS=8, no accum)
```bash
modal run scripts/train/modal_wrapper.py::run \
  --steps 10 --run-name smoke-regression --batch-size 8 --no-compile
```
**Validates**: Fixed `mp_prefetch` (no `--no-multiprocessing` needed), streaming stdout (`PYTHONUNBUFFERED=1`), default accum=1 is no-op.

### 1.2 Smoke test — gradient accumulation (effective BS=256)
```bash
modal run scripts/train/modal_wrapper.py::run \
  --steps 10 --run-name smoke-accum --batch-size 32 \
  --gradient-accumulation-steps 8 --no-compile
```
**Validates**: Grad accum works, no OOM at micro-BS=32, loss is finite and decreasing.

### 1.3 First real training run
```bash
modal run scripts/train/modal_wrapper.py::run \
  --run-name first-real --batch-size 32 \
  --gradient-accumulation-steps 8 --total-steps 1000 --no-compile
```
**Goal**: Verify loss curve looks healthy over 1000 steps. Watch for divergence, NaNs, or stalled loss.

---

## Phase 2: torch.compile & Performance

### 2.1 Test torch.compile
```bash
modal run scripts/train/modal_wrapper.py::run \
  --steps 50 --run-name smoke-compile --batch-size 32 \
  --gradient-accumulation-steps 8
```
With compile enabled (default), RoPE and attention should fuse, potentially allowing larger micro-batch sizes and faster throughput.

### 2.2 Tune micro-batch size with compile
Once compile works, try larger micro-BS (64, 128) to reduce accumulation steps and improve GPU utilization. Target: effective BS=256 with fewer micro-steps.

### 2.3 Measure throughput
Track tokens/sec from training logs. Compare:
- `--no-compile` with BS=32×8 accum
- Compiled with BS=32×8 accum
- Compiled with BS=64×4 accum (if memory allows)

---

## Phase 3: Full Training Run

### 3.1 Run to convergence
```bash
modal run scripts/train/modal_wrapper.py::run \
  --run-name full-v1 --batch-size 32 \
  --gradient-accumulation-steps 8 --total-steps 14000
```
~14k steps at effective BS=256, seq_len=1024.

### 3.2 Monitor via wandb
- Loss curve should show steady decrease through warmup → stable → warmdown phases
- Eval loss should track train loss without large gap

---

## Phase 4: Evaluation & Analysis

### 4.1 Run evaluation suite
```bash
python scripts/eval_paper_metrics.py \
  --checkpoint <path> --output-dir outputs/paper_metrics/
```

### 4.2 Key metrics
- RTT prediction accuracy (MAE, calibration)
- Timestamp prediction (delta accuracy)
- IP prediction (byte-level accuracy)
- Conditional generation: P(RTT | src_ip, dst_ip)
- Live ping comparison

---

## Phase 5: Future Work

- **Multi-GPU training**: Test SLURM scripts with DDP/FSDP
- **Data scaling**: More RIPE Atlas data, additional probe rows
- **Model scaling**: Experiment with larger models (deeper, wider)
- **Ablations**: Timestamp mode, window size, model depth
