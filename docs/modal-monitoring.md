# Modal Training: Monitoring & Iteration Plan

## Current Gaps

### 1. No persistent logs
If the `modal run` CLI disconnects, all stdout is lost. Wandb only captures
metrics, not debug messages.

### 2. No GPU memory monitoring
Can't see memory pressure during training — only discover OOM after the fact.

### 3. Compile cache is opaque
Can't tell if torch.compile cache was hit or recompiled.

### 4. GPU/CPU not configurable from CLI
Hardcoded `gpu="A100"`, `cpu=8`. Must edit code to try H100 or multi-GPU.

### 5. No log interval control from Modal CLI
Uses config default (10 steps). Can't adjust without editing config.

---

## Proposed Changes

### A. Persistent log file (`modal_wrapper.py`)
Write subprocess stdout to a file on the output volume alongside console:
```python
log_path = f"{OUTPUTS_MOUNT}/logs/{run_name}.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as log_file:
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        print(line, end="", flush=True)
```
Then check logs anytime:
```bash
modal volume get ping-llm logs/my-run.log
```

### B. GPU memory logging (`train.py`)
Add to the wandb log dict at each `log_interval`:
```python
if torch.cuda.is_available():
    wandb.log({
        "gpu/allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu/peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    })
    torch.cuda.reset_peak_memory_stats()
```

### C. Gradient norm logging (`train.py`)
After `clip_grad_norm_`, log the returned total norm:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(...)
wandb.log({"train/grad_norm": grad_norm.item()})
```
Useful for detecting instability (spikes = loss about to diverge).

### D. Compile time logging (`train.py`)
Wrap the first forward pass with timing:
```python
if step == start_step:
    t_compile = time.time()
    # ... forward pass ...
    print(f"First step (incl. compile): {time.time() - t_compile:.1f}s")
```

### E. Configurable GPU/CPU in modal wrapper
Add CLI params that map to `@app.function()` decorator. Since Modal decorators
are static, use a factory pattern or environment variable override:
```python
gpu: str = "A100"      # "A100", "A100-80GB:2", "H100:8"
cpu: int = 8
```

### F. Additional CLI passthrough params
Wire these through `_run()` and `run()`:
- `--log-interval` (default: 10)
- `--eval-interval` (default: 100)
- `--wandb-mode` (default: "online")
- Model size overrides: `--n-layer`, `--n-embd`, `--n-head`, `--head-dim`

---

## Priority Order

1. **Persistent logs** — cheapest, most impactful. No more lost output.
2. **GPU memory logging** — catches OOM risks early in wandb dashboard.
3. **Gradient norm** — early warning for training instability.
4. **CLI passthrough params** — less code editing per experiment.
5. **Configurable GPU** — needed for Phase 2 (DDP on 8×H100).
6. **Compile time logging** — nice-to-have visibility.
