# NEXT STEPS

Prioritized, actionable steps to get ping-llm fully working and production-ready.

---

## Phase 1: Validate & Run (immediate)

### 1.1 Verify config loads without warnings
```bash
DECOUPLE_GCLOUD=TRUE python -c "
from MaxText import pyconfig
config = pyconfig.initialize(None, ['src/MaxText/configs/latency_network.yml', 'run_name=test', 'steps=1'])
print('warmup_steps_fraction:', config.warmup_steps_fraction)
print('adam_weight_decay:', config.adam_weight_decay)
print('cosine_learning_rate_final_fraction:', config.cosine_learning_rate_final_fraction)
print('packing:', config.packing)
"
```
**Why**: Config field mismatches were silently dropping keys — `adam_weight_decay` was 10x too high (0.1 vs intended 0.01), `warmup_steps` was 25x too long. These have been fixed but should be validated.

### 1.2 Run a local smoke test (CPU, 5 steps)
```bash
DECOUPLE_GCLOUD=TRUE python -m MaxText.train \
  src/MaxText/configs/latency_network.yml \
  run_name=smoke_test hardware=cpu steps=5 per_device_batch_size=2 \
  enable_checkpointing=false
```
**Why**: Confirm the full pipeline works end-to-end with the corrected config.

### 1.3 Run a Modal training run (10 steps)
```bash
modal run scripts/train/modal_wrapper.py::run \
  --run-name validation_run --steps 10 --batch-size 256
```
**Why**: Verify Modal deployment still works after config changes. Watch for:
- No Pydantic warnings about dropped keys
- Loss is finite and decreasing
- Batch size is actually 256 (not overridden to 128 — see archive/PERFORMANCE_ISSUE_DIAGNOSIS.md)

---

## Phase 2: Fix Known Issues (before next full training run)

### 2.1 Fix modal_wrapper.py batch size default
The modal wrapper may still default to `batch_size=128`, overriding the config's `per_device_batch_size: 256`. Check and fix:
```python
# scripts/train/modal_wrapper.py
def run(
    batch_size: int = 256,  # ← Must match config, was 128
    ...
)
```
**Impact**: Without this fix, GPU memory is 70% idle (see archive/PERFORMANCE_ISSUE_DIAGNOSIS.md).

### 2.2 Fix stale test imports
These scripts import from the old `tokenization` module (moved to `MaxText.input_pipeline.network_tokenization`):
- `scripts/data/verify_tokenization.py`
- `scripts/tests/smoke_test_maxtext.py`
- `scripts/tests/test_tokenization_standalone.py`

Update imports or add a compatibility shim.

### 2.3 Increase eval_steps
Currently `eval_steps: 1` means only 1 batch per eval interval — unreliable loss estimates.
```yaml
eval_steps: 5  # or 10, for more stable eval metrics
```

---

## Phase 3: Upstream Sync (high priority, complex)

### Problem
We are **450 commits behind** upstream `google/maxtext`. Upstream has:
- **Deleted** `configs/types.py` (restructured config system)
- **Restructured** `input_pipeline/` (PR #3124: `aireen/input_restructure2`)
- Added new features (MTP, Qwen3, DeepSeek R1, etc.)

### Our modifications to upstream files (total ~40 lines)
1. **`input_pipeline_interface.py`** (+3 lines): import + registration of `network` backend
2. **`configs/types.py`** (+12 lines): `NetworkDataset` class, `DatasetType.NETWORK` enum, mixin
3. **`_grain_data_processing.py`** (+2 lines): cosmetic whitespace only
4. **`train.py`** (+25 lines): TF import removal, eval reset removal, KeyboardInterrupt checkpoint handler

### Strategy
1. **Read upstream's restructuring** — understand where `types.py` content moved and how `input_pipeline_interface.py` changed
2. **Create a fresh branch from upstream/main**
3. **Cherry-pick our custom files** (these have no upstream equivalent, so no conflicts):
   - `src/MaxText/input_pipeline/_network_data_processing.py`
   - `src/MaxText/input_pipeline/probe_chunk_pipeline.py`
   - `src/MaxText/input_pipeline/_probe_chunk_datasource.py`
   - `src/MaxText/input_pipeline/network_tokenization.py`
   - `src/MaxText/configs/latency_network.yml`
   - `src/MaxText/configs/decoupled_base_test.yml`
4. **Manually re-apply our 3 integration points** to the new upstream code:
   - Register `network` in the new input pipeline interface
   - Add `NetworkDataset` config class to wherever types moved
   - Add `DatasetType.NETWORK` enum value
5. **Decide on train.py changes**: The KeyboardInterrupt handler is useful; the TF removal and eval reset changes should be re-evaluated against the new upstream.

### Risk
The upstream restructuring may have changed interfaces enough that `_network_data_processing.py` needs updates (e.g., new iterator signatures, different config access patterns).

---

## Phase 4: Performance Optimization

### 4.1 Verify batch size is actually applied
After fixing modal_wrapper.py (Phase 2.1), verify:
```
# In training logs, look for:
per_device_batch_size: 256
# NOT 128
```

### 4.2 XLA optimization flags
The A100-optimized flags in `modal_wrapper.py` should improve MFU:
```bash
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_triton_softmax_fusion=true
--xla_gpu_triton_gemm_any=true
```
Expected: +5-10% MFU on top of batch size fix.

### 4.3 Profile data pipeline
If GPU utilization is still <90%:
```yaml
grain_debug_mode: true
grain_visualization_dir: "/mnt/grain_viz"
```
Check for data loading bottlenecks.

---

## Phase 5: Evaluation & Paper

### 5.1 Run comprehensive evaluation suite
```bash
python scripts/eval_paper_metrics.py \
  --checkpoint outputs/latency_network/<run>/checkpoints/<step>/ \
  --output-dir outputs/paper_metrics/
```

### 5.2 Key metrics to measure
- **RTT prediction accuracy**: MAE, calibration
- **Timestamp prediction**: Delta accuracy
- **IP prediction**: Byte-level accuracy
- **Conditional generation**: P(RTT | src_ip, dst_ip)
- **Live ping comparison**: Model predictions vs actual pings

### 5.3 Ablation studies
- Timestamp mode ablation (full only vs mixed)
- Window size ablation (fixed vs log-uniform)
- Model depth ablation (10 vs 20 vs 30 layers)

---

## Phase 6: Multi-Host Scaling (future)

### 6.1 Implement data sharding
In `_network_data_processing.py`, pass host index/count to `build_probe_chunk_dataset()`:
```python
# Shard data across hosts
dataset = dataset[dataloading_host_index::dataloading_host_count]
```

### 6.2 Add finite eval repeats
Add a `repeat` parameter to `build_probe_chunk_dataset()` so eval can use `repeat(1)` instead of `repeat(None)`.

---

## Archive

Historical documents have been moved to `archive/`:
- `PLAN.md` → `archive/PLAN.md` (original project plan, describes early tokenization scheme)
- `OPTIMIZATIONS_IMPLEMENTED.md` → `archive/` (batch size, XLA flags, data pipeline tuning)
- `OPTIMIZATION_NEXT_STEPS.md` → `archive/` (optimization roadmap)
- `PERFORMANCE_ISSUE_DIAGNOSIS.md` → `archive/` (modal batch size override bug)
- `PAPER_METRICS_PLAN.md` → `archive/` (evaluation plan)
- `REFACTORING_SUMMARY.md` → `archive/` (December 2024 refactoring)
- `MODAL_FIX.md` → `archive/` (modal deployment fixes)
