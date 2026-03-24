# Run Status Tracker

## Active Runs
_No active runs._

## Validation Ladder Results (2026-03-24)

All tiers validated with data pipeline fix applied.

| Tier | Run Name | Result | Key Metrics |
|------|----------|--------|-------------|
| 0: Local sanity | — | PASS | 98.6M params, tokenization OK |
| 1: Data pipeline | val-pipeline-v3 | PASS | 11k tok/s (w/warmup), ~35k steady |
| 2: Forward pass | (piggybacked on Tier 1) | PASS | Loss 5.63, GPU 30GB |
| 3: Compile warmup | val-compile-95m | PASS | Compile ~60s (not 23 min!) |
| 4: Short training | val-short-95m | PASS | **51k tok/s**, loss 5.81→2.86 |

### Key Discovery
The "23 min torch.compile" was actually data pipeline latency. Real compile for 95M on A100 is ~60s.

## Architecture Experiment Results (2026-03-24)

Tested nanochat architecture changes. Bundled test showed changes need individual validation:

| Run | Changes | Train Loss @200 | Eval Loss @200 | tok/s |
|-----|---------|-----------------|----------------|-------|
| val-short-95m (baseline) | None | 2.86 | 3.02 | 51k |
| arch-no-resid | Fused QKV + zero init | 3.11 | 3.17 | 47k |
| arch-all | + resid scalars | 3.05 | 3.20 | 47k |

**Conclusion**: Bundled changes hurt at 200 steps. Need autoresearch-style individual testing.

## Autoresearch Status

Script ready at `scripts/autoresearch.py`. Three debugging runs fixed bugs:
1. LR=0 with `--total-steps 0` (fixed: use large total_steps)
2. Stale checkpoint resume (fixed: use /tmp/ checkpoint dir)
3. Compile time confounding results (fixed: reset timer after first step)
4. Script timeout too short (fixed: +360s grace for compile)

**Ready for clean run** — estimated cost ~$1.50 for 6 experiments.

## Budget Tracking

| Date | Category | Cost |
|------|----------|------|
| 2026-03-24 | Validation ladder (Tiers 1-4) | $3.27 |
| 2026-03-24 | Pipeline diagnostic | $0.51 |
| 2026-03-24 | Architecture experiments | $0.76 |
| 2026-03-24 | Autoresearch debugging (3 runs) | $3.87 |
| | **Total** | **$8.42** |

## Past Runs (2026-03-09)

### 95m-full attempt 2 (FAILED)
- **Config**: 95M, BS=32, 14000 steps, A100-SXM4-40GB, compile
- **Status**: CLI disconnected, container cancelled at step 30
- **tok/s**: 1,076 (data pipeline bottleneck, pre-fix)

### 95m-full attempt 1 (FAILED)
- **Config**: 95M, BS=32, 14000 steps, A100-80GB, compile
- **Status**: Container killed ~30 min in, 0 steps completed
- **Cause**: CLI disconnected during compile

---

## Launch Checklist
- [ ] Always use `modal run --detach`
- [ ] Compile for 95M takes ~60s on A100 (not 23 min)
- [ ] Check costs: `python scripts/modal_usage.py --hours 24`
- [ ] Check status: `modal app list` / `modal container list`
- [ ] Retrieve logs: `modal volume get ping-llm logs/<run-name>.log`
