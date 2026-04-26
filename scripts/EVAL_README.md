# Evaluation Guide

## Three-Stage Pipeline (primary)

The main evaluation workflow is a three-stage pipeline. Each stage can be
re-run independently.

### Stage 1: Harness (baselines)

Runs once per test dataset. Extracts ground truth, trains all baselines
(MF, Vivaldi, TRMF), caches predictions to parquet.

```bash
python -m ping_llm.eval.harness \
    --test-data data/probe_rows/test.arrayrecord \
    --output-dir outputs/eval_harness/default \
    --num-sequences 200
```

On SLURM (GPU node, also runs stages 2+3):
```bash
sbatch scripts/train/slurm_eval_pipeline.sh
```

### Stage 2: Model eval (per checkpoint)

Forward pass on test sequences, extracts top-1 RTT predictions, logprobs,
and per-token-type loss breakdown.

```bash
python -m ping_llm.eval.model_eval \
    --checkpoint outputs/checkpoints/deep60-60k/latest.pt \
    --test-data data/probe_rows/test.arrayrecord \
    --harness-dir outputs/eval_harness/default \
    --run-name deep60-60k
```

No-timestamp ablation (strips timestamps before forward pass):
```bash
python -m ping_llm.eval.model_eval \
    --checkpoint outputs/checkpoints/deep60-60k/latest.pt \
    --test-data data/probe_rows/test.arrayrecord \
    --harness-dir outputs/eval_harness/default \
    --run-name deep60-60k-nots \
    --strip-timestamps
```

### Stage 3: Analysis (local, no GPU)

Joins harness + model predictions, generates CDF figures, context curves,
and percentile tables.

```bash
python -m ping_llm.eval.analysis \
    --harness-dir outputs/eval_harness/default \
    --model-runs deep60-60k,680m-200k,deep60-60k-nots,680m-200k-nots
```

Outputs:
- `outputs/figures/cdf_rel_err.pdf` / `cdf_rel_err_log.pdf` — relative error CDFs
- `outputs/figures/cdf_abs_err_ms.pdf` / `cdf_abs_err_ms_log.pdf` — absolute error CDFs
- `outputs/figures/context_curve.pdf` — accuracy vs prior context
- `outputs/tables/percentile_table.csv`
- `outputs/tables/loss_breakdown.csv`

### When to re-run each stage

| Change | Re-run |
|--------|--------|
| Test data or baseline config | Stage 1 + 2 + 3 |
| New model checkpoint | Stage 2 + 3 |
| New figure or metric | Stage 3 only |

## Legacy Scripts (supplemental)

These scripts predate the pipeline and provide interactive/diagnostic tools:

- `scripts/eval_ordering_likelihood.py` — test field-ordering preferences
- `scripts/eval_next_token_predictions.py` — pretty-print predicted vs actual tokens
- `scripts/eval_paper_metrics.py` — archived to `archive/scripts/` (superseded by pipeline)
- `scripts/eval_live_ping.py` — removed (superseded by `ping_llm.eval.history_ping`)

### Field ordering likelihood

```bash
python scripts/eval_ordering_likelihood.py \
    --checkpoint path/to/latest.pt \
    --data data/training_data.parquet \
    --num-samples 100
```

### Next-token prediction display

```bash
python scripts/eval_next_token_predictions.py \
    --checkpoint path/to/latest.pt \
    --data data/probe_rows/test.arrayrecord \
    --num-sequences 5
```

## Unified runner (legacy)

The old `python -m ping_llm.eval.run_all` still works for quick checks:

```bash
python -m ping_llm.eval.run_all \
    --checkpoint path/to/latest.pt \
    --test-data data/probe_rows/test.arrayrecord \
    --tests loss_breakdown,baselines
```
