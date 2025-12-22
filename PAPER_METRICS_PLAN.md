# Paper Metrics Plan

## Goals
- Produce PNG plots that visualize model likelihoods and behavior in three focused evaluations.
- Keep evaluation self-contained (no ASN/geo datasets) and runnable on Modal with param-only checkpoints.

## Libraries
- matplotlib for plots (PNG output)
- numpy for histograms and stats
- pandas/pyarrow for parquet sampling
- duckdb for fast sampling from large parquet files
- scipy.stats.entropy for KL divergence
- jax/MaxText MaxEngine for prompt logprobs

## Outputs
- Metrics saved to `outputs/paper_metrics/default/metrics/`
- Figures saved to `outputs/paper_metrics/default/figures/`

## Evaluation 1: Timestamp vs No-Timestamp Likelihood (RTT only)
- Sample rows from `data/training_data.parquet`.
- Build two contexts per row:
  - `full`: MEASUREMENT_START + src + dst + timestamp, then predict RTT tokens.
  - `none`: MEASUREMENT_START + src + dst, then predict RTT tokens.
- Compute per-token log-probabilities only over the RTT token block using `MaxEngine.prefill(..., return_prompt_logp=True)`.
- Plot a histogram of log P(correct token) for `full` vs `none`.
- Save figure: `timestamp_logprob_hist.png`.

## Evaluation 2: Prediction-Mode Accuracy Differences
- Sample measurements from `data/training_data.parquet`.
- For each measurement, construct four variants where one field is last:
  - `src`, `dst`, `rtt`, `timestamp`.
- For each variant, compute the likelihood of the exact expected field tokens.
- Group by field (and IP version) and compute per-sub-token mean probability.
- Figures:
  - Summary bar chart across groups (mean P(correct token)).
  - Per-group token-text bar charts using an example token text as labels.
- Save figures:
  - `mode_accuracy_summary.png`
  - `mode_<group>_token_accuracy.png`

## Evaluation 3: Live Ping Distribution Match
- Use a fixed default list of domains that resolve to stable IPs (e.g., umass.edu, hampshire.edu).
- Resolve domains to IPv4 addresses and ping each target N times.
- For each target:
  - Collect N real RTTs via `ping`.
  - Sample N RTTs from the model conditioned on `(src, dst)` (timestamp optional).
  - Plot overlaid histograms and label KL divergence on the figure.
- Save figures:
  - `live_ping_grid.png` (or `live_ping_grid_XX.png` when split across pages)
- Report average KL divergence across all successful targets.

## Implementation Steps
1. Add `scripts/eval_paper_metrics.py` (collector):
   - CLI controls for each evaluation and output directory.
   - Sampling routines for Parquet.
   - Prompt log-prob extraction helper using MaxEngine.
   - Live ping evaluation with domain-resolved targets.
   - Emit one metrics JSON per evaluation in `outputs/.../metrics/`.
2. Add `scripts/eval_paper_metrics_plot.py` (plotter):
   - Load metrics JSONs and generate PNGs.
   - Split large ping grids into multiple pages when needed.
3. Update `scripts/EVAL_README.md` with a short usage example and output description.

## CLI Sketch
- `python scripts/eval_paper_metrics.py \
    --output-dir outputs/paper_metrics/run_001 \
    --timestamp-contexts 200 \
    --mode-samples 200 \
    --pings-per-ip 20 \
    --model-samples 100`
- `python scripts/eval_paper_metrics_plot.py \
    --run-dir outputs/paper_metrics/run_001`

## Notes
- Timestamp evaluation focuses only on RTT tokens (partial measurement conditional).
- All plots are PNG as requested.
