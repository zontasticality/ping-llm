"""
Stage 3: Analysis and figure generation.

Joins harness observations with model predictions, computes error metrics,
generates CDF figures, context-conditioning curves, and percentile tables.

Usage:
    python -m ping_llm.eval.analysis \
        --harness-dir outputs/eval_harness/default \
        --model-runs deep60-60k,680m-200k,deep60-60k-nots,680m-200k-nots
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ── Visual hierarchy ──────────────────────────────────────────────────────────

SIMPLE_BASELINES = {"global_median", "last_seen", "window_mean", "ema"}
STRUCTURED_BASELINES = {"vivaldi", "dmfsgd", "biased_mf"}

_MODEL_COLORS = [
    "#000000", "#2c2c2c", "#555555", "#777777",
]

STRUCTURED_STYLES = {
    "vivaldi":   {"color": "#9467bd", "linestyle": "--", "linewidth": 1.8},
    "dmfsgd":    {"color": "#8c564b", "linestyle": "--", "linewidth": 1.8},
    "biased_mf": {"color": "#d62728", "linestyle": "--", "linewidth": 1.8},
}

SIMPLE_STYLES = {
    "global_median": {"color": "#98df8a", "linestyle": ":", "linewidth": 1.0, "alpha": 0.55},
    "last_seen":     {"color": "#ffbb78", "linestyle": ":", "linewidth": 1.0, "alpha": 0.55},
    "ema":           {"color": "#aec7e8", "linestyle": ":", "linewidth": 1.0, "alpha": 0.55},
    "window_mean":   {"color": "#c7c7c7", "linestyle": ":", "linewidth": 1.0, "alpha": 0.55},
}

_model_color_idx = 0


def _style_for(method_name):
    global _model_color_idx
    if method_name in SIMPLE_STYLES:
        return SIMPLE_STYLES[method_name]
    if method_name in STRUCTURED_STYLES:
        return STRUCTURED_STYLES[method_name]
    nots = method_name.endswith("-nots")
    c = _MODEL_COLORS[_model_color_idx % len(_MODEL_COLORS)]
    _model_color_idx += 1
    return {
        "color": c,
        "linestyle": "--" if nots else "-",
        "linewidth": 2.5,
    }


def _reset_model_colors():
    global _model_color_idx
    _model_color_idx = 0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_merge(harness_dir, model_run_names=None):
    harness_dir = Path(harness_dir)
    obs = pd.read_parquet(harness_dir / "observations.parquet")

    if model_run_names:
        for name in model_run_names:
            path = harness_dir / "model_preds" / f"{name}.parquet"
            if path.exists():
                mpreds = pd.read_parquet(path)
                mpreds = mpreds.rename(columns={"model_top1_pred": f"{name}_pred"})
                obs = obs.merge(mpreds, on=["seq_idx", "meas_idx"], how="left")
                print(f"Joined {name}: {mpreds.shape[0]} predictions")
            else:
                print(f"Warning: {path} not found, skipping")

    return obs


def get_pred_columns(df):
    return [c for c in df.columns if c.endswith("_pred")]


def compute_errors(df, pred_cols):
    actual = df["actual_rtt_ms"]
    for col in pred_cols:
        df[f"{col}__rel_err"] = np.abs(df[col] - actual) / actual
        df[f"{col}__abs_err_ms"] = np.abs(df[col] - actual)
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_order(pred_cols):
    """Return pred_cols sorted so models draw last (on top)."""
    def key(col):
        name = col.replace("_pred", "")
        if name in SIMPLE_BASELINES:
            return (0, name)
        if name in STRUCTURED_BASELINES:
            return (1, name)
        return (2, name)
    return sorted(pred_cols, key=key)


def plot_cdf(df, pred_cols, output_dir, log_scale=False, metric="rel_err",
             xlabel=None, xlim=None, suffix_extra=""):
    _reset_model_colors()
    fig, ax = plt.subplots(figsize=(8, 5))

    err_suffix = f"__{metric}"
    ordered = _draw_order(pred_cols)

    for col in ordered:
        label = col.replace("_pred", "")
        err = df[f"{col}{err_suffix}"].dropna().sort_values().values
        if len(err) == 0:
            continue
        cdf = np.arange(1, len(err) + 1) / len(err)
        style = _style_for(label)
        ax.plot(err, cdf, label=label, **style)

    if xlabel is None:
        if metric == "rel_err":
            xlabel = "Relative Error  |pred - actual| / actual"
        else:
            xlabel = "Absolute Error (ms)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    if log_scale:
        ax.set_xscale("log")
        if xlim is None:
            xlim = (1e-3, 1e2)
        ax.set_xlim(*xlim)
    else:
        if xlim is None:
            xlim = (0, 3)
        ax.set_xlim(*xlim)

    ax.set_ylim(0, 1.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_log" if log_scale else ""
    path = output_dir / f"cdf_{metric}{suffix}{suffix_extra}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _bootstrap_median_ci(series, n_boot=1000, ci=0.95, rng=None):
    """Bootstrap confidence interval for the median."""
    if rng is None:
        rng = np.random.RandomState(42)
    vals = series.dropna().values
    if len(vals) < 3:
        m = float(np.median(vals)) if len(vals) else np.nan
        return m, m, m
    medians = np.array([
        np.median(rng.choice(vals, size=len(vals), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(medians, 100 * alpha)), \
           float(np.median(vals)), \
           float(np.percentile(medians, 100 * (1 - alpha)))


def plot_context_curve(df, pred_cols, output_dir, max_k=10):
    """Median absolute error (ms) vs number of prior RTT observations."""
    _reset_model_colors()
    df = df.copy()
    if "prior_rtts" not in df.columns:
        df["prior_rtts"] = df.groupby("seq_idx").cumcount()
    df["prior_rtts_bin"] = df["prior_rtts"].clip(upper=max_k)

    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = _draw_order(pred_cols)
    rng = np.random.RandomState(42)

    for col in ordered:
        label = col.replace("_pred", "")
        err_col = f"{col}__abs_err_ms"
        if err_col not in df.columns:
            continue

        bins = sorted(df["prior_rtts_bin"].unique())
        lo, med, hi = [], [], []
        for b in bins:
            mask = df["prior_rtts_bin"] == b
            l, m, h = _bootstrap_median_ci(df.loc[mask, err_col], rng=rng)
            lo.append(l); med.append(m); hi.append(h)

        style = _style_for(label)
        marker = "o" if label not in SIMPLE_BASELINES else ""
        fill_alpha = style.pop("alpha", 1.0) * 0.15
        ax.plot(bins, med, label=label, marker=marker, markersize=4, **style)
        ax.fill_between(bins, lo, hi, color=style["color"], alpha=fill_alpha)

    ax.set_xlabel("Prior RTT observations in context")
    ax.set_ylabel("Median Absolute Error (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ticks = list(range(max_k + 1))
    tick_labels = [str(t) for t in ticks]
    tick_labels[-1] = f"{max_k}+"
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "context_curve.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


# ── Tables ────────────────────────────────────────────────────────────────────

def percentile_table(df, pred_cols):
    rows = []
    for col in sorted(pred_cols):
        label = col.replace("_pred", "")
        rel = df[f"{col}__rel_err"].dropna()
        abs_e = df[f"{col}__abs_err_ms"].dropna()
        if len(rel) == 0:
            continue
        rows.append({
            "method": label,
            "count": len(rel),
            "mae_ms": round(float(abs_e.mean()), 2),
            "median_ae_ms": round(float(abs_e.median()), 2),
            "p50_rel": round(float(rel.quantile(0.50)), 4),
            "p75_rel": round(float(rel.quantile(0.75)), 4),
            "p90_rel": round(float(rel.quantile(0.90)), 4),
            "p95_rel": round(float(rel.quantile(0.95)), 4),
        })
    return pd.DataFrame(rows)


def loss_breakdown_table(harness_dir, model_run_names):
    rows = []
    if not model_run_names:
        return pd.DataFrame()
    for name in model_run_names:
        path = Path(harness_dir) / "model_preds" / f"{name}_loss_breakdown.json"
        if not path.exists():
            continue
        lb = json.loads(path.read_text())
        for token_type, metrics in lb.items():
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "model": name,
                "token_type": token_type,
                "count": metrics.get("count", 0),
                "mean_ce": metrics.get("mean_ce", float("nan")),
                "accuracy": metrics.get("accuracy", float("nan")),
            })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(harness_dir="outputs/eval_harness/default",
                 model_runs=None, output_dir=None):
    harness_dir = Path(harness_dir)
    if output_dir is None:
        output_dir = Path("outputs")
    else:
        output_dir = Path(output_dir)

    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    df = load_and_merge(harness_dir, model_runs)
    pred_cols = get_pred_columns(df)
    print(f"\n{len(df)} observations, {len(pred_cols)} methods: "
          f"{[c.replace('_pred', '') for c in sorted(pred_cols)]}")

    df = compute_errors(df, pred_cols)

    # Relative error CDFs
    path = plot_cdf(df, pred_cols, fig_dir, log_scale=False, metric="rel_err")
    print(f"CDF (relative, linear) → {path}")
    path = plot_cdf(df, pred_cols, fig_dir, log_scale=True, metric="rel_err")
    print(f"CDF (relative, log)    → {path}")

    # Absolute error CDFs (ms)
    path = plot_cdf(df, pred_cols, fig_dir, log_scale=False, metric="abs_err_ms",
                    xlabel="Absolute Error (ms)", xlim=(0, 200))
    print(f"CDF (absolute, linear) → {path}")
    path = plot_cdf(df, pred_cols, fig_dir, log_scale=True, metric="abs_err_ms",
                    xlabel="Absolute Error (ms)", xlim=(0.01, 1e4))
    print(f"CDF (absolute, log)    → {path}")

    # Context conditioning curve
    path = plot_context_curve(df, pred_cols, fig_dir)
    print(f"Context curve          → {path}")

    # Percentile table
    table = percentile_table(df, pred_cols)
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "percentile_table.csv"
    table.to_csv(table_path, index=False)
    print(f"\nPercentile table → {table_path}")
    print(table.to_string(index=False))

    # Loss breakdown table
    lb = loss_breakdown_table(harness_dir, model_runs)
    if not lb.empty:
        lb_path = table_dir / "loss_breakdown.csv"
        lb.to_csv(lb_path, index=False)
        print(f"\nLoss breakdown → {lb_path}")
        print(lb.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Stage 3: analysis and figures")
    p.add_argument("--harness-dir", default="outputs/eval_harness/default")
    p.add_argument("--model-runs", default=None,
                   help="Comma-separated model run names")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    model_runs = args.model_runs.split(",") if args.model_runs else None
    run_analysis(args.harness_dir, model_runs, args.output_dir)


if __name__ == "__main__":
    main()
