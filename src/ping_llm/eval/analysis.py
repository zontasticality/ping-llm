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
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ── Visual hierarchy ──────────────────────────────────────────────────────────

SIMPLE_BASELINES = {"global_median", "last_seen", "window_mean", "ema"}
STRUCTURED_BASELINES = {
    "vivaldi", "vivaldi_time", "dmfsgd", "dmfsgd_time",
    "dmfsgd_paper", "dmfsgd_paper_time", "biased_mf",
}

MODEL_STYLES = {
    "680m-200k-timeclean": {"label": "680m-200k", "color": "#0072B2"},
    "deep60-was-60k-timeclean": {"label": "deep60-was-60k", "color": "#D55E00"},
    "deep60-60k-timeclean": {"label": "deep60-60k", "color": "#009E73"},
}

_MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
_MODEL_COLOR_BY_BASE = {}
_model_color_idx = 0

CONTEXT_LINEWIDTHS = {
    0: 1.0,
    1: 1.45,
    2: 1.9,
    5: 2.45,
    10: 3.1,
}
DEFAULT_MODEL_LINEWIDTH = 2.2

STRUCTURED_STYLES = {
    "dmfsgd": {"color": "#000000", "linestyle": "--", "linewidth": 2.2},
    "dmfsgd_time": {"color": "#000000", "linestyle": "-.", "linewidth": 2.0},
    "vivaldi": {"color": "#CC79A7", "linestyle": "--", "linewidth": 2.1},
    "vivaldi_time": {"color": "#CC79A7", "linestyle": "-.", "linewidth": 1.9},
    "biased_mf": {"color": "#E69F00", "linestyle": "--", "linewidth": 2.0},
    "dmfsgd_paper": {"color": "#7f7f7f", "linestyle": "--", "linewidth": 1.6},
    "dmfsgd_paper_time": {"color": "#7f7f7f", "linestyle": "-.", "linewidth": 1.5},
}

SIMPLE_STYLES = {
    "global_median": {"color": "#6B6B6B", "linestyle": ":", "linewidth": 1.7, "alpha": 1.0},
    "last_seen": {"color": "#A6761D", "linestyle": ":", "linewidth": 1.7, "alpha": 1.0},
    "ema": {"color": "#7570B3", "linestyle": ":", "linewidth": 1.7, "alpha": 1.0},
    "window_mean": {"color": "#66A61E", "linestyle": ":", "linewidth": 1.7, "alpha": 1.0},
}

_CTX_RE = re.compile(r"^(?P<base>.+)_ctx(?P<context>\d+)$")


def _split_model_context(method_name):
    match = _CTX_RE.match(method_name)
    if not match:
        return method_name, None
    return match.group("base"), int(match.group("context"))


def _model_style_for(base_name):
    global _model_color_idx
    if base_name in MODEL_STYLES:
        return MODEL_STYLES[base_name]
    if base_name not in _MODEL_COLOR_BY_BASE:
        _MODEL_COLOR_BY_BASE[base_name] = _MODEL_COLORS[_model_color_idx % len(_MODEL_COLORS)]
        _model_color_idx += 1
    return {"label": base_name, "color": _MODEL_COLOR_BY_BASE[base_name]}


def _style_for(method_name):
    if method_name in SIMPLE_STYLES:
        return dict(SIMPLE_STYLES[method_name])
    if method_name in STRUCTURED_STYLES:
        return dict(STRUCTURED_STYLES[method_name])

    base_name, context = _split_model_context(method_name)
    model_style = _model_style_for(base_name)
    return {
        "color": model_style["color"],
        "linestyle": "-",
        "linewidth": CONTEXT_LINEWIDTHS.get(context, DEFAULT_MODEL_LINEWIDTH),
        "alpha": 0.95,
    }


def _reset_model_colors():
    global _model_color_idx, _MODEL_COLOR_BY_BASE
    _model_color_idx = 0
    _MODEL_COLOR_BY_BASE = {}


def _legend_label(method_name):
    if method_name in SIMPLE_BASELINES or method_name in STRUCTURED_BASELINES:
        return method_name
    base_name, _ = _split_model_context(method_name)
    return _model_style_for(base_name)["label"]


def _cdf_legend_handles(pred_cols):
    bases = []
    for col in pred_cols:
        label = col.replace("_pred", "")
        if label in SIMPLE_BASELINES or label in STRUCTURED_BASELINES:
            continue
        base, _ = _split_model_context(label)
        if base not in bases:
            bases.append(base)

    handles = []
    for base in bases:
        model_style = _model_style_for(base)
        handles.append(Line2D(
            [0], [0],
            color=model_style["color"],
            linestyle="-",
            linewidth=2.4,
            label=model_style["label"],
        ))

    baseline_order = [
        "dmfsgd", "vivaldi", "biased_mf", "dmfsgd_time", "vivaldi_time",
        "dmfsgd_paper", "dmfsgd_paper_time",
        "ema", "last_seen", "window_mean", "global_median",
    ]
    for name in baseline_order:
        if f"{name}_pred" not in pred_cols:
            continue
        style = _style_for(name)
        handles.append(Line2D(
            [0], [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=name,
            alpha=style.get("alpha", 1.0),
        ))
    return handles


def _context_legend_handles(pred_cols):
    contexts = sorted({
        context
        for col in pred_cols
        for _, context in [_split_model_context(col.replace("_pred", ""))]
        if context is not None
    })
    if not contexts:
        return []
    return [
        Line2D(
            [0], [0],
            color="#333333",
            linestyle="-",
            linewidth=CONTEXT_LINEWIDTHS.get(context, DEFAULT_MODEL_LINEWIDTH),
            label=f"ctx {context}",
        )
        for context in contexts
    ]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_merge(harness_dir, model_run_names=None):
    harness_dir = Path(harness_dir)
    obs = pd.read_parquet(harness_dir / "observations.parquet")
    if "obs_id" not in obs.columns:
        obs = obs.copy()
        obs["obs_id"] = np.arange(len(obs), dtype=np.int64)

    if model_run_names:
        for name in model_run_names:
            path = harness_dir / "model_preds" / f"{name}.parquet"
            if path.exists():
                mpreds = pd.read_parquet(path)
                pred_col = (
                    "model_top1_pred" if "model_top1_pred" in mpreds.columns else
                    "model_pred" if "model_pred" in mpreds.columns else
                    None
                )
                if pred_col is None:
                    print(f"Warning: {path} has no model prediction column, skipping")
                    continue

                if "obs_id" in mpreds.columns:
                    if "num_context" in mpreds.columns and mpreds["num_context"].nunique() > 1:
                        wide = mpreds.pivot_table(
                            index="obs_id",
                            columns="num_context",
                            values=pred_col,
                            aggfunc="first",
                        )
                        wide.columns = [f"{name}_ctx{int(c)}_pred" for c in wide.columns]
                        wide = wide.reset_index()
                    else:
                        out_col = f"{name}_pred"
                        if "num_context" in mpreds.columns and not mpreds.empty:
                            out_col = f"{name}_ctx{int(mpreds['num_context'].iloc[0])}_pred"
                        wide = mpreds[["obs_id", pred_col]].rename(columns={pred_col: out_col})
                    obs = obs.merge(wide, on="obs_id", how="left")
                    print(f"Joined {name}: {mpreds.shape[0]} predictions")
                elif {"seq_idx", "meas_idx"}.issubset(mpreds.columns):
                    mpreds = mpreds.rename(columns={pred_col: f"{name}_pred"})
                    obs = obs.merge(mpreds, on=["seq_idx", "meas_idx"], how="left")
                    print(f"Joined legacy {name}: {mpreds.shape[0]} predictions")
                else:
                    print(f"Warning: {path} has no supported join key, skipping")
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
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    err_suffix = f"__{metric}"
    ordered = _draw_order(pred_cols)

    for col in ordered:
        label = col.replace("_pred", "")
        err = df[f"{col}{err_suffix}"].dropna().sort_values().values
        if len(err) == 0:
            continue
        cdf = np.arange(1, len(err) + 1) / len(err)
        style = _style_for(label)
        ax.plot(err, cdf, label=_legend_label(label), **style)

    if xlabel is None:
        if metric == "rel_err":
            xlabel = "Relative Error  |pred - actual| / actual"
        else:
            xlabel = "Absolute Error (ms)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")

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

    method_handles = _cdf_legend_handles(pred_cols)
    context_handles = _context_legend_handles(pred_cols)
    if method_handles:
        method_legend = ax.legend(
            handles=method_handles,
            fontsize=7.5,
            loc="lower right",
            framealpha=0.92,
            ncol=2,
            title="Method / model family",
            title_fontsize=8,
        )
        ax.add_artist(method_legend)
    if context_handles:
        ax.legend(
            handles=context_handles,
            fontsize=7.5,
            loc="center right",
            bbox_to_anchor=(1.0, 0.52),
            framealpha=0.92,
            title="Model context\n(line width)",
            title_fontsize=8,
        )

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

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
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
        ax.plot(bins, med, label=_legend_label(label), marker=marker, markersize=4, **style)
        ax.fill_between(bins, lo, hi, color=style["color"], alpha=fill_alpha)

    ax.set_xlabel("Prior RTT observations in context")
    ax.set_ylabel("Median Absolute Error (ms)")
    method_handles = _cdf_legend_handles(pred_cols)
    context_handles = _context_legend_handles(pred_cols)
    if method_handles:
        method_legend = ax.legend(
            handles=method_handles,
            fontsize=7.5,
            loc="upper right",
            framealpha=0.92,
            ncol=2,
            title="Method / model family",
            title_fontsize=8,
        )
        ax.add_artist(method_legend)
    if context_handles:
        ax.legend(
            handles=context_handles,
            fontsize=7.5,
            loc="center right",
            bbox_to_anchor=(1.0, 0.48),
            framealpha=0.92,
            title="Model context\n(line width)",
            title_fontsize=8,
        )
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
