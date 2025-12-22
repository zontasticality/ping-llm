#!/usr/bin/env python3
"""
Plot paper metrics from JSON outputs produced by eval_paper_metrics.py.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("seaborn-v0_8")

DEFAULT_RUN_DIR = Path("outputs/paper_metrics/default")


def parse_only_arg(value):
    items = {item.strip().lower() for item in value.split(",") if item.strip()}
    if "all" in items or not items:
        return {"timestamps", "modes", "ping"}
    return items


def clear_dir(path, suffixes=None):
    if not path.exists():
        return
    for child in path.iterdir():
        if not child.is_file():
            continue
        if suffixes is None or child.suffix in suffixes:
            child.unlink()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_logps(logps):
    if len(logps) == 0:
        return {
            "count": 0,
            "mean_logp": float("nan"),
            "median_logp": float("nan"),
            "mean_prob": float("nan"),
            "median_prob": float("nan"),
        }
    probs = np.exp(logps)
    return {
        "count": int(len(logps)),
        "mean_logp": float(np.mean(logps)),
        "median_logp": float(np.median(logps)),
        "mean_prob": float(np.mean(probs)),
        "median_prob": float(np.median(probs)),
    }


def build_hist_bins(logps_full, logps_none, hist_bins):
    all_logps = (
        np.concatenate([logps_full, logps_none])
        if len(logps_full) and len(logps_none)
        else np.array([])
    )
    if all_logps.size == 0:
        return hist_bins
    low, high = np.percentile(all_logps, [1, 99])
    if low == high:
        low, high = low - 1.0, high + 1.0
    if isinstance(hist_bins, (list, tuple, np.ndarray)):
        return np.array(hist_bins)
    return np.linspace(low, high, int(hist_bins))


def plot_logprob_hist(output_path, logps_full, logps_none, bins, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(logps_full) > 0:
        ax.hist(
            logps_full,
            bins=bins,
            alpha=0.6,
            label="full timestamps",
            density=True,
        )
    if len(logps_none) > 0:
        ax.hist(
            logps_none,
            bins=bins,
            alpha=0.6,
            label="no timestamps",
            density=True,
        )
    ax.set_xlabel("log P(correct token)")
    ax.set_ylabel("density")
    ax.set_title(title)
    if len(logps_full) > 0 or len(logps_none) > 0:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_logprob_hist_with_cdf(
    output_path,
    logps_full,
    logps_none,
    bins,
    title,
    clip_percentiles=(2, 98),
):
    fig, ax = plt.subplots(figsize=(8, 5))
    color_full = "#4C78A8"
    color_none = "#72B7B2"
    all_logps = (
        np.concatenate([logps_full, logps_none])
        if len(logps_full) and len(logps_none)
        else np.array([])
    )
    x_min = None
    x_max = None
    if all_logps.size:
        low_pct, high_pct = clip_percentiles
        x_min, x_max = np.percentile(all_logps, [low_pct, high_pct])
        if x_min == x_max:
            x_min, x_max = x_min - 1.0, x_max + 1.0
    elif isinstance(bins, (list, tuple, np.ndarray)) and len(bins) >= 2:
        x_min = float(bins[0])
        x_max = float(bins[-1])
    if x_min is not None and x_max is not None:
        bins = np.linspace(x_min, x_max, len(bins) if isinstance(bins, (list, tuple, np.ndarray)) else int(bins))

    if len(logps_full) > 0:
        clipped_full = (
            logps_full[(logps_full >= x_min) & (logps_full <= x_max)]
            if x_min is not None and x_max is not None
            else logps_full
        )
        ax.hist(
            clipped_full,
            bins=bins,
            alpha=0.45,
            label="full timestamps",
            density=True,
            color=color_full,
        )
    if len(logps_none) > 0:
        clipped_none = (
            logps_none[(logps_none >= x_min) & (logps_none <= x_max)]
            if x_min is not None and x_max is not None
            else logps_none
        )
        ax.hist(
            clipped_none,
            bins=bins,
            alpha=0.45,
            label="no timestamps",
            density=True,
            color=color_none,
        )
    ax.set_xlabel("log P(correct token)")
    ax.set_ylabel("density (normalized histogram)")
    ax.set_title(title)
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)

    ax2 = ax.twinx()
    if len(logps_full) > 0:
        clipped_full = (
            logps_full[(logps_full >= x_min) & (logps_full <= x_max)]
            if x_min is not None and x_max is not None
            else logps_full
        )
        sorted_full = np.sort(clipped_full)
        cdf_full = np.linspace(0, 1, len(sorted_full))
        ax2.plot(
            sorted_full,
            cdf_full,
            color=color_full,
            linewidth=1.5,
            label="full CDF",
        )
    if len(logps_none) > 0:
        clipped_none = (
            logps_none[(logps_none >= x_min) & (logps_none <= x_max)]
            if x_min is not None and x_max is not None
            else logps_none
        )
        sorted_none = np.sort(clipped_none)
        cdf_none = np.linspace(0, 1, len(sorted_none))
        ax2.plot(
            sorted_none,
            cdf_none,
            color=color_none,
            linewidth=1.5,
            label="no timestamp CDF",
        )
    ax2.set_ylabel("CDF (fraction of tokens with logP ≤ x)")
    ax2.set_ylim(0, 1.0)
    if x_min is not None and x_max is not None:
        ax2.set_xlim(x_min, x_max)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2, loc="upper left")
    if x_min is not None and x_max is not None:
        ax.text(
            0.99,
            0.02,
            f"range clipped to {clip_percentiles[0]}–{clip_percentiles[1]}th pct",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_timestamp_effects(
    output_path,
    logps_full,
    logps_none,
    hist_bins,
    thresholds=None,
):
    if thresholds is None:
        thresholds = [0.5, 0.8, 0.95]

    full_logps = np.array(logps_full, dtype=np.float32)
    none_logps = np.array(logps_none, dtype=np.float32)
    if full_logps.size == 0 or none_logps.size == 0:
        print("Timestamp plot missing full/none logps; skipping.")
        return None

    paired_size = min(full_logps.size, none_logps.size)
    if full_logps.size != none_logps.size:
        print(
            "Timestamp logps size mismatch; truncating to paired size "
            f"{paired_size}."
        )
    delta_logps = full_logps[:paired_size] - none_logps[:paired_size]

    if delta_logps.size:
        low, high = np.percentile(delta_logps, [1, 99])
        if low == high:
            low, high = low - 1.0, high + 1.0
        if isinstance(hist_bins, (list, tuple, np.ndarray)):
            bins = np.array(hist_bins)
        else:
            bins = np.linspace(low, high, int(hist_bins))
    else:
        bins = hist_bins

    full_summary = summarize_logps(full_logps)
    none_summary = summarize_logps(none_logps)
    mean_delta = float(np.mean(delta_logps)) if delta_logps.size else float("nan")
    median_delta = float(np.median(delta_logps)) if delta_logps.size else float("nan")
    ratio = math.exp(mean_delta) if not math.isnan(mean_delta) else float("nan")
    pct_improved = (
        float(np.mean(delta_logps > 0) * 100) if delta_logps.size else float("nan")
    )

    full_probs = np.exp(full_logps)
    none_probs = np.exp(none_logps)
    full_fracs = [
        float(np.mean(full_probs >= threshold)) if full_probs.size else float("nan")
        for threshold in thresholds
    ]
    none_fracs = [
        float(np.mean(none_probs >= threshold)) if none_probs.size else float("nan")
        for threshold in thresholds
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.6),
        gridspec_kw={"width_ratios": [1.4, 1.0, 1.2]},
    )

    ax = axes[0]
    ax.hist(delta_logps, bins=bins, density=True, alpha=0.8, color="#4C78A8")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("delta log P(correct token) (full - none)")
    ax.set_ylabel("density")
    ax.set_title("Delta logP distribution")

    ax = axes[1]
    ax.axis("off")
    summary_lines = ["Summary (RTT tokens)"]
    if full_logps.size != none_logps.size:
        summary_lines.append(
            f"N full / none: {full_logps.size:,} / {none_logps.size:,}"
        )
    summary_lines.extend(
        [
            f"N paired: {paired_size:,}",
            f"mean logP full: {full_summary['mean_logp']:.3f}",
            f"mean logP none: {none_summary['mean_logp']:.3f}",
            f"mean dlogP: {mean_delta:+.3f} (x{ratio:.2f})",
            f"median dlogP: {median_delta:+.3f}",
            f"dlogP > 0: {pct_improved:.1f}%",
        ]
    )
    ax.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    ax = axes[2]
    x = np.arange(len(thresholds))
    width = 0.38
    ax.bar(x - width / 2, full_fracs, width, label="full timestamps", color="#4C78A8")
    ax.bar(x + width / 2, none_fracs, width, label="no timestamps", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p>={t:.2f}" for t in thresholds])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of tokens")
    ax.set_title("Share above thresholds")
    ax.legend()

    fig.suptitle("Timestamp impact on RTT token likelihood")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_timestamp_threshold_cdf(output_path, logps_full, logps_none, num_points=200):
    full_logps = np.array(logps_full, dtype=np.float32)
    none_logps = np.array(logps_none, dtype=np.float32)
    if full_logps.size == 0 or none_logps.size == 0:
        print("Timestamp CDF missing full/none logps; skipping.")
        return None

    full_probs = np.exp(full_logps)
    none_probs = np.exp(none_logps)
    thresholds = np.linspace(0.0, 1.0, num_points)

    full_curve = np.array([np.mean(full_probs >= t) for t in thresholds])
    none_curve = np.array([np.mean(none_probs >= t) for t in thresholds])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, full_curve, label="full timestamps", color="#4C78A8")
    ax.plot(thresholds, none_curve, label="no timestamps", color="#72B7B2")
    ax.set_xlabel("probability threshold P(correct token)")
    ax.set_ylabel("fraction of tokens with P >= threshold")
    ax.set_title("Continuous threshold curve (1 - CDF)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_timestamp_buckets(
    output_path,
    logps_full,
    logps_none,
    full_rtt_ms,
    none_rtt_ms,
    full_event_hour,
    none_event_hour,
):
    full_logps = np.array(logps_full, dtype=np.float32)
    none_logps = np.array(logps_none, dtype=np.float32)
    full_rtt = np.array(full_rtt_ms, dtype=np.float32)
    none_rtt = np.array(none_rtt_ms, dtype=np.float32)
    full_hour = np.array(full_event_hour, dtype=np.float32)
    none_hour = np.array(none_event_hour, dtype=np.float32)

    if (
        full_logps.size == 0
        or none_logps.size == 0
        or full_rtt.size == 0
        or none_rtt.size == 0
        or full_hour.size == 0
        or none_hour.size == 0
    ):
        print("Timestamp buckets missing logps or metadata; skipping.")
        return None

    full_len = min(full_logps.size, full_rtt.size, full_hour.size)
    none_len = min(none_logps.size, none_rtt.size, none_hour.size)
    if full_logps.size != full_len or none_logps.size != none_len:
        print("Timestamp buckets length mismatch; truncating to aligned sizes.")
    full_logps = full_logps[:full_len]
    full_rtt = full_rtt[:full_len]
    full_hour = full_hour[:full_len]
    none_logps = none_logps[:none_len]
    none_rtt = none_rtt[:none_len]
    none_hour = none_hour[:none_len]

    rtt_bins = np.array([0, 10, 20, 50, 100, 200, 500, 1000, np.inf], dtype=float)
    hour_bins = np.array([0, 4, 8, 12, 16, 20, 24], dtype=float)

    def bucket_means(logps, values, bins):
        probs = np.exp(logps)
        means = []
        counts = []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (values >= low) & (values < high)
            count = int(np.sum(mask))
            counts.append(count)
            means.append(float(np.mean(probs[mask])) if count else float("nan"))
        return means, counts

    full_rtt_mask = np.isfinite(full_rtt) & (full_rtt >= 0)
    none_rtt_mask = np.isfinite(none_rtt) & (none_rtt >= 0)
    full_hour_mask = full_rtt_mask & np.isfinite(full_hour)
    none_hour_mask = none_rtt_mask & np.isfinite(none_hour)

    full_rtt_means, _ = bucket_means(
        full_logps[full_rtt_mask], full_rtt[full_rtt_mask], rtt_bins
    )
    none_rtt_means, _ = bucket_means(
        none_logps[none_rtt_mask], none_rtt[none_rtt_mask], rtt_bins
    )
    full_hour_means, _ = bucket_means(
        full_logps[full_hour_mask], full_hour[full_hour_mask], hour_bins
    )
    none_hour_means, _ = bucket_means(
        none_logps[none_hour_mask], none_hour[none_hour_mask], hour_bins
    )

    rtt_labels = []
    for low, high in zip(rtt_bins[:-1], rtt_bins[1:]):
        if np.isinf(high):
            label = f"{int(low)}+"
        else:
            label = f"{int(low)}-{int(high)}"
        rtt_labels.append(label)
    hour_labels = [
        f"{int(low):02d}-{int(high):02d}"
        for low, high in zip(hour_bins[:-1], hour_bins[1:])
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    x_rtt = np.arange(len(rtt_labels))
    axes[0].plot(
        x_rtt, full_rtt_means, marker="o", color="#4C78A8", label="full timestamps"
    )
    axes[0].plot(
        x_rtt, none_rtt_means, marker="o", color="#72B7B2", label="no timestamps"
    )
    axes[0].set_xticks(x_rtt)
    axes[0].set_xticklabels(rtt_labels, rotation=30, ha="right")
    axes[0].set_ylabel("mean P(correct token)")
    axes[0].set_title("By RTT magnitude (ms)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[0].text(
        0.99,
        0.05,
        "failed RTTs excluded",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=8,
    )

    x_hour = np.arange(len(hour_labels))
    axes[1].plot(
        x_hour, full_hour_means, marker="o", color="#4C78A8", label="full timestamps"
    )
    axes[1].plot(
        x_hour, none_hour_means, marker="o", color="#72B7B2", label="no timestamps"
    )
    axes[1].set_xticks(x_hour)
    axes[1].set_xticklabels(hour_labels, rotation=30, ha="right")
    axes[1].set_ylabel("mean P(correct token)")
    axes[1].set_title("By time-of-day bucket (hour)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].text(
        0.99,
        0.05,
        "failed RTTs excluded",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=8,
    )

    fig.suptitle("Timestamp impact by RTT magnitude and time of day")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_mode_summary(output_path, summary, order):
    groups = [g for g in order if g in summary and summary[g]["count"] > 0]
    values = [summary[g]["mean_prob"] for g in groups]
    max_val = max(values) if values else 1.0
    if max_val <= 0:
        max_val = 1.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(groups, values, color="#4C78A8")
    ax.set_ylabel("mean P(correct token)")
    ax.set_title("Prediction-mode accuracy (mean correct-token probability)")
    ax.set_ylim(0, max_val * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_token_accuracy_bar(output_path, token_labels, mean_probs, title):
    positions = np.arange(len(mean_probs))
    max_val = max(mean_probs) if mean_probs else 1.0
    if max_val <= 0:
        max_val = 1.0
    fig, ax = plt.subplots(figsize=(max(6, len(mean_probs) * 0.6), 4.5))
    ax.bar(positions, mean_probs, color="#F58518")
    ax.set_xticks(positions)
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_ylabel("mean P(correct token)")
    ax.set_title(title)
    ax.set_ylim(0, max_val * 1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_rtt_range(real_vals, model_vals, max_rtt):
    combined = [v for v in list(real_vals) + list(model_vals) if v >= 0]
    if not combined:
        return 0.0, float(max_rtt or 1.0)
    values = np.array(combined, dtype=np.float32)
    if values.size >= 5:
        low, high = np.percentile(values, [5, 95])
    else:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        low = max(0.0, low - 0.5)
        high = high + 0.5
    spread = high - low
    padding = max(0.5, 0.1 * spread)
    range_min = max(0.0, low - padding)
    range_max = high + padding
    if max_rtt is not None:
        range_max = min(range_max, max_rtt)
    if range_max <= range_min:
        range_min = max(0.0, range_max - 1.0)
        range_max = range_min + 1.0
    range_min = max(0.1, range_min)
    if range_max <= range_min:
        range_max = range_min + 1.0
    return float(range_min), float(range_max)


def compute_rtt_median(values):
    valid = [v for v in values if v >= 0]
    if not valid:
        return None
    return float(np.median(valid))


def discretize_rtt_with_timeout(
    rtt_values,
    timeout_count,
    bins=50,
    range_min=0.0,
    range_max=1000.0,
    bin_edges=None,
):
    if bin_edges is None:
        bin_edges = np.linspace(range_min, range_max, bins + 1)
    valid_rtts = [r for r in rtt_values if r >= 0]
    if valid_rtts:
        clipped = np.clip(valid_rtts, bin_edges[0], bin_edges[-1])
    else:
        clipped = []
    hist, edges = np.histogram(clipped, bins=bin_edges)
    total = len(valid_rtts) + timeout_count
    prob = np.zeros(len(bin_edges))
    if total > 0:
        prob[: len(hist)] = hist / total
        prob[-1] = timeout_count / total
    return prob, edges


def smooth_hist(prob, kernel=None):
    if prob.size <= 1:
        return prob
    body = prob[:-1]
    if body.sum() <= 0:
        return prob
    if kernel is None:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    smoothed = np.convolve(body, kernel, mode="same")
    if smoothed.sum() > 0:
        smoothed = smoothed * (body.sum() / smoothed.sum())
    out = np.zeros_like(prob)
    out[:-1] = smoothed
    out[-1] = prob[-1]
    return out


def kl_divergence(p, q, epsilon=1e-10):
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def plot_ping_grid(
    output_path,
    results,
    bins,
    avg_kl=None,
    src_ip=None,
    page_index=None,
    page_count=None,
    grid_rows=4,
    grid_cols=5,
):
    if not results:
        return
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(grid_cols * 4.0, grid_rows * 3.0),
        squeeze=False,
    )
    rows_used = math.ceil(len(results) / grid_cols)
    last_row = max(0, rows_used - 1)
    for idx, result in enumerate(results):
        row_idx = idx // grid_cols
        col_idx = idx % grid_cols
        ax = axes[row_idx][col_idx]
        real_rtts = result.get("real_rtts", [])
        model_rtts = result.get("model_rtts", [])
        range_min = result.get("range_min", 0.0)
        range_max = result.get("range_max", range_min + 1.0)
        range_min = max(0.1, range_min)
        range_max = max(range_max, range_min * 1.1)
        bin_edges = np.geomspace(range_min, range_max, bins + 1)
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        timeout_center = range_max * 2.5
        timeout_width = range_max * 0.4
        centers = np.concatenate([centers, [timeout_center]])
        widths = np.concatenate([bin_widths, [timeout_width]])

        real_timeout = sum(1 for v in real_rtts if v < 0)
        model_timeout = result.get("model_timeout_count", 0)
        model_invalid = result.get("model_invalid_count", 0)
        model_timeout_total = model_timeout + model_invalid

        real_prob, _ = discretize_rtt_with_timeout(
            real_rtts,
            real_timeout,
            bins=bins,
            range_min=range_min,
            range_max=range_max,
            bin_edges=bin_edges,
        )
        model_prob, _ = discretize_rtt_with_timeout(
            model_rtts,
            model_timeout_total,
            bins=bins,
            range_min=range_min,
            range_max=range_max,
            bin_edges=bin_edges,
        )
        real_plot = smooth_hist(real_prob)
        model_plot = smooth_hist(model_prob)

        show_legend = idx == 0
        ax.bar(
            centers,
            real_plot,
            width=widths * 0.9,
            alpha=0.6,
            label="real" if show_legend else None,
            color="#4C78A8",
        )
        ax.bar(
            centers,
            model_plot,
            width=widths * 0.9,
            alpha=0.6,
            label="model" if show_legend else None,
            color="#F58518",
        )

        real_median = compute_rtt_median(real_rtts)
        if real_median is not None and real_median > 0:
            ax.axvline(
                real_median,
                color="black",
                linestyle="-",
                linewidth=1.2,
                zorder=4,
            )
        model_median = compute_rtt_median(model_rtts)
        if model_median is not None and model_median > 0:
            ax.axvline(
                model_median,
                color="black",
                linestyle="--",
                linewidth=1.2,
                zorder=4,
            )

        ax.set_title(f"{result['label']} | KL={result['kl']:.3f}")
        ax.set_xlim(range_min, timeout_center + timeout_width)
        ax.set_xscale("log")
        tick_vals = list(np.geomspace(range_min, range_max, num=5))
        tick_vals.append(timeout_center)

        def format_tick(val, precision):
            if val >= 100:
                fmt = f"{{val:.{precision}f}}" if precision else "{val:.0f}"
            elif val >= 10:
                fmt = f"{{val:.{precision}f}}" if precision else "{val:.0f}"
            elif val >= 1:
                fmt = f"{{val:.{max(1, precision)}f}}"
            else:
                fmt = f"{{val:.{max(2, precision)}f}}"
            return fmt.format(val=val)

        def build_labels(precision):
            return [format_tick(val, precision) for val in tick_vals[:-1]]

        tick_labels = build_labels(0)
        if len(set(tick_labels)) < len(tick_labels):
            tick_labels = build_labels(1)
        if len(set(tick_labels)) < len(tick_labels):
            tick_labels = build_labels(2)
        tick_labels.append("Timeout")
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labels, rotation=25, ha="right")
        ax.tick_params(axis="x", which="both", length=3, width=0.8, labelsize=7)
        if row_idx == last_row:
            ax.set_xlabel("RTT (ms)")
        if col_idx == 0:
            ax.set_ylabel("probability mass")
        ax.tick_params(axis="y", which="both", length=3, width=0.8, labelsize=7)
        if col_idx != 0:
            ax.tick_params(axis="y", labelleft=False)
        if real_median is not None and real_median > 0:
            ax.text(
                real_median,
                0.95,
                f"R {real_median:.1f}",
                transform=ax.get_xaxis_transform(),
                fontsize=6,
                rotation=90,
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
            )
        if model_median is not None and model_median > 0:
            ax.text(
                model_median,
                0.8,
                f"M {model_median:.1f}",
                transform=ax.get_xaxis_transform(),
                fontsize=6,
                rotation=90,
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
            )

    for idx in range(len(results), grid_rows * grid_cols):
        row_idx = idx // grid_cols
        col_idx = idx % grid_cols
        axes[row_idx][col_idx].axis("off")

    legend_handles, legend_labels = axes[0][0].get_legend_handles_labels()
    median_handles = [
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=1.2, label="real median"
        ),
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.2, label="model median"
        ),
    ]
    if legend_handles:
        fig.legend(
            legend_handles + median_handles,
            legend_labels + ["real median", "model median"],
            loc="upper right",
        )
    if avg_kl is not None and not math.isnan(avg_kl):
        title = f"Live ping avg KL: {avg_kl:.4f}"
        if src_ip:
            title = f"{title} | src {src_ip}"
        if page_index is not None and page_count is not None and page_count > 1:
            title = f"{title} | page {page_index}/{page_count}"
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_timestamps(metrics, output_dir, hist_bins_override=None):
    data = metrics.get("timestamp_accuracy")
    if not data:
        print("Timestamp metrics missing timestamp_accuracy; skipping.")
        return None
    if "full_logps" not in data or "none_logps" not in data:
        print("Timestamp metrics missing log-prob arrays; skipping histogram.")
        return None

    full_logps = np.array(data.get("full_logps", []), dtype=np.float32)
    none_logps = np.array(data.get("none_logps", []), dtype=np.float32)
    hist_bins = hist_bins_override or metrics.get("params", {}).get("hist_bins", 200)
    hist_bins_arr = build_hist_bins(full_logps, none_logps, hist_bins)

    hist_path = output_dir / "timestamp_logprob_hist.png"
    plot_timestamp_effects(
        hist_path,
        full_logps,
        none_logps,
        hist_bins,
    )

    hist_cdf_path = output_dir / "timestamp_logprob_hist_cdf.png"
    plot_logprob_hist_with_cdf(
        hist_cdf_path,
        full_logps,
        none_logps,
        hist_bins_arr,
        "Log P(correct token): full vs no timestamps",
        clip_percentiles=(2, 98),
    )

    cdf_path = output_dir / "timestamp_logprob_cdf.png"
    plot_timestamp_threshold_cdf(
        cdf_path,
        full_logps,
        none_logps,
    )

    buckets_path = output_dir / "timestamp_logprob_buckets.png"
    plot_timestamp_buckets(
        buckets_path,
        full_logps,
        none_logps,
        data.get("full_rtt_ms", []),
        data.get("none_rtt_ms", []),
        data.get("full_event_hour", []),
        data.get("none_event_hour", []),
    )
    return hist_path


def plot_modes(metrics, output_dir):
    order = metrics.get(
        "group_order",
        [
            "src_ipv4",
            "src_ipv6",
            "dst_ipv4",
            "dst_ipv6",
            "rtt",
            "rtt_failed",
            "timestamp_abs",
            "timestamp_delta1",
            "timestamp_delta4",
        ],
    )
    if "groups" in metrics:
        groups = metrics.get("groups", {})
        summary = {}
        for group, payload in groups.items():
            per_pos = payload.get("per_pos_logps", [])
            flat = [lp for pos in per_pos for lp in pos]
            summary[group] = summarize_logps(np.array(flat))

        summary_path = output_dir / "mode_accuracy_summary.png"
        plot_mode_summary(summary_path, summary, order)

        per_group_figs = {}
        for group, payload in groups.items():
            per_pos = payload.get("per_pos_logps", [])
            mean_probs = []
            for pos_vals in per_pos:
                if pos_vals:
                    mean_probs.append(float(np.mean(np.exp(pos_vals))))
                else:
                    mean_probs.append(0.0)
            labels = payload.get("labels", [str(i) for i in range(len(mean_probs))])
            fig_path = output_dir / f"mode_{group}_token_accuracy.png"
            plot_token_accuracy_bar(
                fig_path,
                labels,
                mean_probs,
                f"{group} token accuracy",
            )
            per_group_figs[group] = fig_path
        return summary_path, per_group_figs

    mode_accuracy = metrics.get("mode_accuracy", {})
    summary = mode_accuracy.get("groups", {})
    if not summary:
        print("Mode metrics missing per-position logps and summary; skipping.")
        return None, {}
    summary_path = output_dir / "mode_accuracy_summary.png"
    plot_mode_summary(summary_path, summary, order)
    print("Mode metrics missing per-position logps; only summary plot generated.")
    return summary_path, {}


def plot_ping(
    metrics,
    output_dir,
    ping_bins_override=None,
    max_rtt_override=None,
    grid_rows=4,
    grid_cols=5,
):
    params = metrics.get("params", {})
    ping_data = metrics.get("ping", metrics.get("ping_data", {}))
    targets = ping_data.get("targets", [])
    if not targets:
        return []

    ping_bins = ping_bins_override or params.get("ping_bins", 40)
    max_rtt = max_rtt_override
    if max_rtt is None:
        max_rtt = params.get("max_rtt", 1000)

    results = []
    for target in targets:
        real_rtts = target.get("real_rtts", [])
        model_rtts = target.get("model_rtts", [])
        range_min, range_max = compute_rtt_range(real_rtts, model_rtts, max_rtt)
        real_success = target.get("real_success_count")
        if real_success is None and real_rtts:
            real_success = sum(1 for r in real_rtts if r >= 0)
        if real_rtts:
            real_timeout = (len(real_rtts) - real_success) / len(real_rtts)
            real_timeout_count = len(real_rtts) - real_success
        else:
            real_timeout = 0.0
            real_timeout_count = 0

        model_timeout_count = target.get("model_timeout_count", 0)
        model_invalid_count = target.get("model_invalid_count", 0)
        model_total = len(model_rtts) + model_timeout_count + model_invalid_count
        model_timeout = model_timeout_count / model_total if model_total else 0.0

        bin_edges = np.geomspace(range_min, range_max, ping_bins + 1)
        real_dist, _ = discretize_rtt_with_timeout(
            real_rtts,
            real_timeout_count,
            bins=ping_bins,
            range_min=range_min,
            range_max=range_max,
            bin_edges=bin_edges,
        )
        model_dist, _ = discretize_rtt_with_timeout(
            model_rtts,
            model_timeout_count + model_invalid_count,
            bins=ping_bins,
            range_min=range_min,
            range_max=range_max,
            bin_edges=bin_edges,
        )
        kl_value = kl_divergence(real_dist, model_dist)

        results.append(
            {
                "dst_ip": target.get("dst_ip"),
                "domain": target.get("domain"),
                "label": target.get("label", ""),
                "kl": kl_value,
                "real_rtts": real_rtts,
                "model_rtts": model_rtts,
                "model_timeout_count": model_timeout_count,
                "model_invalid_count": model_invalid_count,
                "real_timeout_count": real_timeout_count,
                "real_timeout_rate": target.get("real_timeout_rate", real_timeout),
                "model_timeout_rate": target.get("model_timeout_rate", model_timeout),
                "range_min": range_min,
                "range_max": range_max,
            }
        )

    avg_kl = float(np.mean([r["kl"] for r in results])) if results else float("nan")
    per_page = grid_rows * grid_cols
    pages = [results[i : i + per_page] for i in range(0, len(results), per_page)]
    output_paths = []
    for idx, chunk in enumerate(pages):
        suffix = f"_{idx + 1:02d}" if len(pages) > 1 else ""
        fig_path = output_dir / f"live_ping_grid{suffix}.png"
        plot_ping_grid(
            fig_path,
            chunk,
            ping_bins,
            avg_kl=avg_kl,
            src_ip=ping_data.get("src_ip"),
            page_index=idx + 1,
            page_count=len(pages),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        output_paths.append(fig_path)
    return output_paths


def build_parser():
    parser = argparse.ArgumentParser(description="Plot paper metrics from JSON files")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run output directory (contains metrics/)",
    )
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Directory containing metrics JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figures (default: run-dir/figures)",
    )
    parser.add_argument(
        "--only",
        default="all",
        help="Comma list: timestamps,modes,ping,all",
    )
    parser.add_argument(
        "--timestamp-metrics",
        default=None,
        help="Path to timestamp_metrics.json",
    )
    parser.add_argument(
        "--mode-metrics",
        default=None,
        help="Path to mode_metrics.json",
    )
    parser.add_argument(
        "--ping-metrics",
        default=None,
        help="Path to ping_metrics.json",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=None,
        help="Override histogram bins for timestamp plots",
    )
    parser.add_argument(
        "--ping-bins",
        type=int,
        default=None,
        help="Override histogram bins for ping plots",
    )
    parser.add_argument(
        "--max-rtt",
        type=int,
        default=None,
        help="Override max RTT for ping plots",
    )
    parser.add_argument(
        "--ping-grid-rows",
        type=int,
        default=4,
        help="Rows per ping grid page",
    )
    parser.add_argument(
        "--ping-grid-cols",
        type=int,
        default=5,
        help="Columns per ping grid page",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.metrics_dir is None and args.run_dir is None:
        run_dir = DEFAULT_RUN_DIR
        metrics_dir = run_dir / "metrics"
        output_dir = run_dir / "figures"
    else:
        if args.metrics_dir is None:
            metrics_dir = Path(args.run_dir) / "metrics"
        else:
            metrics_dir = Path(args.metrics_dir)

        if args.output_dir is None:
            if args.run_dir is not None:
                output_dir = Path(args.run_dir) / "figures"
            else:
                output_dir = metrics_dir.parent / "figures"
        else:
            output_dir = Path(args.output_dir)

    if not metrics_dir.exists() or not metrics_dir.is_dir():
        print(f"Metrics directory not found: {metrics_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    clear_dir(output_dir, suffixes={".png"})

    selected = parse_only_arg(args.only)

    if "timestamps" in selected:
        timestamp_path = (
            args.timestamp_metrics or metrics_dir / "timestamp_metrics.json"
        )
        if Path(timestamp_path).exists():
            timestamp_metrics = load_json(timestamp_path)
            plot_timestamps(timestamp_metrics, output_dir, args.hist_bins)
        else:
            print(f"Timestamp metrics not found: {timestamp_path}")

    if "modes" in selected:
        mode_path = args.mode_metrics or metrics_dir / "mode_metrics.json"
        if Path(mode_path).exists():
            mode_metrics = load_json(mode_path)
            plot_modes(mode_metrics, output_dir)
        else:
            print(f"Mode metrics not found: {mode_path}")

    if "ping" in selected:
        ping_path = args.ping_metrics or metrics_dir / "ping_metrics.json"
        if Path(ping_path).exists():
            ping_metrics = load_json(ping_path)
            plot_ping(
                ping_metrics,
                output_dir,
                ping_bins_override=args.ping_bins,
                max_rtt_override=args.max_rtt,
                grid_rows=args.ping_grid_rows,
                grid_cols=args.ping_grid_cols,
            )
        else:
            print(f"Ping metrics not found: {ping_path}")


if __name__ == "__main__":
    main()
