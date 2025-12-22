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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
    return float(range_min), float(range_max)


def compute_rtt_median(values):
    valid = [v for v in values if v >= 0]
    if not valid:
        return None
    return float(np.median(valid))


def discretize_rtt_with_timeout(
    rtt_values, timeout_count, bins=50, range_min=0.0, range_max=1000.0
):
    valid_rtts = [r for r in rtt_values if r >= 0]
    if valid_rtts:
        clipped = np.clip(valid_rtts, range_min, range_max)
    else:
        clipped = []
    hist, edges = np.histogram(clipped, bins=bins, range=(range_min, range_max))
    total = len(valid_rtts) + timeout_count
    prob = np.zeros(bins + 1)
    if total > 0:
        prob[:bins] = hist / total
        prob[-1] = timeout_count / total
    return prob, edges


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
        bin_edges = np.linspace(range_min, range_max, bins + 1)
        bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        timeout_center = range_max + bin_width / 2.0
        centers = np.concatenate([centers, [timeout_center]])

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
        )
        model_prob, _ = discretize_rtt_with_timeout(
            model_rtts,
            model_timeout_total,
            bins=bins,
            range_min=range_min,
            range_max=range_max,
        )

        show_legend = idx == 0
        ax.bar(
            centers,
            real_prob,
            width=bin_width * 0.9,
            alpha=0.6,
            label="real" if show_legend else None,
            color="#4C78A8",
        )
        ax.bar(
            centers,
            model_prob,
            width=bin_width * 0.9,
            alpha=0.6,
            label="model" if show_legend else None,
            color="#F58518",
        )

        real_median = compute_rtt_median(real_rtts)
        if real_median is not None:
            ax.axvline(real_median, color="black", linestyle="-", linewidth=1.2)
        model_median = compute_rtt_median(model_rtts)
        if model_median is not None:
            ax.axvline(model_median, color="black", linestyle="--", linewidth=1.2)

        ax.set_title(f"{result['label']} | KL={result['kl']:.3f}")
        ax.set_xlim(range_min, range_max + bin_width)
        if row_idx == last_row:
            ax.set_xlabel("RTT (ms)")
        ax.set_xticks([range_min, range_max, timeout_center])
        ax.set_xticklabels([f"{range_min:.0f}", f"{range_max:.0f}", "Timeout"])
        ax.tick_params(axis="x", labelsize=7)
        if col_idx == 0:
            ax.set_ylabel("probability mass")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="y", labelsize=7)

    for idx in range(len(results), grid_rows * grid_cols):
        row_idx = idx // grid_cols
        col_idx = idx % grid_cols
        axes[row_idx][col_idx].axis("off")

    legend_handles, legend_labels = axes[0][0].get_legend_handles_labels()
    median_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.2, label="real median"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="model median"),
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
    hist_bins = hist_bins_override or metrics.get("params", {}).get("hist_bins", 60)

    all_logps = (
        np.concatenate([full_logps, none_logps])
        if full_logps.size and none_logps.size
        else np.array([])
    )
    if all_logps.size:
        low, high = np.percentile(all_logps, [1, 99])
        if low == high:
            low, high = low - 1.0, high + 1.0
        bins = np.linspace(low, high, hist_bins)
    else:
        bins = hist_bins

    hist_path = output_dir / "timestamp_logprob_hist.png"
    plot_logprob_hist(
        hist_path,
        full_logps,
        none_logps,
        bins,
        "Log P(correct token): full vs no timestamps",
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

        real_dist, _ = discretize_rtt_with_timeout(
            real_rtts,
            real_timeout_count,
            bins=ping_bins,
            range_min=range_min,
            range_max=range_max,
        )
        model_dist, _ = discretize_rtt_with_timeout(
            model_rtts,
            model_timeout_count + model_invalid_count,
            bins=ping_bins,
            range_min=range_min,
            range_max=range_max,
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
    pages = [
        results[i : i + per_page] for i in range(0, len(results), per_page)
    ]
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
        timestamp_path = args.timestamp_metrics or metrics_dir / "timestamp_metrics.json"
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
