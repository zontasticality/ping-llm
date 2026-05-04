"""Token-type accuracy by measurement position in a packed eval context.

This analysis greedily packs the most recent same-source train measurements
before each test query up to the model sequence length, runs a full forward
pass, and aggregates next-token top-1 accuracy by semantic token type and
measurement offset from the query.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ping_llm.data.tokenization import MEASUREMENT_START, encode_measurement
from ping_llm.eval.model_eval import _meas_row_to_dict
from ping_llm.eval.token_classify import classify_tokens
from ping_llm.inference import load_model


TOKEN_TYPE_ORDER = [
    "role",
    "src_ip_byte",
    "dst_ip_byte",
    "timestamp_byte",
    "rtt_byte",
    "unknown",
]

TOKEN_TYPE_COLORS = {
    "role": "#4C78A8",
    "src_ip_byte": "#59A14F",
    "dst_ip_byte": "#8CD17D",
    "timestamp_byte": "#F28E2B",
    "rtt_byte": "#E15759",
    "unknown": "#9D9D9D",
}


def _tokenize_rows(context_rows, test_row, include_timestamp=True):
    tokens = []
    prev_ts = None
    for _, row in context_rows.iterrows():
        meas = _meas_row_to_dict(row)
        meas_tokens = encode_measurement(
            meas,
            prev_timestamp=prev_ts,
            include_timestamp=include_timestamp,
        )
        tokens.extend(meas_tokens)
        prev_ts = row["event_time"]

    test_meas = _meas_row_to_dict(test_row)
    tokens.extend(encode_measurement(
        test_meas,
        prev_timestamp=prev_ts,
        include_timestamp=include_timestamp,
    ))
    return tokens


def _measurement_indices(tokens):
    indices = []
    meas_idx = -1
    for token in tokens:
        if int(token) == MEASUREMENT_START:
            meas_idx += 1
        indices.append(meas_idx)
    return indices


def _pack_max_context(ctx, test_row, seq_len, candidate_cap=160,
                      include_timestamp=True):
    if ctx is None or len(ctx) == 0:
        context_rows = ctx.iloc[:0] if ctx is not None else pd.DataFrame()
        tokens = _tokenize_rows(context_rows, test_row, include_timestamp)
        return context_rows, tokens

    candidates = ctx.tail(candidate_cap)
    selected = []
    for _, row in reversed(list(candidates.iterrows())):
        trial = [row] + selected
        trial_df = pd.DataFrame(trial).sort_values("event_time")
        tokens = _tokenize_rows(trial_df, test_row, include_timestamp)
        if len(tokens) <= seq_len:
            selected = trial
        else:
            break

    context_rows = pd.DataFrame(selected).sort_values("event_time")
    tokens = _tokenize_rows(context_rows, test_row, include_timestamp)
    while len(tokens) > seq_len and len(context_rows) > 0:
        context_rows = context_rows.iloc[1:]
        tokens = _tokenize_rows(context_rows, test_row, include_timestamp)
    return context_rows, tokens


def _make_test_sample(eval_dir, max_test):
    observations_path = eval_dir / "observations.parquet"
    if observations_path.exists():
        test_df = pd.read_parquet(observations_path)
        test_df = test_df.copy()
        test_df["rtt"] = test_df["actual_rtt_ms"]
    else:
        test_df = pd.read_parquet(eval_dir / "test_measurements.parquet")
        test_df = test_df.copy()
    if "obs_id" not in test_df.columns:
        test_df["obs_id"] = np.arange(len(test_df), dtype=np.int64)

    if len(test_df) > max_test:
        test_df = test_df.sample(n=max_test, random_state=42)
    return test_df.sort_values(["src_addr", "event_time"]).reset_index(drop=True)


def _plot_capacity(capacity_df, output_dir):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.hist(
        capacity_df["context_count"],
        bins=np.arange(capacity_df["context_count"].max() + 2) - 0.5,
        color="#4C78A8",
        edgecolor="white",
    )
    ax.axvline(capacity_df["context_count"].median(), color="#E15759",
               linestyle="--", linewidth=2, label="median")
    ax.set_xlabel("Max same-source context measurements that fit")
    ax.set_ylabel("Test queries")
    ax.set_title("Context capacity under 1024-token sequence length")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "context_capacity.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _plot_heatmap(acc_df, output_dir):
    pivot = acc_df.pivot(index="token_type", columns="offset", values="accuracy")
    token_types = [t for t in TOKEN_TYPE_ORDER if t in pivot.index]
    token_types += [t for t in pivot.index if t not in token_types]
    offsets = sorted(pivot.columns)
    mat = pivot.reindex(index=token_types, columns=offsets).to_numpy()

    fig_width = max(9.0, min(16.0, 5.0 + len(offsets) * 0.13))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                   vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(token_types)))
    ax.set_yticklabels(token_types)

    tick_step = max(1, len(offsets) // 14)
    tick_positions = np.arange(0, len(offsets), tick_step)
    if len(offsets) - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, len(offsets) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(offsets[i]) for i in tick_positions], rotation=0)
    ax.set_xlabel("Measurement offset from query (0 = query, -1 = most recent context)")
    ax.set_title("Top-1 token accuracy by token type and context position")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Top-1 accuracy")

    path = output_dir / "token_position_accuracy_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _plot_relative(acc_df, output_dir):
    token_types = [t for t in TOKEN_TYPE_ORDER if t in set(acc_df["token_type"])]
    n = len(token_types)
    fig, axes = plt.subplots(n, 1, figsize=(9.2, max(2.0 * n, 4.5)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, token_type in zip(axes, token_types):
        sub = acc_df[acc_df["token_type"] == token_type].sort_values("offset")
        if sub.empty:
            continue
        baseline = sub["correct"].sum() / sub["count"].sum()
        y = (sub["accuracy"] - baseline) * 100.0
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.plot(
            sub["offset"],
            y,
            color=TOKEN_TYPE_COLORS.get(token_type, "#4C78A8"),
            linewidth=1.8,
            marker="o",
            markersize=2.5,
        )
        ax.set_ylabel(token_type)
        ax.grid(True, axis="both", alpha=0.25)
        ax.text(
            0.99,
            0.82,
            f"mean={baseline:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )

    axes[-1].set_xlabel("Measurement offset from query (0 = query)")
    fig.suptitle("Token accuracy change relative to each token type's own mean")
    fig.tight_layout()
    path = output_dir / "token_position_accuracy_relative.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def run_token_position_analysis(checkpoint, eval_dir, output_dir, max_test=2000,
                                device=None, dtype="bfloat16"):
    t0 = time.time()
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    print(f"Loading model from {checkpoint} -> {device} ({dtype})")
    model, model_cfg = load_model(checkpoint, device=device, dtype=dtype_map[dtype])
    model.eval()
    seq_len = int(model_cfg.seq_len)
    print(f"Model seq_len: {seq_len}")

    print("Loading train and test data...")
    train_df = pd.read_parquet(eval_dir / "train_measurements.parquet")
    test_df = _make_test_sample(eval_dir, max_test=max_test)
    train_by_src = {
        src: group.sort_values("event_time")
        for src, group in train_df.sort_values("event_time").groupby("src_addr")
    }
    print(f"Analyzing {len(test_df):,} test queries")

    agg = defaultdict(lambda: {"count": 0, "correct": 0, "total_ce": 0.0})
    capacity_rows = []

    for i, (_, test_row) in enumerate(test_df.iterrows()):
        ctx = train_by_src.get(test_row["src_addr"])
        context_rows, tokens = _pack_max_context(ctx, test_row, seq_len)
        if len(tokens) < 2:
            continue
        n_ctx = len(context_rows)
        capacity_rows.append({
            "obs_id": int(test_row["obs_id"]),
            "src_addr": test_row["src_addr"],
            "dst_addr": test_row["dst_addr"],
            "context_count": int(n_ctx),
            "token_length": int(len(tokens)),
            "seq_len": seq_len,
        })

        labels = classify_tokens(tokens)
        meas_indices = _measurement_indices(tokens)

        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx)
        logits = logits[0]
        targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)
        pred = logits[:-1].argmax(dim=-1)
        correct = (pred == targets)
        log_probs = F.log_softmax(logits[:-1], dim=-1)
        ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        correct_np = correct.cpu().numpy()
        ce_np = ce.float().cpu().numpy()
        for pos in range(1, len(tokens)):
            token_type = labels[pos]
            offset = int(meas_indices[pos] - n_ctx)
            key = (token_type, offset)
            agg[key]["count"] += 1
            agg[key]["correct"] += int(correct_np[pos - 1])
            agg[key]["total_ce"] += float(ce_np[pos - 1])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_df)}")

    capacity_df = pd.DataFrame(capacity_rows)
    acc_rows = []
    for (token_type, offset), bucket in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        count = bucket["count"]
        if count == 0:
            continue
        correct = bucket["correct"]
        acc_rows.append({
            "token_type": token_type,
            "offset": int(offset),
            "count": int(count),
            "correct": int(correct),
            "accuracy": correct / count,
            "mean_ce": bucket["total_ce"] / count,
        })
    acc_df = pd.DataFrame(acc_rows)

    capacity_path = table_dir / "context_capacity.csv"
    accuracy_path = table_dir / "token_position_accuracy.csv"
    summary_path = table_dir / "context_capacity_summary.json"
    capacity_df.to_csv(capacity_path, index=False)
    acc_df.to_csv(accuracy_path, index=False)
    summary = {
        "num_queries": int(len(capacity_df)),
        "seq_len": seq_len,
        "context_min": int(capacity_df["context_count"].min()),
        "context_median": float(capacity_df["context_count"].median()),
        "context_mean": float(capacity_df["context_count"].mean()),
        "context_p90": float(capacity_df["context_count"].quantile(0.90)),
        "context_max": int(capacity_df["context_count"].max()),
        "token_length_min": int(capacity_df["token_length"].min()),
        "token_length_median": float(capacity_df["token_length"].median()),
        "token_length_max": int(capacity_df["token_length"].max()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Capacity summary: {summary}")
    print(f"Wrote {capacity_path}")
    print(f"Wrote {accuracy_path}")
    print(f"Wrote {_plot_capacity(capacity_df, fig_dir)}")
    print(f"Wrote {_plot_heatmap(acc_df, fig_dir)}")
    print(f"Wrote {_plot_relative(acc_df, fig_dir)}")


def main():
    p = argparse.ArgumentParser(description="Token accuracy by context position")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-dir", default="data/eval_timeclean")
    p.add_argument("--output-dir", default="outputs/eval_timeclean_models")
    p.add_argument("--max-test", type=int, default=2000)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = p.parse_args()
    run_token_position_analysis(
        args.checkpoint,
        args.eval_dir,
        args.output_dir,
        max_test=args.max_test,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
