"""
Model evaluation on the analysis train/test dataset.

Builds tokenized sequences from raw measurements: N context measurements
from the analysis train set, followed by 1 test measurement to predict.
Runs model forward pass and extracts top-1 RTT prediction at each test
measurement's RTT byte positions.

Usage:
    python -m ping_llm.eval.model_eval \
        --checkpoint path/to/latest.pt \
        --eval-dir data/eval \
        --run-name deep60-60k \
        --num-context 0,1,2,5,10

    # No-timestamp ablation:
    python -m ping_llm.eval.model_eval \
        --checkpoint path/to/latest.pt \
        --eval-dir data/eval \
        --run-name deep60-60k-nots --strip-timestamps
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ping_llm.inference import load_model
from ping_llm.eval.baselines import extract_rtt_positions, model_top1_rtt
from ping_llm.data.tokenization import (
    encode_measurement, TIMESTAMP_ABS, TIMESTAMP_DELTA1, TIMESTAMP_DELTA4,
)
from ping_llm.eval.token_classify import ROLE_BYTE_COUNTS


def strip_timestamps(tokens):
    """Remove all timestamp blocks from a token sequence."""
    ts_tokens = {TIMESTAMP_ABS, TIMESTAMP_DELTA1, TIMESTAMP_DELTA4}
    result = []
    i = 0
    n = len(tokens)
    while i < n:
        t = int(tokens[i])
        if t in ts_tokens:
            i += 1 + ROLE_BYTE_COUNTS[t]
        else:
            result.append(t)
            i += 1
    return result


def _meas_row_to_dict(row):
    """Convert a DataFrame row to the dict format encode_measurement expects."""
    addr = row["src_addr"]
    try:
        import ipaddress
        ip = ipaddress.ip_address(addr)
        ip_version = 4 if ip.version == 4 else 6
    except Exception:
        ip_version = 4
    return {
        "src_addr": row["src_addr"],
        "dst_addr": row["dst_addr"],
        "ip_version": ip_version,
        "rtt": float(row["rtt"]),
        "event_time": row["event_time"],
    }


def build_sequence(context_rows, test_row, strip_ts=False):
    """
    Build a token sequence: context measurements followed by the test measurement.

    Returns (tokens, rtt_byte1_pos, rtt_byte2_pos) or None if encoding fails.
    """
    tokens = []
    prev_ts = None

    for _, row in context_rows.iterrows():
        meas_dict = _meas_row_to_dict(row)
        meas_tokens = encode_measurement(meas_dict, prev_timestamp=prev_ts,
                                         include_timestamp=not strip_ts)
        tokens.extend(meas_tokens)
        prev_ts = row["event_time"]

    test_dict = _meas_row_to_dict(test_row)
    test_tokens = encode_measurement(test_dict, prev_timestamp=prev_ts,
                                     include_timestamp=not strip_ts)
    test_start = len(tokens)
    tokens.extend(test_tokens)

    if strip_ts:
        tokens = strip_timestamps(tokens)

    positions = extract_rtt_positions(tokens)
    if not positions:
        return None

    last_pos = positions[-1]
    if last_pos["rtt_ms"] <= 0 or last_pos["byte1_pos"] is None:
        return None

    return tokens, last_pos["byte1_pos"], last_pos["byte2_pos"]


def run_model_eval(checkpoint, eval_dir, run_name, num_context_list,
                   device=None, strip_ts=False, max_test=10000, batch_size=1):
    t0 = time.time()
    eval_dir = Path(eval_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {checkpoint} → {device}")
    model, _ = load_model(checkpoint, device=device)
    model.eval()

    train_df = pd.read_parquet(eval_dir / "train_measurements.parquet")
    test_df = pd.read_parquet(eval_dir / "test_measurements.parquet")

    # Build per-src train index: for each src, its train measurements sorted by time
    train_by_src = {}
    for src, group in train_df.sort_values("event_time").groupby("src_addr"):
        train_by_src[src] = group

    # Sample test measurements (can't run all 2M+)
    if len(test_df) > max_test:
        test_df = test_df.sample(n=max_test, random_state=42)
    test_df = test_df.sort_values(["src_addr", "event_time"]).reset_index(drop=True)

    print(f"Evaluating {len(test_df):,} test measurements with "
          f"context sizes {num_context_list}")

    results = []

    for n_ctx in num_context_list:
        print(f"\n  Context = {n_ctx}...")
        preds = []
        actuals = []
        skipped = 0

        for i, (_, test_row) in enumerate(test_df.iterrows()):
            src = test_row["src_addr"]
            ctx = train_by_src.get(src)

            if ctx is not None and n_ctx > 0:
                if len(ctx) > n_ctx:
                    ctx_sample = ctx.sample(n=n_ctx, random_state=i)
                else:
                    ctx_sample = ctx
                ctx_sample = ctx_sample.sort_values("event_time")
            else:
                ctx_sample = pd.DataFrame(columns=train_df.columns).iloc[:0]

            seq = build_sequence(ctx_sample, test_row, strip_ts=strip_ts)
            if seq is None:
                skipped += 1
                preds.append(float("nan"))
                actuals.append(float(test_row["rtt"]))
                continue

            tokens, b1_pos, b2_pos = seq

            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(idx)
            logits = logits[0]

            pred = model_top1_rtt(logits[b1_pos - 1], logits[b2_pos - 1])
            preds.append(pred if pred is not None else float("nan"))
            actuals.append(float(test_row["rtt"]))

            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(test_df)}")

        preds = np.array(preds)
        actuals = np.array(actuals)
        valid = ~np.isnan(preds) & ~np.isnan(actuals) & (actuals > 0)
        abs_err = np.abs(preds[valid] - actuals[valid])

        print(f"    valid={valid.sum()}, skipped={skipped}, "
              f"MAE={abs_err.mean():.2f}ms, median_AE={np.median(abs_err):.2f}ms")

        for j in range(len(test_df)):
            results.append({
                "src_addr": test_df.iloc[j]["src_addr"],
                "dst_addr": test_df.iloc[j]["dst_addr"],
                "actual_rtt_ms": actuals[j],
                "num_context": n_ctx,
                "model_pred": preds[j],
            })

    results_df = pd.DataFrame(results)
    out_path = eval_dir / "model_preds" / f"{run_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(results_df):,} predictions → {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser(description="Model eval on analysis train/test data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-dir", default="data/eval")
    p.add_argument("--run-name", required=True)
    p.add_argument("--num-context", default="0,1,2,5,10",
                   help="Comma-separated context sizes to evaluate")
    p.add_argument("--device", default=None)
    p.add_argument("--max-test", type=int, default=10000)
    p.add_argument("--strip-timestamps", action="store_true")
    args = p.parse_args()

    num_context_list = [int(x) for x in args.num_context.split(",")]
    run_model_eval(
        args.checkpoint, args.eval_dir, args.run_name,
        num_context_list, args.device,
        strip_ts=args.strip_timestamps, max_test=args.max_test,
    )


if __name__ == "__main__":
    main()
