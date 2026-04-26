"""
Stage 2: Per-checkpoint model evaluation.

Runs once per checkpoint. Forward pass on test sequences, extracts
model's top-1 RTT predictions and byte-level logprobs. Optionally
strips timestamps for ablation.

Usage:
    python -m ping_llm.eval.model_eval \
        --checkpoint path/to/latest.pt \
        --test-data path/to/test.arrayrecord \
        --harness-dir outputs/eval_harness/default \
        --run-name deep60-60k

    # No-timestamp ablation:
    python -m ping_llm.eval.model_eval \
        --checkpoint path/to/latest.pt \
        --test-data path/to/test.arrayrecord \
        --harness-dir outputs/eval_harness/default \
        --run-name deep60-60k-nots --strip-timestamps
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_torch
from pathlib import Path

from ping_llm.inference import load_model
from ping_llm.eval.baselines import extract_rtt_positions, model_top1_rtt
from ping_llm.eval.run_all import load_test_sequences
from ping_llm.data.tokenization import (
    TIMESTAMP_ABS, TIMESTAMP_DELTA1, TIMESTAMP_DELTA4, BYTE_TOKEN_OFFSET,
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


def run_model_eval(checkpoint, test_data, harness_dir="outputs/eval_harness/default",
                   run_name="model", device=None, strip_ts=False):
    t0 = time.time()
    harness_dir = Path(harness_dir)

    meta = json.loads((harness_dir / "baselines_meta.json").read_text())
    seed = meta["seed"]
    num_sequences = meta["num_sequences"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {checkpoint} → {device}")
    model, _ = load_model(checkpoint, device=device)

    print(f"Loading {num_sequences} test sequences (seed={seed})...")
    sequences = load_test_sequences(test_data, num_sequences=num_sequences, seed=seed)

    if strip_ts:
        print("Stripping timestamps from all sequences...")
        sequences = [strip_timestamps(s) for s in sequences]

    rows = []
    all_byte1_lp = []
    all_byte2_lp = []

    for seq_idx, tokens in enumerate(sequences):
        positions = extract_rtt_positions(tokens)
        valid = [p for p in positions if p["rtt_ms"] > 0 and p["byte1_pos"] is not None]
        if not valid:
            continue

        token_list = [int(t) for t in tokens]
        idx = torch.tensor([token_list], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx)
        logits = logits[0]

        for pos in valid:
            b1_logits = logits[pos["byte1_pos"] - 1]
            b2_logits = logits[pos["byte2_pos"] - 1]

            pred = model_top1_rtt(b1_logits, b2_logits)

            b1_lp = F_torch.log_softmax(b1_logits, dim=-1)[BYTE_TOKEN_OFFSET:BYTE_TOKEN_OFFSET + 256]
            b2_lp = F_torch.log_softmax(b2_logits, dim=-1)[BYTE_TOKEN_OFFSET:BYTE_TOKEN_OFFSET + 256]
            all_byte1_lp.append(b1_lp.cpu().numpy())
            all_byte2_lp.append(b2_lp.cpu().numpy())

            rows.append({
                "seq_idx": seq_idx,
                "meas_idx": pos["measurement_index"],
                "model_top1_pred": pred if pred is not None else float("nan"),
            })

        if (seq_idx + 1) % 50 == 0:
            print(f"  {seq_idx + 1}/{len(sequences)} sequences")

    preds_dir = harness_dir / "model_preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    out_path = preds_dir / f"{run_name}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved {len(rows)} predictions → {out_path}")

    if all_byte1_lp:
        lp_path = preds_dir / f"{run_name}_logprobs.npz"
        np.savez(lp_path,
                 byte1=np.stack(all_byte1_lp),
                 byte2=np.stack(all_byte2_lp))
        print(f"Saved logprobs → {lp_path}")

    # Loss breakdown (reuses existing eval module, extra forward passes)
    print("\nComputing loss breakdown...")
    from ping_llm.eval.loss_breakdown import eval_loss_breakdown
    lb = eval_loss_breakdown(model, sequences, device=device,
                             max_sequences=len(sequences))
    lb_path = preds_dir / f"{run_name}_loss_breakdown.json"
    lb_path.write_text(json.dumps(
        lb, indent=2,
        default=lambda x: round(float(x), 4) if hasattr(x, "item") else x,
    ))
    print(f"Saved loss breakdown → {lb_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser(description="Stage 2: per-checkpoint model eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-data", required=True)
    p.add_argument("--harness-dir", default="outputs/eval_harness/default")
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--strip-timestamps", action="store_true",
                   help="Remove timestamps from test sequences before eval")
    args = p.parse_args()
    run_model_eval(
        args.checkpoint, args.test_data, args.harness_dir,
        args.run_name, args.device, strip_ts=args.strip_timestamps,
    )


if __name__ == "__main__":
    main()
