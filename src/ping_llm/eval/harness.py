"""
Stage 1: Evaluation harness.

Extracts ground truth at every RTT position, trains all baselines,
caches predictions to observations.parquet. Run once per test dataset.

Usage:
    python -m ping_llm.eval.harness \
        --test-data path/to/test.arrayrecord \
        --output-dir outputs/eval_harness/default
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from pathlib import Path

from ping_llm.eval.baselines import extract_rtt_positions
from ping_llm.eval.mf_baseline import BiasedMF, extract_measurements_from_sequences
from ping_llm.eval.vivaldi import fit_vivaldi, predict_vivaldi
from ping_llm.eval.trmf import TRMF
from ping_llm.eval.run_all import load_test_sequences


def ip_key_to_str(key):
    if key is None:
        return ""
    role, bytes_tuple = key
    return f"{role}:{'.'.join(str(b) for b in bytes_tuple)}"


def run_harness(test_data, output_dir="outputs/eval_harness/default",
                num_sequences=500, seed=42):
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_test_sequences(test_data, num_sequences=num_sequences, seed=seed)

    # --- Extract all valid RTT positions with timestamps ---
    rows = []
    for seq_idx, tokens in enumerate(sequences):
        for pos in extract_rtt_positions(tokens):
            if pos["rtt_ms"] <= 0 or pos["byte1_pos"] is None:
                continue
            rows.append({
                "seq_idx": seq_idx,
                "meas_idx": pos["measurement_index"],
                "byte1_pos": pos["byte1_pos"],
                "byte2_pos": pos["byte2_pos"],
                "actual_rtt_ms": pos["rtt_ms"],
                "timestamp": pos.get("timestamp"),
                "src_key": pos["src_key"],
                "dst_key": pos["dst_key"],
            })

    if not rows:
        print("No valid RTT observations found.")
        return

    df = pd.DataFrame(rows)
    n_obs = len(df)
    global_median = float(df["actual_rtt_ms"].median())
    n_timestamped = int(df["timestamp"].notna().sum())
    print(f"Extracted {n_obs} observations from {len(sequences)} sequences "
          f"({n_timestamped} with timestamps)")
    print(f"Global median RTT: {global_median:.2f} ms")

    # --- Simple baselines (per-sequence, history-based) ---
    preds = {k: [] for k in [
        "global_median_pred", "last_seen_pred", "window_mean_pred", "ema_pred",
    ]}
    for _, group in df.groupby("seq_idx", sort=False):
        rtts = group["actual_rtt_ms"].values
        ema = None
        for i in range(len(rtts)):
            if i == 0:
                for col in preds:
                    preds[col].append(global_median)
                ema = rtts[0]
            else:
                preds["global_median_pred"].append(global_median)
                preds["last_seen_pred"].append(float(rtts[i - 1]))
                preds["window_mean_pred"].append(float(np.mean(rtts[max(0, i - 3):i])))
                preds["ema_pred"].append(float(ema))
                ema = 0.3 * rtts[i] + 0.7 * ema
    for col, vals in preds.items():
        df[col] = vals

    # --- Biased MF ---
    print("\nTraining biased MF (r=16, 10 epochs)...")
    all_meas = extract_measurements_from_sequences([list(s) for s in sequences])
    mf = BiasedMF(embed_dim=16, lr=0.01, reg=0.1)
    if all_meas:
        mf.train(all_meas, epochs=10, verbose=True)
    df["mf_pred"] = [
        mf.predict_rtt(r["src_key"], r["dst_key"]) or global_median
        for _, r in df.iterrows()
    ]

    # --- Vivaldi ---
    print("\nTraining Vivaldi (dim=4, 5 epochs)...")
    viv = fit_vivaldi(all_meas, dim=4, n_epochs=5)
    df["vivaldi_pred"] = [
        predict_vivaldi(viv, r["src_key"], r["dst_key"]) or global_median
        for _, r in df.iterrows()
    ]

    # --- TRMF ---
    trmf_meas = [
        (r["src_key"], r["dst_key"], r["actual_rtt_ms"], int(r["timestamp"]))
        for _, r in df.iterrows()
        if pd.notna(r["timestamp"])
    ]
    if len(trmf_meas) >= 100:
        print(f"\nTraining TRMF on {len(trmf_meas)} timestamped observations...")
        trmf = TRMF()
        trmf.fit(trmf_meas)
        trmf_preds = []
        for _, r in df.iterrows():
            ts = int(r["timestamp"]) if pd.notna(r["timestamp"]) else None
            pred = trmf.predict(r["src_key"], r["dst_key"], ts)
            trmf_preds.append(
                pred if pred is not None
                else (mf.predict_rtt(r["src_key"], r["dst_key"]) or global_median)
            )
        df["trmf_pred"] = trmf_preds
        np.savez(
            output_dir / "trmf_model.npz",
            F=trmf.F, X=trmf.X, W=trmf.W,
            row_mean=trmf.row_mean, row_std=trmf.row_std,
        )
    else:
        print(f"\nSkipping TRMF ({len(trmf_meas)} timestamped obs, need ≥100)")
        df["trmf_pred"] = global_median

    # --- Save ---
    df["src_key"] = df["src_key"].apply(ip_key_to_str)
    df["dst_key"] = df["dst_key"].apply(ip_key_to_str)

    df.to_parquet(output_dir / "observations.parquet", index=False)

    meta = {
        "test_data_path": str(test_data),
        "num_sequences": len(sequences),
        "num_observations": n_obs,
        "num_timestamped": n_timestamped,
        "global_median_rtt_ms": global_median,
        "num_unique_pairs": int(df[["src_key", "dst_key"]].drop_duplicates().shape[0]),
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (output_dir / "baselines_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nHarness complete → {output_dir} ({time.time() - t0:.1f}s)")


def main():
    p = argparse.ArgumentParser(description="Stage 1: evaluation harness")
    p.add_argument("--test-data", required=True)
    p.add_argument("--output-dir", default="outputs/eval_harness/default")
    p.add_argument("--num-sequences", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    run_harness(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
