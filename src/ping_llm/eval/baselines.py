"""
Test 3: Baseline comparison.

Compare model RTT predictions to simple baselines on test sequences:
  - Global median RTT
  - Last-seen RTT
  - Window mean (last 3)
  - Exponential moving average (alpha=0.3)

Uses single forward pass per sequence (no sampling) for efficiency.
The model's score is the CE loss it assigns to the actual RTT bytes.
For baselines, we encode their predicted RTT and check what CE loss the
model would assign (plus compute MAE vs actual).
"""

import math
import struct
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from ping_llm.data.tokenization import (
    MEASUREMENT_START, RTT_START, FAILED,
    SRC_IPV4, SRC_IPV6, DST_IPV4, DST_IPV6,
    TIMESTAMP_ABS, TIMESTAMP_DELTA1, TIMESTAMP_DELTA4,
    decode_rtt_exponent_mantissa, encode_rtt_exponent_mantissa,
    token_to_byte, byte_to_token,
    BYTE_TOKEN_OFFSET, VOCAB_SIZE,
)
from ping_llm.eval.token_classify import ROLE_BYTE_COUNTS


def extract_rtt_positions(tokens):
    """
    Find all RTT prediction positions in a token sequence.

    Returns list of dicts:
        [{rtt_ms, byte1_pos, byte2_pos, measurement_index,
          src_key, dst_key, timestamp}, ...]
    where byte1_pos/byte2_pos are the indices of the 2 RTT byte tokens,
    src_key/dst_key are hashable IP identifiers, and timestamp is Unix
    seconds (int) or None if the measurement had no timestamp token.

    Positions are buffered per-measurement and flushed at the next
    MEASUREMENT_START (or end of sequence) so that the timestamp is
    attached even when the timestamp token follows the RTT token
    (field blocks are shuffled during encoding).
    """
    positions = []
    i = 0
    n = len(tokens)
    meas_idx = -1
    cur_src = None
    cur_dst = None
    current_time_sec = None
    meas_had_timestamp = False
    meas_buf = []

    def _flush():
        nonlocal meas_buf, meas_had_timestamp
        ts = current_time_sec if meas_had_timestamp else None
        for pos in meas_buf:
            pos["timestamp"] = ts
            positions.append(pos)
        meas_buf = []
        meas_had_timestamp = False

    while i < n:
        t = int(tokens[i])

        if t == MEASUREMENT_START:
            _flush()
            meas_idx += 1
            cur_src = None
            cur_dst = None
            i += 1
            continue

        if t in (SRC_IPV4, SRC_IPV6):
            nbytes = 4 if t == SRC_IPV4 else 16
            if i + nbytes < n:
                cur_src = (t, tuple(int(tokens[j]) for j in range(i+1, i+1+nbytes)))
            i += 1 + nbytes
            continue

        if t in (DST_IPV4, DST_IPV6):
            nbytes = 4 if t == DST_IPV4 else 16
            if i + nbytes < n:
                cur_dst = (t, tuple(int(tokens[j]) for j in range(i+1, i+1+nbytes)))
            i += 1 + nbytes
            continue

        if t in (TIMESTAMP_ABS, TIMESTAMP_DELTA1, TIMESTAMP_DELTA4):
            nbytes = ROLE_BYTE_COUNTS[t]
            if i + nbytes < n:
                data = [int(tokens[j]) for j in range(i + 1, i + 1 + nbytes)]
                if t == TIMESTAMP_ABS:
                    current_time_sec = struct.unpack(
                        ">Q", bytes(token_to_byte(x) for x in data),
                    )[0]
                    meas_had_timestamp = True
                elif t == TIMESTAMP_DELTA1:
                    if current_time_sec is not None:
                        current_time_sec += token_to_byte(data[0])
                        meas_had_timestamp = True
                else:
                    if current_time_sec is not None:
                        current_time_sec += struct.unpack(
                            ">I", bytes(token_to_byte(x) for x in data),
                        )[0]
                        meas_had_timestamp = True
            i += 1 + nbytes
            continue

        if t == FAILED:
            meas_buf.append({
                "rtt_ms": -1.0,
                "byte1_pos": None,
                "byte2_pos": None,
                "measurement_index": meas_idx,
                "src_key": cur_src,
                "dst_key": cur_dst,
            })
            i += 1
            continue

        if t == RTT_START and i + 2 < n:
            try:
                byte1 = token_to_byte(int(tokens[i + 1]))
                byte2 = token_to_byte(int(tokens[i + 2]))
                rtt_ms = decode_rtt_exponent_mantissa(byte1, byte2)
                meas_buf.append({
                    "rtt_ms": rtt_ms,
                    "byte1_pos": i + 1,
                    "byte2_pos": i + 2,
                    "measurement_index": meas_idx,
                    "src_key": cur_src,
                    "dst_key": cur_dst,
                })
            except Exception:
                pass
            i += 3
            continue

        # Skip other role tokens and their byte payloads
        if t in ROLE_BYTE_COUNTS:
            i += 1 + ROLE_BYTE_COUNTS[t]
        else:
            i += 1

    _flush()
    return positions


def model_top1_rtt(logits_byte1, logits_byte2):
    """
    Get the model's most likely RTT from logits at the 2 RTT byte positions.

    Args:
        logits_byte1: logits tensor of shape [V] at RTT byte1 position
        logits_byte2: logits tensor of shape [V] at RTT byte2 position

    Returns:
        predicted RTT in ms, or None if invalid
    """
    pred_token1 = int(logits_byte1.argmax())
    pred_token2 = int(logits_byte2.argmax())
    try:
        byte1 = token_to_byte(pred_token1)
        byte2 = token_to_byte(pred_token2)
        return decode_rtt_exponent_mantissa(byte1, byte2)
    except Exception:
        return None


def rtt_to_byte_tokens(rtt_ms):
    """Encode an RTT value to its 2 byte token IDs."""
    encoded = encode_rtt_exponent_mantissa(rtt_ms)
    if len(encoded) == 3:  # [RTT_START, byte1, byte2]
        return encoded[1], encoded[2]
    return None, None


def eval_baselines(model, sequences, device="cpu", max_sequences=100, **kwargs):
    """
    Compare model RTT predictions to baselines using single forward pass.

    For each sequence:
    1. Single forward pass → logits at every position
    2. At each RTT byte position, extract model's top-1 prediction
    3. Compare model and baseline predictions to actual RTT

    Returns:
        dict of {method: {count, mae, median_ae, log2_err}}
    """
    # First pass: collect all RTTs for global median
    all_rtts = []
    parsed = []
    for tokens in sequences[:max_sequences]:
        positions = extract_rtt_positions(tokens)
        valid = [p for p in positions if p["rtt_ms"] > 0 and p["byte1_pos"] is not None]
        parsed.append((tokens, valid))
        all_rtts.extend(p["rtt_ms"] for p in valid)

    if not all_rtts:
        print("  No valid RTT measurements found")
        return {}

    global_median = float(np.median(all_rtts))
    print(f"  Global median RTT: {global_median:.2f}ms (from {len(all_rtts)} measurements)")

    # Train MF baseline on all test data
    from ping_llm.eval.mf_baseline import BiasedMF, extract_measurements_from_sequences
    mf_measurements = extract_measurements_from_sequences(sequences[:max_sequences])
    mf_model = BiasedMF(embed_dim=16, lr=0.01, reg=0.1)
    if mf_measurements:
        print(f"  Training MF baseline on {len(mf_measurements)} measurements...")
        mf_model.train(mf_measurements, epochs=10, verbose=True)

    methods = ["global_median", "last_seen", "window_mean_3", "ema_0.3", "mf_biased", "model_top1"]
    errors = {m: {"abs": [], "log2": []} for m in methods}

    eval_count = 0
    for seq_idx, (tokens, positions) in enumerate(parsed):
        if len(positions) < 2:
            continue

        token_list = [int(t) for t in tokens]

        # Single forward pass for the whole sequence
        idx = torch.tensor([token_list], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx)  # [1, T, V]
        logits = logits[0]  # [T, V]

        ema = positions[0]["rtt_ms"]
        alpha = 0.3

        for i in range(1, len(positions)):
            actual_rtt = positions[i]["rtt_ms"]
            if actual_rtt <= 0:
                continue

            byte1_pos = positions[i]["byte1_pos"]
            byte2_pos = positions[i]["byte2_pos"]
            if byte1_pos is None:
                continue

            history_rtts = [p["rtt_ms"] for p in positions[:i] if p["rtt_ms"] > 0]
            if not history_rtts:
                continue

            # Model's top-1 prediction from logits
            # logits[byte1_pos - 1] predicts the token at byte1_pos
            model_rtt = model_top1_rtt(
                logits[byte1_pos - 1],
                logits[byte2_pos - 1],
            )

            mf_pred = None
            src_key = positions[i].get("src_key")
            dst_key = positions[i].get("dst_key")
            if src_key and dst_key:
                mf_pred = mf_model.predict_rtt(src_key, dst_key)

            preds = {
                "global_median": global_median,
                "last_seen": history_rtts[-1],
                "window_mean_3": float(np.mean(history_rtts[-3:])),
                "ema_0.3": ema,
                "mf_biased": mf_pred if mf_pred is not None else global_median,
                "model_top1": model_rtt if model_rtt is not None else global_median,
            }

            for method, pred in preds.items():
                if pred <= 0:
                    pred = 0.001
                abs_err = abs(pred - actual_rtt)
                log2_err = abs(math.log2(pred / actual_rtt)) if actual_rtt > 0 and pred > 0 else 10.0
                errors[method]["abs"].append(abs_err)
                errors[method]["log2"].append(log2_err)

            ema = alpha * actual_rtt + (1 - alpha) * ema
            eval_count += 1

        if (seq_idx + 1) % 20 == 0:
            print(f"  Processed {seq_idx + 1}/{len(parsed)} sequences ({eval_count} predictions)")

    print(f"  Total predictions: {eval_count}")

    results = {}
    for method in methods:
        abs_errors = errors[method]["abs"]
        log2_errors = errors[method]["log2"]
        if not abs_errors:
            continue
        results[method] = {
            "count": len(abs_errors),
            "mae": round(float(np.mean(abs_errors)), 3),
            "median_ae": round(float(np.median(abs_errors)), 3),
            "log2_err_mean": round(float(np.mean(log2_errors)), 3),
            "log2_err_median": round(float(np.median(log2_errors)), 3),
        }

    return results


def print_baselines(results):
    """Pretty-print baseline comparison results."""
    print("\n" + "=" * 75)
    print("BASELINE COMPARISON: RTT PREDICTION")
    print("=" * 75)
    print(f"{'Method':<18} {'Count':>8} {'MAE (ms)':>10} {'Median AE':>10} {'Log2 Mean':>10} {'Log2 Med':>10}")
    print("-" * 75)

    display_order = ["global_median", "last_seen", "window_mean_3", "ema_0.3", "mf_biased", "model_top1"]
    for method in display_order:
        if method not in results:
            continue
        r = results[method]
        marker = " >> " if method == "model_top1" else "    "
        print(f"{marker}{method:<14} {r['count']:>8,} {r['mae']:>10.2f} {r['median_ae']:>10.2f} "
              f"{r['log2_err_mean']:>10.3f} {r['log2_err_median']:>10.3f}")
    print("=" * 75)
