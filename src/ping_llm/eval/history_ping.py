"""
Test 2: History-conditioned live ping.

Give model 0-5 real measurement history, sample RTT predictions,
compare predicted mean/std to actual ping distribution.
"""

import subprocess
import re
import time
import socket
import numpy as np
from datetime import datetime

from ping_llm.data.tokenization import (
    MEASUREMENT_START, RTT_START,
    encode_ip_merged, encode_rtt_exponent_mantissa,
    encode_timestamp_delta, decode_rtt_exponent_mantissa,
    token_to_byte, BYTE_TOKEN_OFFSET, VOCAB_SIZE,
)
from ping_llm.inference import generate


# Targets: mix of near, medium, far
DEFAULT_TARGETS = [
    ("8.8.8.8", "Google DNS"),
    ("1.1.1.1", "Cloudflare DNS"),
    ("198.41.0.4", "Root DNS A"),
    ("9.9.9.9", "Quad9 DNS"),
    ("208.67.222.222", "OpenDNS"),
]


def ping_host(ip, timeout=2):
    """Ping a host, return RTT in ms or -1 on failure."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout), ip],
            capture_output=True, text=True, timeout=timeout + 1,
        )
        match = re.search(r"time[=\s]+(\d+\.?\d*)\s*ms", result.stdout)
        return float(match.group(1)) if match else -1.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return -1.0


def get_src_ip():
    """Get this machine's outbound IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "10.0.0.1"


def build_measurement_tokens(src_ip, dst_ip, rtt_ms, timestamp=None, prev_timestamp=None):
    """Build a complete measurement token sequence."""
    tokens = [MEASUREMENT_START]
    tokens.extend(encode_ip_merged(src_ip, 4, is_src=True))
    tokens.extend(encode_ip_merged(dst_ip, 4, is_src=False))
    if timestamp is not None:
        tokens.extend(encode_timestamp_delta(timestamp, prev_timestamp))
    tokens.extend(encode_rtt_exponent_mantissa(rtt_ms))
    return tokens


def build_query_context(src_ip, dst_ip, history_rtts, history_timestamps=None):
    """
    Build a token context with k history measurements + a query for the next RTT.

    Args:
        src_ip: source IP string
        dst_ip: destination IP string
        history_rtts: list of RTT values in ms (the k history measurements)
        history_timestamps: optional list of datetime objects

    Returns:
        list of token IDs ending with RTT_START (model predicts the 2 RTT bytes)
    """
    tokens = []
    prev_ts = None
    for i, rtt in enumerate(history_rtts):
        ts = history_timestamps[i] if history_timestamps else None
        tokens.extend(build_measurement_tokens(src_ip, dst_ip, rtt, ts, prev_ts))
        prev_ts = ts

    # Query measurement: src + dst + RTT_START (model must predict RTT bytes)
    tokens.append(MEASUREMENT_START)
    tokens.extend(encode_ip_merged(src_ip, 4, is_src=True))
    tokens.extend(encode_ip_merged(dst_ip, 4, is_src=False))
    if history_timestamps:
        tokens.extend(encode_timestamp_delta(datetime.now(), prev_ts))
    tokens.append(RTT_START)
    return tokens


def sample_rtt_from_model(model, context_tokens, num_samples=200, temperature=1.0, device="cpu"):
    """Sample RTT values by generating 2 tokens after context."""
    rtts = []
    for _ in range(num_samples):
        generated = generate(
            model, context_tokens,
            max_new_tokens=2, temperature=temperature, device=device,
        )
        byte1_token = generated[-2]
        byte2_token = generated[-1]
        try:
            byte1 = token_to_byte(byte1_token)
            byte2 = token_to_byte(byte2_token)
            rtt_ms = decode_rtt_exponent_mantissa(byte1, byte2)
            rtts.append(rtt_ms)
        except Exception:
            continue
    return rtts


def robust_stats(values, max_rtt=10000):
    """Compute robust stats: clip outliers, use median for center, IQR for spread."""
    arr = np.array(values)
    arr = arr[arr <= max_rtt]  # drop extreme outliers
    if len(arr) == 0:
        return 0.0, 0.0, 0
    median = float(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return median, iqr, len(arr)


def eval_history_ping(model, device="cpu", targets=None, pings_per_target=50,
                      model_samples=200, temperature=1.0, history_sizes=(0, 1, 2, 3, 5)):
    """
    Run the history-conditioned live ping evaluation.

    Returns:
        list of per-target result dicts
    """
    if targets is None:
        targets = DEFAULT_TARGETS

    src_ip = get_src_ip()
    print(f"  Source IP: {src_ip}")

    all_results = []

    for ip, label in targets:
        print(f"\n  [{label}] Pinging {ip} x{pings_per_target}...")
        real_rtts = []
        timestamps = []
        for i in range(pings_per_target):
            rtt = ping_host(ip)
            if rtt >= 0:
                real_rtts.append(rtt)
                timestamps.append(datetime.now())
            time.sleep(0.05)  # small delay between pings

        if len(real_rtts) < 5:
            print(f"    Only {len(real_rtts)} successful pings, skipping")
            continue

        real_median = float(np.median(real_rtts))
        real_iqr = float(np.percentile(real_rtts, 75) - np.percentile(real_rtts, 25))
        print(f"    Real: median={real_median:.2f}ms, IQR={real_iqr:.2f}ms (n={len(real_rtts)})")

        target_result = {
            "ip": ip, "label": label,
            "real_median": round(real_median, 3),
            "real_iqr": round(real_iqr, 3),
            "real_count": len(real_rtts),
            "history_results": {},
        }

        for k in history_sizes:
            if k > len(real_rtts) - 1:
                continue

            # Use the first k pings as history
            history = real_rtts[:k]
            history_ts = timestamps[:k] if timestamps else None

            context = build_query_context(src_ip, ip, history, history_ts)
            model_rtts = sample_rtt_from_model(
                model, context, num_samples=model_samples,
                temperature=temperature, device=device,
            )

            if not model_rtts:
                print(f"    k={k}: no valid samples")
                continue

            pred_median, pred_iqr, valid_n = robust_stats(model_rtts)
            median_err = abs(pred_median - real_median)
            pct_within_2x = sum(1 for r in model_rtts if real_median / 3 <= r <= real_median * 3) / len(model_rtts)

            target_result["history_results"][k] = {
                "pred_median": round(pred_median, 3),
                "pred_iqr": round(pred_iqr, 3),
                "median_err": round(float(median_err), 3),
                "pct_within_3x": round(float(pct_within_2x), 3),
                "valid_samples": len(model_rtts),
                "inlier_samples": valid_n,
            }
            print(f"    k={k}: pred median={pred_median:.1f}ms (err={median_err:.1f}), "
                  f"IQR={pred_iqr:.1f}ms, within 3x={pct_within_2x:.0%} "
                  f"({valid_n}/{len(model_rtts)} inliers)")

        all_results.append(target_result)

    return all_results


def print_history_ping(results):
    """Pretty-print history ping results."""
    print("\n" + "=" * 80)
    print("HISTORY-CONDITIONED LIVE PING")
    print("=" * 80)

    for r in results:
        print(f"\n  {r['label']} ({r['ip']}): real median={r['real_median']:.1f}ms, IQR={r['real_iqr']:.1f}ms")
        print(f"  {'k':>4} {'Pred Med':>10} {'Med Err':>10} {'Pred IQR':>10} {'Within 3x':>10} {'Inliers':>10}")
        print(f"  {'-'*58}")
        for k in sorted(r["history_results"].keys()):
            h = r["history_results"][k]
            print(f"  {k:>4} {h['pred_median']:>8.1f}ms {h['median_err']:>8.1f}ms "
                  f"{h['pred_iqr']:>8.1f}ms {h['pct_within_3x']:>9.0%} "
                  f"{h['inlier_samples']:>5}/{h['valid_samples']}")

    # Aggregate summary
    if not results:
        return
    all_ks = set()
    for r in results:
        all_ks.update(r["history_results"].keys())

    print(f"\n  AGGREGATE (median across targets):")
    print(f"  {'k':>4} {'Med Median Err':>16} {'Med Within 3x':>16}")
    print(f"  {'-'*40}")
    for k in sorted(all_ks):
        med_errs = [r["history_results"][k]["median_err"] for r in results if k in r["history_results"]]
        within3x = [r["history_results"][k]["pct_within_3x"] for r in results if k in r["history_results"]]
        if med_errs:
            print(f"  {k:>4} {np.median(med_errs):>14.1f}ms {np.median(within3x):>15.0%}")
    print("=" * 80)
