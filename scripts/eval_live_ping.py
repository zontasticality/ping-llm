#!/usr/bin/env python3
"""
Live ping evaluation: Compare model's predicted latency distribution vs actual pings.

This script:
1. Generates 100 random IPv4 addresses
2. Pings each address 100 times to get empirical latency distribution
3. For each address, conditions the model on (src, dst) and samples latency predictions
4. Measures KL divergence between real distribution and model distribution
5. Tests both with and without timestamp conditioning

Usage (local CPU):
    python scripts/eval_live_ping.py \
        --checkpoint outputs/checkpoints/default/latest.pt \
        --num-ips 100 \
        --pings-per-ip 100 \
        --model-samples 100

Usage (Modal GPU):
    modal run scripts/eval_live_ping.py::eval_on_modal \
        --num-ips 10 \
        --pings-per-ip 20

Requires: root/sudo for raw ICMP pings (or uses socket-based ping as fallback)
"""

import os
from pathlib import Path

import argparse
import sys
import time
import socket
import struct
from collections import Counter
from datetime import datetime
import subprocess
import re

# Add project src to path (must be before imports from ping_llm).
repo_root = Path(__file__).resolve().parent.parent
workspace_root = Path("/workspace")

IN_MODAL_RUNTIME = bool(os.environ.get("MODAL_IS_REMOTE")) or (
    workspace_root / "src"
).exists()

if IN_MODAL_RUNTIME and (workspace_root / "src").exists():
    sys.path.insert(0, str(workspace_root / "src"))
else:
    sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import torch
from scipy.stats import entropy

from ping_llm.data.tokenization import (
    encode_ip_merged,
    encode_timestamp_delta,
    MEASUREMENT_START,
    RTT_START,
    token_to_byte,
    decode_rtt_exponent_mantissa,
)
from ping_llm.inference import load_model, generate

# ============================================================================
# Modal Setup (for GPU acceleration)
# ============================================================================
try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:
    APP_NAME = "ping-llm-eval-live-ping"
    WORKDIR = "/workspace"
    VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")

    IGNORE_PATTERNS = [
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "outputs",
        "logs",
        "data",
        "local_datasets",
        "archive",
        "tests",
        "docs",
        "benchmarks",
        "end_to_end",
        "*.parquet",
        "*.arrayrecord",
        ".DS_Store",
    ]

    # Build image in stages to optimize caching.
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "iputils-ping")
        .pip_install("uv")
        .run_commands("uv pip install --system torch pyarrow numpy scipy")
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    )

    app = modal.App(APP_NAME)
    shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DEFAULT_CHECKPOINT_MODAL = "/mnt/outputs/checkpoints/default/latest.pt"
DEFAULT_CHECKPOINT_LOCAL = "outputs/checkpoints/default/latest.pt"
DEFAULT_CHECKPOINT = (
    DEFAULT_CHECKPOINT_MODAL if IN_MODAL_RUNTIME else DEFAULT_CHECKPOINT_LOCAL
)


def ping_host(ip, timeout=2):
    """
    Ping a host using system ping command.

    Args:
        ip: IP address to ping
        timeout: Timeout in seconds

    Returns:
        RTT in milliseconds, or -1 if failed
    """
    try:
        # Use system ping (works without root)
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout), ip],
            capture_output=True,
            text=True,
            timeout=timeout + 1,
        )

        # Parse RTT from output
        # Format: "time=X.XXX ms" or "time=X ms"
        match = re.search(r"time[=\s]+(\d+\.?\d*)\s*ms", result.stdout)
        if match:
            return float(match.group(1))
        else:
            return -1.0  # Failed ping
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return -1.0


def generate_random_ipv4(count=100, seed=42):
    """
    Generate random public IPv4 addresses (avoiding private ranges).

    Returns:
        List of IPv4 address strings
    """
    np.random.seed(seed)
    ips = []

    # Use public IP ranges (simplified)
    # Avoiding: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8
    while len(ips) < count:
        # Generate random IP
        octets = np.random.randint(1, 255, size=4)

        # Skip private ranges
        if octets[0] == 10:
            continue
        if octets[0] == 172 and 16 <= octets[1] <= 31:
            continue
        if octets[0] == 192 and octets[1] == 168:
            continue
        if octets[0] == 127:
            continue

        ip = f"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}"
        ips.append(ip)

    return ips


def get_src_ip():
    """Get the source IP address of this machine."""
    try:
        # Connect to a public DNS server to determine our external IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        src_ip = s.getsockname()[0]
        s.close()
        return src_ip
    except:
        return "127.0.0.1"  # Fallback to localhost


def create_conditioning_tokens(src_ip, dst_ip, include_timestamp=False):
    """
    Create conditioning tokens for P(RTT | src, dst, [timestamp]).

    Args:
        src_ip: Source IP address string
        dst_ip: Destination IP address string
        include_timestamp: Whether to include timestamp

    Returns:
        List of token IDs representing the conditioning context
    """
    tokens = [MEASUREMENT_START]

    # Add src and dst
    tokens.extend(encode_ip_merged(src_ip, 4, is_src=True))
    tokens.extend(encode_ip_merged(dst_ip, 4, is_src=False))

    # Add timestamp if requested
    if include_timestamp:
        current_time = datetime.now()
        timestamp_tokens = encode_timestamp_delta(current_time, prev_time=None)
        tokens.extend(timestamp_tokens)

    # Add RTT_START token (model will predict the 2 RTT bytes next)
    tokens.append(RTT_START)

    return tokens


def pad_tokens(tokens, target_len):
    """Pad a token list to target_len with zeros."""
    seq = tokens[:target_len]
    padded = np.zeros(target_len, dtype=np.int32)
    padded[: len(seq)] = seq
    return padded, len(seq)


def sample_rtt_from_model(
    model,
    conditioning_tokens,
    num_samples=100,
    temperature=1.0,
    device="cpu",
):
    """
    Sample RTT values from the model's conditional distribution.

    Args:
        model: PyTorch GPT model
        conditioning_tokens: Context tokens (src, dst, RTT_START)
        num_samples: Number of samples to draw
        temperature: Sampling temperature
        device: Device string

    Returns:
        List of RTT values in milliseconds
    """
    rtt_samples = []

    for _ in range(num_samples):
        # Generate 2 tokens (RTT byte1 and byte2) after the conditioning context
        generated = generate(
            model,
            conditioning_tokens,
            max_new_tokens=2,
            temperature=temperature,
            device=device,
        )

        # Extract the last 2 generated tokens
        byte1_token = generated[-2]
        byte2_token = generated[-1]

        try:
            byte1 = token_to_byte(byte1_token)
            byte2 = token_to_byte(byte2_token)
            rtt_ms = decode_rtt_exponent_mantissa(byte1, byte2)
            rtt_samples.append(rtt_ms)
        except Exception:
            continue

    return rtt_samples


def discretize_rtt(rtt_values, bins=50, max_rtt=1000):
    """
    Discretize RTT values into bins for KL divergence calculation.

    Args:
        rtt_values: List of RTT values in milliseconds
        bins: Number of bins
        max_rtt: Maximum RTT to consider (ms)

    Returns:
        Probability distribution over bins
    """
    # Filter failed pings
    valid_rtts = [r for r in rtt_values if r >= 0]

    if len(valid_rtts) == 0:
        return np.zeros(bins)

    # Create histogram
    hist, _ = np.histogram(valid_rtts, bins=bins, range=(0, max_rtt))

    # Normalize to probability distribution
    prob = hist / hist.sum() if hist.sum() > 0 else np.zeros(bins)

    return prob


def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence KL(P || Q).

    Args:
        p: True distribution
        q: Model distribution
        epsilon: Smoothing factor to avoid log(0)

    Returns:
        KL divergence value
    """
    # Add smoothing
    p = p + epsilon
    q = q + epsilon

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return entropy(p, q)


def main():
    parser = argparse.ArgumentParser(
        description="Live ping evaluation with KL divergence"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--num-ips",
        type=int,
        default=10,
        help="Number of random IPs to test (default 10 for speed)",
    )
    parser.add_argument(
        "--pings-per-ip",
        type=int,
        default=20,
        help="Number of pings per IP (default 20 for speed)",
    )
    parser.add_argument(
        "--model-samples", type=int, default=100, help="Number of model samples per IP"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Get source IP
    src_ip = get_src_ip()
    print(f"\nSource IP: {src_ip}")

    # Generate random destination IPs
    print(f"\nGenerating {args.num_ips} random public IPv4 addresses...")
    dst_ips = generate_random_ipv4(count=args.num_ips, seed=args.seed)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.checkpoint} (device={device})...")
    model, model_cfg = load_model(args.checkpoint, device=device)
    print(f"Model loaded: {model.num_params:,} params")

    # Results storage
    results_no_ts = []
    results_with_ts = []

    print("\n" + "=" * 80)
    print("Starting live ping evaluation...")
    print("=" * 80)

    for idx, dst_ip in enumerate(dst_ips):
        print(f"\n[{idx+1}/{args.num_ips}] Testing {dst_ip}...")

        # Ping the host
        print(f"  Pinging {args.pings_per_ip} times...")
        real_rtts = []
        for _ in range(args.pings_per_ip):
            rtt = ping_host(dst_ip)
            real_rtts.append(rtt)

        success_count = sum(1 for r in real_rtts if r >= 0)
        print(f"  Real pings: {success_count}/{args.pings_per_ip} successful")

        if success_count < 5:
            print("  Skipping (too few successful pings)")
            continue

        # Get real distribution
        real_dist = discretize_rtt(real_rtts)

        # Test without timestamp
        print("  Sampling model (no timestamp)...")
        cond_tokens_no_ts = create_conditioning_tokens(
            src_ip, dst_ip, include_timestamp=False
        )
        model_rtts_no_ts = sample_rtt_from_model(
            model,
            cond_tokens_no_ts,
            num_samples=args.model_samples,
            temperature=args.temperature,
            device=device,
        )
        model_dist_no_ts = discretize_rtt(model_rtts_no_ts)
        kl_no_ts = kl_divergence(real_dist, model_dist_no_ts)

        # Test with timestamp
        print("  Sampling model (with timestamp)...")
        cond_tokens_with_ts = create_conditioning_tokens(
            src_ip, dst_ip, include_timestamp=True
        )
        model_rtts_with_ts = sample_rtt_from_model(
            model,
            cond_tokens_with_ts,
            num_samples=args.model_samples,
            temperature=args.temperature,
            device=device,
        )
        model_dist_with_ts = discretize_rtt(model_rtts_with_ts)
        kl_with_ts = kl_divergence(real_dist, model_dist_with_ts)

        print(f"  KL divergence (no timestamp):   {kl_no_ts:.4f}")
        print(f"  KL divergence (with timestamp):  {kl_with_ts:.4f}")

        results_no_ts.append(
            {
                "dst_ip": dst_ip,
                "kl": kl_no_ts,
                "real_mean": np.mean([r for r in real_rtts if r >= 0]),
                "model_mean": (
                    np.mean(model_rtts_no_ts) if model_rtts_no_ts else float("nan")
                ),
            }
        )

        results_with_ts.append(
            {
                "dst_ip": dst_ip,
                "kl": kl_with_ts,
                "real_mean": np.mean([r for r in real_rtts if r >= 0]),
                "model_mean": (
                    np.mean(model_rtts_with_ts)
                    if model_rtts_with_ts
                    else float("nan")
                ),
            }
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    if results_no_ts:
        kl_no_ts_vals = [r["kl"] for r in results_no_ts if not np.isnan(r["kl"])]
        kl_with_ts_vals = [r["kl"] for r in results_with_ts if not np.isnan(r["kl"])]

        print(f"\nKL Divergence (lower is better):")
        print(f"  WITHOUT timestamp:")
        print(f"    Mean: {np.mean(kl_no_ts_vals):.4f}")
        print(f"    Std:  {np.std(kl_no_ts_vals):.4f}")
        print(f"    Min:  {np.min(kl_no_ts_vals):.4f}")
        print(f"    Max:  {np.max(kl_no_ts_vals):.4f}")

        print(f"\n  WITH timestamp:")
        print(f"    Mean: {np.mean(kl_with_ts_vals):.4f}")
        print(f"    Std:  {np.std(kl_with_ts_vals):.4f}")
        print(f"    Min:  {np.min(kl_with_ts_vals):.4f}")
        print(f"    Max:  {np.max(kl_with_ts_vals):.4f}")

        # Compare means
        if np.mean(kl_no_ts_vals) < np.mean(kl_with_ts_vals):
            print(f"\n  Model performs better WITHOUT timestamp (lower KL)")
        else:
            print(f"\n  Model performs better WITH timestamp (lower KL)")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  - KL divergence measures how different the model's predicted distribution")
    print("    is from the real measured distribution")
    print("  - Lower KL = better match between model and reality")
    print("  - Compare 'with timestamp' vs 'without timestamp' to see if temporal")
    print("    information helps the model predict latencies")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================================
# Modal GPU Function
# ============================================================================
if MODAL_AVAILABLE:

    @app.function(
        image=image,
        gpu="A100",
        cpu=4,
        volumes={"/mnt": shared_vol},
        timeout=60 * 60 * 4,  # 4 hours (pinging takes time)
    )
    def eval_on_modal(
        checkpoint_path: str = DEFAULT_CHECKPOINT_MODAL,
        num_ips: int = 10,
        pings_per_ip: int = 20,
        model_samples: int = 100,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        """
        Run eval_live_ping on Modal with GPU acceleration.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            num_ips: Number of random IPs to test
            pings_per_ip: Number of pings per IP
            model_samples: Number of model samples per IP
            temperature: Sampling temperature
            seed: Random seed
        """
        import sys

        # Set up symlinks for paths
        os.makedirs(f"{WORKDIR}/outputs", exist_ok=True)

        if not os.path.exists(f"{WORKDIR}/outputs"):
            os.symlink("/mnt/outputs", f"{WORKDIR}/outputs")

        # Resolve checkpoint path
        if not checkpoint_path.startswith("/"):
            checkpoint_path = f"{WORKDIR}/{checkpoint_path}"

        # Prepare argv for main()
        sys.argv = [
            "eval_live_ping.py",
            "--checkpoint",
            checkpoint_path,
            "--num-ips",
            str(num_ips),
            "--pings-per-ip",
            str(pings_per_ip),
            "--model-samples",
            str(model_samples),
            "--temperature",
            str(temperature),
            "--seed",
            str(seed),
        ]

        # Change to workspace directory
        os.chdir(WORKDIR)

        # Run main function
        main()
