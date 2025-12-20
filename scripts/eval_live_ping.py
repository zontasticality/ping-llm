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
        --checkpoint checkpoints/full_run/checkpoints/2000 \
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

# Conditionally force CPU usage (only for local runs, not Modal).
def _in_modal_runtime():
    return bool(os.environ.get("MODAL_IS_REMOTE")) or (Path("/workspace") / "src").exists()


IN_MODAL_RUNTIME = _in_modal_runtime()
if not IN_MODAL_RUNTIME:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import argparse
import sys
import time
import socket
import struct
from collections import Counter
from datetime import datetime
import subprocess
import re

# Add project root to path (must be before imports from src).
repo_root = Path(__file__).resolve().parent.parent
workspace_root = Path("/workspace")
if IN_MODAL_RUNTIME and (workspace_root / "src").exists():
    sys.path.insert(0, str(workspace_root))
else:
    sys.path.insert(0, str(repo_root))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import entropy

from src.MaxText.input_pipeline.network_tokenization import (
    encode_ip_merged, encode_timestamp_delta,
    MEASUREMENT_START, SRC_IPV4, DST_IPV4, RTT_START,
    byte_to_token, token_to_byte, encode_rtt_exponent_mantissa,
    decode_rtt_exponent_mantissa,
)
from src.MaxText import pyconfig, model_creation_utils, maxtext_utils, checkpointing, max_utils
from src.MaxText.inference_utils import sampling
from flax.linen import partitioning as nn_partitioning

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
        ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
        "outputs", "logs", "data", "local_datasets", "archive",
        "tests", "docs", "benchmarks", "end_to_end",
        "*.parquet", "*.arrayrecord", ".DS_Store",
    ]

    # Build image in stages to optimize caching.
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "cmake", "ninja-build", "iputils-ping")  # Add ping
        .pip_install("uv")
        # Stage 1: Dependency files
        .add_local_file("pyproject.toml", f"{WORKDIR}/pyproject.toml", copy=True)
        .add_local_file("README.md", f"{WORKDIR}/README.md", copy=True)
        .add_local_file("build_hooks.py", f"{WORKDIR}/build_hooks.py", copy=True)
        .add_local_dir("dependencies", f"{WORKDIR}/dependencies", copy=True)
        .add_local_file("src/MaxText/__init__.py", f"{WORKDIR}/src/MaxText/__init__.py", copy=True)
        .add_local_dir("src/install_maxtext_extra_deps", f"{WORKDIR}/src/install_maxtext_extra_deps", copy=True)
        # Stage 2: Install dependencies
        .run_commands(
            f"cd {WORKDIR} && CC=gcc CXX=g++ uv pip install --system -e '.[cuda12]' --resolution=lowest",
            f"cd {WORKDIR} && install_maxtext_github_deps",
        )
        .pip_install("scipy")
        # Stage 3: Copy code
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    )

    app = modal.App(APP_NAME)
    shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_config(checkpoint_path, config_path, use_gpu=False):
    """Build a MaxText config with resolved paths and eval overrides."""
    checkpoint_path = str(Path(checkpoint_path).resolve())
    config_path = str(Path(config_path).resolve())

    # pyconfig.initialize expects argv[0] to be program name, argv[1] to be config path,
    # and argv[2:] to be key=value overrides
    argv = [
        "eval_script",  # argv[0] - program name (ignored)
        config_path,    # argv[1] - config file path
        f"load_full_state_path={checkpoint_path}",  # Use load_full_state_path for full checkpoint
        f"hardware={'gpu' if use_gpu else 'cpu'}",
        "skip_jax_distributed_system=true",
    ]

    # Use simpler attention for CPU
    if not use_gpu:
        argv.append("attention=dot_product")

    config = pyconfig.initialize(argv)
    return config, checkpoint_path


def load_model_and_params(checkpoint_path, config):
    """Load model and parameters from checkpoint using MaxText's native approach."""
    # Create model
    model = model_creation_utils.from_config(config)
    mesh = model.mesh

    # Get abstract params structure (just shapes, no actual arrays - minimal memory)
    print("Getting abstract parameter structure...")
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        abstract_params = maxtext_utils.get_abstract_param(model, config)

    # Load ONLY params from checkpoint using Orbax directly (skip optimizer state)
    print(f"Loading parameters only from checkpoint (this saves memory)...")
    import orbax.checkpoint as ocp
    from etils import epath

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        # Create checkpointer
        ckptr = ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(
                restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
                use_ocdbt=True,
                use_zarr3=True,
            )
        )

        # Restore only the params field from items/params
        # The checkpoint structure is: checkpoint/items/{params, opt_state, step, ...}
        # We only want params
        restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_params['params'])

        # Restore from checkpoint_path/items (the items subdirectory)
        checkpoint_items_path = epath.Path(checkpoint_path) / "items"

        restored = ckptr.restore(
            checkpoint_items_path,
            item={'params': abstract_params['params']},
            transforms={},
            restore_args={'params': restore_args},
        )

        params = restored['params']

    print("✓ Parameters loaded successfully (optimizer state skipped to save memory)")
    return model, params, mesh


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
            ['ping', '-c', '1', '-W', str(timeout), ip],
            capture_output=True,
            text=True,
            timeout=timeout + 1
        )

        # Parse RTT from output
        # Format: "time=X.XXX ms" or "time=X ms"
        match = re.search(r'time[=\s]+(\d+\.?\d*)\s*ms', result.stdout)
        if match:
            return float(match.group(1))
        else:
            return -1.0  # Failed ping
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
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


def sample_rtt_from_model(model, params, config, conditioning_tokens, num_samples=100, temperature=1.0):
    """
    Sample RTT values from the model's conditional distribution.

    Args:
        model: Flax model
        params: Model parameters
        config: Config
        conditioning_tokens: Context tokens (src, dst, RTT_START)
        num_samples: Number of samples to draw
        temperature: Sampling temperature

    Returns:
        List of RTT values in milliseconds
    """
    rtt_samples = []
    rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**31)
    max_len = config.max_target_length

    # Sample num_samples 2-grams (byte1, byte2) for RTT
    for i in range(num_samples):
        rng, rng_byte1, rng_byte2 = jax.random.split(rng, 3)

        # Prepare inputs for first byte
        tokens_list = conditioning_tokens
        input_tokens = jnp.array([tokens_list], dtype=jnp.int32)
        positions = jnp.arange(len(tokens_list), dtype=jnp.int32)[None, :]
        segment_ids = jnp.ones((1, len(tokens_list)), dtype=jnp.int32)

        # Pad to max length
        if len(tokens_list) < max_len:
            pad_len = max_len - len(tokens_list)
            input_tokens = jnp.pad(input_tokens, ((0, 0), (0, pad_len)), constant_values=0)
            positions = jnp.pad(positions, ((0, 0), (0, pad_len)), constant_values=0)
            segment_ids = jnp.pad(segment_ids, ((0, 0), (0, pad_len)), constant_values=0)

        # Get logits for first RTT byte
        logits, _ = model.apply(
            params,
            input_tokens,
            positions,
            decoder_segment_ids=segment_ids,
            enable_dropout=False,
            rngs={"dropout": rng_byte1, "params": rng_byte1},
            mutable=["intermediates"],
        )

        # Sample first RTT byte (predict token after last position)
        logits_byte1 = logits[0, len(tokens_list) - 1, :]
        byte1_token = sampling(logits_byte1, rng_byte1, "weighted", temperature=temperature)
        byte1_token = int(byte1_token)

        # Add byte1 to sequence and get logits for second byte
        tokens_list2 = conditioning_tokens + [byte1_token]
        input_tokens2 = jnp.array([tokens_list2], dtype=jnp.int32)
        positions2 = jnp.arange(len(tokens_list2), dtype=jnp.int32)[None, :]
        segment_ids2 = jnp.ones((1, len(tokens_list2)), dtype=jnp.int32)

        # Pad to max length
        if len(tokens_list2) < max_len:
            pad_len = max_len - len(tokens_list2)
            input_tokens2 = jnp.pad(input_tokens2, ((0, 0), (0, pad_len)), constant_values=0)
            positions2 = jnp.pad(positions2, ((0, 0), (0, pad_len)), constant_values=0)
            segment_ids2 = jnp.pad(segment_ids2, ((0, 0), (0, pad_len)), constant_values=0)

        logits2, _ = model.apply(
            params,
            input_tokens2,
            positions2,
            decoder_segment_ids=segment_ids2,
            enable_dropout=False,
            rngs={"dropout": rng_byte2, "params": rng_byte2},
            mutable=["intermediates"],
        )

        # Sample second RTT byte
        logits_byte2 = logits2[0, len(tokens_list2) - 1, :]
        byte2_token = sampling(logits_byte2, rng_byte2, "weighted", temperature=temperature)
        byte2_token = int(byte2_token)

        # Decode RTT from 2-byte encoding
        try:
            byte1 = token_to_byte(byte1_token)
            byte2 = token_to_byte(byte2_token)
            rtt_ms = decode_rtt_exponent_mantissa(byte1, byte2)
            rtt_samples.append(rtt_ms)
        except:
            # Invalid token, skip
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
    parser = argparse.ArgumentParser(description="Live ping evaluation with KL divergence")
    parser.add_argument("--checkpoint",
                       default="outputs/latency_network/full_run/full_run/checkpoints/2000",
                       help="Path to checkpoint directory")
    parser.add_argument("--config", default="src/MaxText/configs/latency_network.yml",
                       help="Config file path")
    parser.add_argument("--num-ips", type=int, default=10,
                       help="Number of random IPs to test (default 10 for speed)")
    parser.add_argument("--pings-per-ip", type=int, default=20,
                       help="Number of pings per IP (default 20 for speed)")
    parser.add_argument("--model-samples", type=int, default=100,
                       help="Number of model samples per IP")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading model and checkpoint...")
    use_gpu = IN_MODAL_RUNTIME or os.environ.get("JAX_PLATFORMS") == "gpu"
    config, checkpoint_path = build_config(args.checkpoint, args.config, use_gpu=use_gpu)
    with max_utils.maybe_get_transformer_engine_context(config):
        model, params, mesh = load_model_and_params(checkpoint_path, config)
        print(f"✓ Model loaded from {checkpoint_path}")

        # Get source IP
        src_ip = get_src_ip()
        print(f"\nSource IP: {src_ip}")

        # Generate random destination IPs
        print(f"\nGenerating {args.num_ips} random public IPv4 addresses...")
        dst_ips = generate_random_ipv4(count=args.num_ips, seed=args.seed)

        # Results storage
        results_no_ts = []
        results_with_ts = []

        print("\n" + "="*80)
        print("Starting live ping evaluation...")
        print("="*80)

        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
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
                cond_tokens_no_ts = create_conditioning_tokens(src_ip, dst_ip, include_timestamp=False)
                model_rtts_no_ts = sample_rtt_from_model(
                    model, params, config, cond_tokens_no_ts,
                    num_samples=args.model_samples, temperature=args.temperature
                )
                model_dist_no_ts = discretize_rtt(model_rtts_no_ts)
                kl_no_ts = kl_divergence(real_dist, model_dist_no_ts)

                # Test with timestamp
                print("  Sampling model (with timestamp)...")
                cond_tokens_with_ts = create_conditioning_tokens(src_ip, dst_ip, include_timestamp=True)
                model_rtts_with_ts = sample_rtt_from_model(
                    model, params, config, cond_tokens_with_ts,
                    num_samples=args.model_samples, temperature=args.temperature
                )
                model_dist_with_ts = discretize_rtt(model_rtts_with_ts)
                kl_with_ts = kl_divergence(real_dist, model_dist_with_ts)

                print(f"  KL divergence (no timestamp):   {kl_no_ts:.4f}")
                print(f"  KL divergence (with timestamp):  {kl_with_ts:.4f}")

                results_no_ts.append({
                    'dst_ip': dst_ip,
                    'kl': kl_no_ts,
                    'real_mean': np.mean([r for r in real_rtts if r >= 0]),
                    'model_mean': np.mean(model_rtts_no_ts) if model_rtts_no_ts else float('nan'),
                })

                results_with_ts.append({
                    'dst_ip': dst_ip,
                    'kl': kl_with_ts,
                    'real_mean': np.mean([r for r in real_rtts if r >= 0]),
                    'model_mean': np.mean(model_rtts_with_ts) if model_rtts_with_ts else float('nan'),
                })

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)

    if results_no_ts:
        kl_no_ts_vals = [r['kl'] for r in results_no_ts if not np.isnan(r['kl'])]
        kl_with_ts_vals = [r['kl'] for r in results_with_ts if not np.isnan(r['kl'])]

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
            print(f"\n✓ Model performs better WITHOUT timestamp (lower KL)")
        else:
            print(f"\n✓ Model performs better WITH timestamp (lower KL)")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("  - KL divergence measures how different the model's predicted distribution")
    print("    is from the real measured distribution")
    print("  - Lower KL = better match between model and reality")
    print("  - Compare 'with timestamp' vs 'without timestamp' to see if temporal")
    print("    information helps the model predict latencies")
    print("="*80)


if __name__ == "__main__":
    main()


# ============================================================================
# Modal GPU Function
# ============================================================================
if MODAL_AVAILABLE:
    @app.function(
        image=image,
        gpu="A100",  # Use A100 for fast inference
        cpu=4,
        volumes={"/mnt": shared_vol},
        timeout=60 * 60 * 4,  # 4 hours (pinging takes time)
        env={
            # Suppress TensorFlow/CUDA plugin registration warnings
            "TF_CPP_MIN_LOG_LEVEL": "2",
            # Suppress redundant XLA warnings
            "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=1",
        },
    )
    def eval_on_modal(
        checkpoint_dir: str = "full_run/full_run/checkpoints/2000",
        config: str = "src/MaxText/configs/latency_network.yml",
        num_ips: int = 10,
        pings_per_ip: int = 20,
        model_samples: int = 100,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        """
        Run eval_live_ping on Modal with GPU acceleration.

        Args:
            checkpoint_dir: Relative path within /mnt/outputs (e.g., "full_run/full_run/checkpoints/2000")
            config: Config file path (relative to workspace)
            num_ips: Number of random IPs to test
            pings_per_ip: Number of pings per IP
            model_samples: Number of model samples per IP
            temperature: Sampling temperature
            seed: Random seed
        """
        import sys

        # Set up symlinks for config paths
        os.makedirs(f"{WORKDIR}/outputs", exist_ok=True)

        # Link outputs from volume
        if not os.path.exists(f"{WORKDIR}/outputs/latency_network"):
            os.symlink("/mnt/outputs/latency_network", f"{WORKDIR}/outputs/latency_network")

        # Build checkpoint path
        checkpoint_path = f"/mnt/outputs/latency_network/{checkpoint_dir}"
        config_path = f"{WORKDIR}/{config}"

        # Prepare argv for main()
        sys.argv = [
            "eval_live_ping.py",
            "--checkpoint", checkpoint_path,
            "--config", config_path,
            "--num-ips", str(num_ips),
            "--pings-per-ip", str(pings_per_ip),
            "--model-samples", str(model_samples),
            "--temperature", str(temperature),
            "--seed", str(seed),
        ]

        # Set environment to indicate GPU usage
        os.environ["JAX_PLATFORMS"] = "gpu"

        # Change to workspace directory
        os.chdir(WORKDIR)

        # Run main function
        main()
