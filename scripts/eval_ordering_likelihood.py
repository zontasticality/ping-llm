#!/usr/bin/env python3
"""
Evaluate how the model continues sequences when different fields end a measurement.

This script:
1. Loads a trained checkpoint
2. Samples random measurements from training_data.parquet
3. Builds 4 variants per measurement where the final field is:
   timestamp, src_ip, dst_ip, or latency (rtt)
4. Generates tokens in batch until MEASUREMENT_START is emitted or 20 tokens elapse
5. Reports stop rates and average tokens-to-stop per variant

Usage (local CPU):
    python scripts/eval_ordering_likelihood.py \
        --checkpoint checkpoints/full_run/checkpoints/2000 \
        --data data/training_data.parquet \
        --num-samples 100 \
        --max-new-tokens 20

Usage (Modal GPU):
    modal run scripts/eval_ordering_likelihood.py::eval_on_modal \
        --num-samples 100
"""

import os
from pathlib import Path


# Conditionally force CPU usage (only for local runs, not Modal).
def _in_modal_runtime():
    return (
        bool(os.environ.get("MODAL_IS_REMOTE")) or (Path("/workspace") / "src").exists()
    )


IN_MODAL_RUNTIME = _in_modal_runtime()
if not IN_MODAL_RUNTIME:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import argparse
import sys
import time
import math
import struct
from datetime import datetime
from collections import defaultdict

# Add project root to path (must be before imports from src).
repo_root = Path(__file__).resolve().parent.parent
workspace_root = Path("/workspace")
if IN_MODAL_RUNTIME and (workspace_root / "src").exists():
    sys.path.insert(0, str(workspace_root))
else:
    sys.path.insert(0, str(repo_root))

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np

from src.MaxText.input_pipeline.network_tokenization import (
    MEASUREMENT_START,
    FAILED,
    decode_token_stream_pretty,
    decode_tokens_to_measurements,
    token_to_byte,
    SRC_IPV4,
    SRC_IPV6,
    DST_IPV4,
    DST_IPV6,
    RTT_START,
    TIMESTAMP_ABS,
    TIMESTAMP_DELTA1,
    TIMESTAMP_DELTA4,
)

# ============================================================================
# Modal Setup (for GPU acceleration)
# ============================================================================
try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:
    APP_NAME = "ping-llm-eval-ordering"
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
        .apt_install("git", "build-essential", "cmake", "ninja-build")
        .pip_install("uv")
        # Stage 1: Dependency files
        .add_local_file("pyproject.toml", f"{WORKDIR}/pyproject.toml", copy=True)
        .add_local_file("README.md", f"{WORKDIR}/README.md", copy=True)
        .add_local_file("build_hooks.py", f"{WORKDIR}/build_hooks.py", copy=True)
        .add_local_dir("dependencies", f"{WORKDIR}/dependencies", copy=True)
        .add_local_file(
            "src/MaxText/__init__.py", f"{WORKDIR}/src/MaxText/__init__.py", copy=True
        )
        .add_local_dir(
            "src/install_maxtext_extra_deps",
            f"{WORKDIR}/src/install_maxtext_extra_deps",
            copy=True,
        )
        # Stage 2: Install dependencies
        .run_commands(
            f"cd {WORKDIR} && CC=gcc CXX=g++ uv pip install --system -e '.[cuda12]' --resolution=lowest",
            f"cd {WORKDIR} && install_maxtext_github_deps",
            "uv pip install --system 'google-jetstream @ https://github.com/AI-Hypercomputer/JetStream/archive/29329e8e73820993f77cfc8efe34eb2a73f5de98.zip' --resolution=lowest",
        )
        .uv_pip_install("pandas", "pyarrow")
        # Stage 3: Copy code
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    )

    app = modal.App(APP_NAME)
    shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

PARAM_ONLY_CHECKPOINT_MODAL = (
    "/mnt/outputs/latency_network/param_only_checkpoint/checkpoints/0/items"
)
PARAM_ONLY_CHECKPOINT_LOCAL = (
    "outputs/latency_network/param_only_checkpoint/checkpoints/0/items"
)
DEFAULT_CHECKPOINT = (
    PARAM_ONLY_CHECKPOINT_MODAL if IN_MODAL_RUNTIME else PARAM_ONLY_CHECKPOINT_LOCAL
)


BASE_FIELDS = ["src", "dst", "rtt", "timestamp"]
VARIANT_END_FIELDS = [
    ("timestamp", "end_timestamp"),
    ("src", "end_src_ip"),
    ("dst", "end_dst_ip"),
    ("rtt", "end_latency"),
]


def create_ordered_measurement(row, field_order, prev_timestamp=None):
    """
    Create a measurement with a specific field ordering.

    Args:
        row: Pandas row with measurement data
        field_order: List of field names in desired order
        prev_timestamp: Previous timestamp for delta encoding

    Returns:
        Tuple of (tokens, field_boundaries) where:
        - tokens: List of token IDs with fields in specified order
        - field_boundaries: Dict mapping field name to (start_idx, end_idx) in tokens
    """
    from src.MaxText.input_pipeline.network_tokenization import (
        encode_ip_merged,
        encode_rtt_exponent_mantissa,
        encode_timestamp_delta,
    )

    # Build field blocks
    field_blocks = {}
    field_blocks["src"] = encode_ip_merged(
        row["src_addr"], row["ip_version"], is_src=True
    )
    field_blocks["dst"] = encode_ip_merged(
        row["dst_addr"], row["ip_version"], is_src=False
    )

    if row["rtt"] < 0:
        field_blocks["rtt"] = [FAILED]
    else:
        field_blocks["rtt"] = encode_rtt_exponent_mantissa(row["rtt"])

    field_blocks["timestamp"] = encode_timestamp_delta(
        row["event_time"], prev_timestamp
    )

    # Build token sequence in specified order and track boundaries
    tokens = [MEASUREMENT_START]
    field_boundaries = {}
    current_pos = len(tokens)

    for field in field_order:
        if field in field_blocks:
            start_idx = current_pos
            tokens.extend(field_blocks[field])
            end_idx = len(tokens)
            field_boundaries[field] = (start_idx, end_idx)
            current_pos = end_idx

    return tokens, field_boundaries


def build_measurement_variants(row_dict):
    """
    Return metadata for the 4 end-field variants.

    The context_tokens exclude the last field, so the model can be evaluated
    on its ability to predict what comes next.
    """
    variants = []
    for end_field, label in VARIANT_END_FIELDS:
        order = [field for field in BASE_FIELDS if field != end_field] + [end_field]
        tokens, field_boundaries = create_ordered_measurement(
            row_dict, order, prev_timestamp=None
        )

        # Truncate tokens to exclude the last field
        last_field_start, last_field_end = field_boundaries[end_field]
        context_tokens = tokens[:last_field_start]
        expected_tokens = tokens[last_field_start:last_field_end]

        variants.append(
            {
                "label": label,
                "context_tokens": context_tokens,
                "expected_tokens": expected_tokens,
                "expected_field": end_field,
            }
        )
    return variants


def build_config(
    checkpoint_path,
    config_path,
    use_gpu=False,
    max_prefill_length=None,
    max_target_length=None,
):
    """Build a MaxText config with resolved paths and eval overrides."""
    from src.MaxText import pyconfig as maxtext_pyconfig

    checkpoint_path = str(Path(checkpoint_path).resolve())
    config_path = str(Path(config_path).resolve())

    # pyconfig.initialize expects argv[0] to be program name, argv[1] to be config path,
    # and argv[2:] to be key=value overrides
    argv = [
        "eval_script",  # argv[0] - program name (ignored)
        config_path,  # argv[1] - config file path
        f"load_parameters_path={checkpoint_path}",
        f"hardware={'gpu' if use_gpu else 'cpu'}",
        "skip_jax_distributed_system=true",
    ]

    if max_prefill_length is not None:
        argv.append(f"max_prefill_predict_length={max_prefill_length}")
    if max_target_length is not None:
        argv.append(f"max_target_length={max_target_length}")

    # MaxEngine decode path requires dot_product attention.
    argv.append("attention=dot_product")

    config = maxtext_pyconfig.initialize(argv)
    return config, checkpoint_path


def setup_engine(config):
    """Create a MaxEngine and load parameters."""
    from src.MaxText import maxengine

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(0)
    params = engine.load_params(rng=rng)
    return engine, params


def pad_tokens(tokens, target_len):
    """Pad a token list to target_len with zeros."""
    seq = tokens[:target_len]
    padded = np.zeros(target_len, dtype=np.int32)
    padded[: len(seq)] = seq
    return padded, len(seq)


def extract_token(result_tokens, slot):
    """Extract the sampled token for a slot from ResultTokens."""
    token_idx = result_tokens.tokens_idx[0]
    return int(np.asarray(result_tokens.data)[slot, token_idx])


def format_pretty_tokens(tokens, max_tokens=120):
    """Format tokens as readable labels with truncation."""
    pretty = decode_token_stream_pretty(tokens)
    if len(pretty) > max_tokens:
        remaining = len(pretty) - max_tokens
        pretty = pretty[:max_tokens] + [f"...(+{remaining} more)"]
    return " ".join(pretty)


def block_format_error(block):
    kind = block.get("kind", "unknown")
    if kind in {
        "unknown",
        "ip_truncated",
        "timestamp_truncated",
        "timestamp_unknown",
        "rtt_truncated",
    }:
        return True
    if kind in ("src_ip", "dst_ip") and block.get("ip") in (None, "InvalidIP"):
        return True
    if kind == "rtt" and block.get("rtt_ms") is None:
        return True
    if kind.startswith("timestamp") and block.get("value") is None:
        return True
    return False


def format_timestamp_value(kind, value):
    if value is None:
        return None
    if kind == "timestamp_abs":
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%dT%H:%M:%SZ")
        return str(int(value))
    return f"+{int(value)}s"


def format_block(block, max_tokens=120):
    kind = block.get("kind", "unknown")
    if kind == "stop":
        return "<STOP>"
    if kind == "empty":
        return "<EMPTY>"
    if block_format_error(block):
        raw = format_pretty_tokens(block.get("tokens", []), max_tokens=max_tokens)
        return f"{kind}: {raw}" if raw else f"{kind}: <no tokens>"
    if kind == "src_ip":
        return f"src_ip(v{block['ip_version']})={block['ip']}"
    if kind == "dst_ip":
        return f"dst_ip(v{block['ip_version']})={block['ip']}"
    if kind == "rtt":
        return f"rtt={block['rtt_ms']:.3f} ms"
    if kind == "failed":
        return "rtt=FAILED"
    if kind.startswith("timestamp"):
        ts_value = format_timestamp_value(kind, block.get("value"))
        if ts_value is None:
            raw = format_pretty_tokens(block.get("tokens", []), max_tokens=max_tokens)
            return f"{kind}: {raw}" if raw else f"{kind}: <no tokens>"
        return f"{kind}={ts_value}"
    raw = format_pretty_tokens(block.get("tokens", []), max_tokens=max_tokens)
    return f"{kind}: {raw}" if raw else f"{kind}: <no tokens>"


def format_measurement_blocks(blocks, max_tokens=120):
    return [format_block(block, max_tokens=max_tokens) for block in blocks]


def decode_first_block(tokens):
    if not tokens:
        return {"kind": "empty", "tokens": []}
    if tokens[0] == MEASUREMENT_START:
        return {"kind": "stop", "tokens": [MEASUREMENT_START]}
    try:
        measurements = decode_tokens_to_measurements(tokens)
    except Exception:
        return {"kind": "unknown", "tokens": tokens}
    if measurements and measurements[0].get("blocks"):
        return measurements[0]["blocks"][0]
    return {"kind": "unknown", "tokens": tokens}


def field_from_role_token(token):
    if token in (SRC_IPV4, SRC_IPV6):
        return "src"
    if token in (DST_IPV4, DST_IPV6):
        return "dst"
    if token in (RTT_START, FAILED):
        return "rtt"
    if token in (TIMESTAMP_ABS, TIMESTAMP_DELTA1, TIMESTAMP_DELTA4):
        return "timestamp"
    return None


def field_from_block(block):
    kind = block.get("kind")
    if kind == "src_ip":
        return "src"
    if kind == "dst_ip":
        return "dst"
    if kind in ("rtt", "failed", "rtt_truncated"):
        return "rtt"
    if kind and kind.startswith("timestamp"):
        return "timestamp"
    if kind in ("ip_truncated", "timestamp_truncated", "unknown"):
        tokens = block.get("tokens", [])
        if tokens:
            return field_from_role_token(tokens[0])
    return None


def timestamp_seconds_from_tokens(tokens):
    if not tokens:
        return None
    token = tokens[0]
    try:
        if token == TIMESTAMP_ABS:
            if len(tokens) < 9:
                return None
            data = tokens[1:9]
            raw = bytes([token_to_byte(x) for x in data])
            return struct.unpack(">Q", raw)[0]
        if token == TIMESTAMP_DELTA1:
            if len(tokens) < 2:
                return None
            return token_to_byte(tokens[1])
        if token == TIMESTAMP_DELTA4:
            if len(tokens) < 5:
                return None
            data = tokens[1:5]
            raw = bytes([token_to_byte(x) for x in data])
            return struct.unpack(">I", raw)[0]
    except Exception:
        return None
    return None


def ip_bytes_from_block(block):
    tokens = block.get("tokens", [])
    if not tokens:
        return None
    role = tokens[0]
    if role in (SRC_IPV4, DST_IPV4):
        expected_len = 4
    elif role in (SRC_IPV6, DST_IPV6):
        expected_len = 16
    else:
        return None
    if len(tokens) < 1 + expected_len:
        return None
    data = tokens[1 : 1 + expected_len]
    try:
        return bytes([token_to_byte(x) for x in data])
    except Exception:
        return None


def format_value_delta(expected_block, predicted_block):
    expected_kind = expected_block.get("kind")
    predicted_kind = predicted_block.get("kind")
    if expected_kind != predicted_kind:
        return None
    if expected_kind in ("src_ip", "dst_ip"):
        expected_bytes = ip_bytes_from_block(expected_block)
        predicted_bytes = ip_bytes_from_block(predicted_block)
        if (
            expected_bytes is None
            or predicted_bytes is None
            or len(expected_bytes) != len(predicted_bytes)
        ):
            return None
        bit_errors = sum(
            (a ^ b).bit_count() for a, b in zip(expected_bytes, predicted_bytes)
        )
        bit_total = len(expected_bytes) * 8
        byte_accuracy = sum(
            a == b for a, b in zip(expected_bytes, predicted_bytes)
        ) / len(expected_bytes)
        return f"bit_err={bit_errors}/{bit_total}, byte_acc={byte_accuracy:.2%}"
    if expected_kind in ("timestamp_abs", "timestamp_delta1", "timestamp_delta4"):
        expected_seconds = timestamp_seconds_from_tokens(
            expected_block.get("tokens", [])
        )
        predicted_seconds = timestamp_seconds_from_tokens(
            predicted_block.get("tokens", [])
        )
        if expected_seconds is None or predicted_seconds is None:
            return None
        abs_err = abs(predicted_seconds - expected_seconds)
        bit_err = (predicted_seconds ^ expected_seconds).bit_count()
        return f"abs_err={abs_err}s, bit_err={bit_err}"
    if expected_kind == "rtt":
        expected_rtt = expected_block.get("rtt_ms")
        predicted_rtt = predicted_block.get("rtt_ms")
        if expected_rtt is None or predicted_rtt is None:
            return None
        abs_err = abs(predicted_rtt - expected_rtt)
        if expected_rtt > 0 and predicted_rtt > 0:
            log2_err = abs(math.log2(predicted_rtt / expected_rtt))
            return f"abs_err={abs_err:.3f}ms, log2_err={log2_err:.3f}"
        return f"abs_err={abs_err:.3f}ms"
    return None


def describe_prediction(
    expected_field, expected_block, predicted_block, max_tokens=120
):
    predicted_field = field_from_block(predicted_block)
    prediction = format_block(predicted_block, max_tokens=max_tokens)
    if predicted_block.get("kind") == "stop":
        return prediction
    if predicted_field != expected_field:
        return f"{prediction} (expected {expected_field})"
    delta = format_value_delta(expected_block, predicted_block)
    if delta:
        return f"{prediction} ({delta})"
    return prediction


def summarize_values(values):
    if not values:
        return None, None
    return float(np.mean(values)), float(np.median(values))


def update_prediction_quality(
    bucket, expected_field, expected_tokens, expected_block, predicted_block
):
    bucket["total"] += 1

    if predicted_block.get("kind") == "stop":
        bucket["stop_first"] += 1

    predicted_field = field_from_block(predicted_block)
    if predicted_field == expected_field:
        bucket["field_match"] += 1
    elif predicted_block.get("kind") not in ("stop", "empty"):
        bucket["wrong_field"] += 1

    predicted_tokens = predicted_block.get("tokens")
    if predicted_tokens is not None and predicted_tokens == expected_tokens:
        bucket["exact_tokens"] += 1

    if predicted_field == expected_field and (
        predicted_block.get("kind") == expected_block.get("kind")
    ):
        bucket["kind_match"] += 1

    if predicted_field == expected_field and block_format_error(predicted_block):
        bucket["format_error"] += 1

    if predicted_field != expected_field:
        return

    if expected_field in ("src", "dst"):
        if block_format_error(predicted_block) or block_format_error(expected_block):
            return
        expected_bytes = ip_bytes_from_block(expected_block)
        predicted_bytes = ip_bytes_from_block(predicted_block)
        if expected_bytes is None or predicted_bytes is None:
            return
        if len(expected_bytes) != len(predicted_bytes):
            bucket["ip_version_mismatch"] += 1
            return
        bit_errors = sum(
            (a ^ b).bit_count() for a, b in zip(expected_bytes, predicted_bytes)
        )
        bit_total = len(expected_bytes) * 8
        byte_accuracy = sum(
            a == b for a, b in zip(expected_bytes, predicted_bytes)
        ) / len(expected_bytes)
        bucket["ip_bit_error_rate"].append(bit_errors / bit_total)
        bucket["ip_byte_accuracy"].append(byte_accuracy)
        return

    if expected_field == "timestamp":
        if block_format_error(predicted_block) or block_format_error(expected_block):
            return
        if predicted_block.get("kind") != expected_block.get("kind"):
            return
        expected_seconds = timestamp_seconds_from_tokens(expected_tokens)
        predicted_seconds = timestamp_seconds_from_tokens(
            predicted_block.get("tokens", [])
        )
        if expected_seconds is None or predicted_seconds is None:
            return
        bucket["timestamp_abs_errors"].append(
            abs(predicted_seconds - expected_seconds)
        )
        bucket["timestamp_bit_errors"].append(
            (predicted_seconds ^ expected_seconds).bit_count()
        )
        return

    if expected_field == "rtt":
        if expected_block.get("kind") == "failed":
            bucket["expected_failed"] += 1
            if predicted_block.get("kind") == "failed":
                bucket["failed_match"] += 1
            return
        if predicted_block.get("kind") != "rtt":
            return
        expected_rtt = expected_block.get("rtt_ms")
        predicted_rtt = predicted_block.get("rtt_ms")
        if expected_rtt is None or predicted_rtt is None:
            return
        abs_err = abs(predicted_rtt - expected_rtt)
        bucket["rtt_abs_errors"].append(abs_err)
        if expected_rtt > 0 and predicted_rtt > 0:
            # Log2 ratio matches the exponent-style encoding better than linear error.
            bucket["rtt_log2_errors"].append(
                abs(math.log2(predicted_rtt / expected_rtt))
            )


def make_quality_bucket():
    return {
        "total": 0,
        "stop_first": 0,
        "field_match": 0,
        "wrong_field": 0,
        "kind_match": 0,
        "exact_tokens": 0,
        "format_error": 0,
        "ip_bit_error_rate": [],
        "ip_byte_accuracy": [],
        "ip_version_mismatch": 0,
        "timestamp_abs_errors": [],
        "timestamp_bit_errors": [],
        "rtt_abs_errors": [],
        "rtt_log2_errors": [],
        "expected_failed": 0,
        "failed_match": 0,
    }


def decode_batch_with_engine(
    engine,
    params,
    sequences,
    config,
    max_new_tokens,
    stop_token,
    rng,
    temperature=1.0,
    sampling_strategy=None,
    progress_interval=1,
    batch_label="",
):
    """Generate tokens for a batch of sequences using MaxEngine decode."""
    batch_size = len(sequences)
    completions = [[] for _ in range(batch_size)]
    generated_counts = [0] * batch_size
    stop_reasons = [None] * batch_size
    active = [True] * batch_size

    rng, rng_state = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng=rng_state)

    start_time = time.time()
    for slot, seq in enumerate(sequences):
        padded, true_length = pad_tokens(seq, config.max_prefill_predict_length)
        rng, rng_prefill = jax.random.split(rng)
        prefix, first_tokens = engine.prefill(
            params=params,
            padded_tokens=jnp.array(padded),
            true_length=true_length,
            rng=rng_prefill,
            slot=slot,
            temperature=temperature,
            algorithm=sampling_strategy,
        )
        decode_state = engine.insert(
            prefix=prefix, decode_state=decode_state, slot=slot
        )
        token = extract_token(first_tokens, 0)
        completions[slot].append(token)
        generated_counts[slot] += 1
        if token == stop_token:
            active[slot] = False
            stop_reasons[slot] = "stop_token"
            engine.release_pages(slot=slot)
        elif generated_counts[slot] >= max_new_tokens:
            active[slot] = False
            stop_reasons[slot] = "timeout"
            engine.release_pages(slot=slot)

    remaining_steps = max(0, max_new_tokens - 1)
    for step in range(remaining_steps):
        active_slots = [i for i, is_active in enumerate(active) if is_active]
        if not active_slots:
            break

        if progress_interval and step % progress_interval == 0:
            elapsed = time.time() - start_time
            label = f"{batch_label} " if batch_label else ""
            print(
                f"  {label}Step {step + 1}/{remaining_steps} | "
                f"active slots: {len(active_slots)} | "
                f"elapsed: {elapsed:.1f}s"
            )

        rng, rng_gen = jax.random.split(rng)
        decode_state, result_tokens = engine.generate(
            params=params,
            decode_state=decode_state,
            rng=rng_gen,
            temperature=temperature,
            algorithm=sampling_strategy,
        )
        tokens = np.asarray(result_tokens.data)[:, result_tokens.tokens_idx[0]]

        for slot in active_slots:
            token = int(tokens[slot])
            completions[slot].append(token)
            generated_counts[slot] += 1
            if token == stop_token:
                active[slot] = False
                stop_reasons[slot] = "stop_token"
                engine.release_pages(slot=slot)
            elif generated_counts[slot] >= max_new_tokens:
                active[slot] = False
                stop_reasons[slot] = "timeout"
                engine.release_pages(slot=slot)

    for slot, is_active in enumerate(active):
        if is_active:
            stop_reasons[slot] = "timeout"
            engine.release_pages(slot=slot)

    return generated_counts, stop_reasons, completions, rng


def decode_streams_with_engine(
    engine,
    params,
    sequences,
    config,
    max_new_tokens,
    stop_token,
    rng,
    temperature=1.0,
    sampling_strategy=None,
    progress_interval=1,
):
    """Decode sequences in chunks that fit the engine's max concurrent decodes."""
    max_slots = engine.max_concurrent_decodes
    all_counts = [0] * len(sequences)
    all_reasons = [None] * len(sequences)
    all_completions = [[] for _ in range(len(sequences))]

    for batch_idx, start in enumerate(range(0, len(sequences), max_slots), start=1):
        end = min(start + max_slots, len(sequences))
        batch_sequences = sequences[start:end]
        print(f"\nDecoding batch {batch_idx} ({len(batch_sequences)} streams)...")
        counts, reasons, completions, rng = decode_batch_with_engine(
            engine,
            params,
            batch_sequences,
            config,
            max_new_tokens,
            stop_token,
            rng,
            temperature=temperature,
            sampling_strategy=sampling_strategy,
            progress_interval=progress_interval,
            batch_label=f"batch {batch_idx}",
        )
        for offset, (count, reason, completion) in enumerate(
            zip(counts, reasons, completions)
        ):
            idx = start + offset
            all_counts[idx] = count
            all_reasons[idx] = reason
            all_completions[idx] = completion

    return all_counts, all_reasons, all_completions


def run_eval(
    checkpoint,
    config,
    data,
    num_samples=100,
    seed=42,
    max_new_tokens=20,
    temperature=1.0,
    sampling_strategy="weighted",
    progress_interval=1,
    print_per_variant=2,
    print_max_tokens=120,
):
    """
    Core evaluation logic (can be called from CLI or Modal).

    Args:
        checkpoint: Path to checkpoint directory
        config: Path to config file
        data: Path to training data parquet file
        num_samples: Number of measurements to evaluate
        seed: Random seed
    """
    from src.MaxText import max_utils

    print(f"\nLoading data from {data}...")
    columns = ["src_addr", "dst_addr", "ip_version", "rtt", "event_time"]
    df = pd.read_parquet(data, columns=columns)
    print(f"✓ Loaded {len(df)} measurements")

    samples = df.sample(n=min(num_samples, len(df)), random_state=seed)
    streams = []
    stream_meta = []
    initial_streams = []

    print("\nPreparing measurement variants...")
    for idx, row in enumerate(samples.itertuples(index=False)):
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{len(samples)}...")
        row_dict = row._asdict()
        for variant in build_measurement_variants(row_dict):
            context_tokens = variant["context_tokens"]
            streams.append(context_tokens)
            initial_streams.append(context_tokens.copy())
            stream_meta.append(
                {
                    "variant": variant["label"],
                    "sample_index": idx,
                    "expected_field": variant["expected_field"],
                    "expected_tokens": variant["expected_tokens"],
                }
            )

    max_prompt_len = max(len(tokens) for tokens in streams)
    max_target_len = max_prompt_len + max_new_tokens

    print("\nLoading MaxEngine and checkpoint...")
    use_gpu = IN_MODAL_RUNTIME or os.environ.get("JAX_PLATFORMS") == "gpu"
    config_obj, checkpoint_path = build_config(
        checkpoint,
        config,
        use_gpu=use_gpu,
        max_prefill_length=max_prompt_len,
        max_target_length=max_target_len,
    )
    with max_utils.maybe_get_transformer_engine_context(config_obj):
        engine, params = setup_engine(config_obj)
        print(f"✓ Engine initialized with params from {checkpoint_path}")

        print(f"\nGenerating up to {max_new_tokens} tokens per stream...")
        rng = jax.random.PRNGKey(seed)
        generated_counts, stop_reasons, completions = decode_streams_with_engine(
            engine,
            params,
            streams,
            config_obj,
            max_new_tokens=max_new_tokens,
            stop_token=MEASUREMENT_START,
            rng=rng,
            temperature=temperature,
            sampling_strategy=sampling_strategy,
            progress_interval=progress_interval,
        )

    stats = defaultdict(
        lambda: {"counts": [], "stop_token": 0, "timeout": 0, "max_len": 0}
    )
    quality = defaultdict(make_quality_bucket)
    expected_blocks = []
    predicted_blocks = []
    for meta, count, reason, completion in zip(
        stream_meta, generated_counts, stop_reasons, completions
    ):
        variant = meta["variant"]
        stats[variant]["counts"].append(count)
        stats[variant][reason] += 1
        expected_block = decode_first_block(meta["expected_tokens"])
        predicted_block = decode_first_block(completion)
        expected_blocks.append(expected_block)
        predicted_blocks.append(predicted_block)
        update_prediction_quality(
            quality[variant],
            meta["expected_field"],
            meta["expected_tokens"],
            expected_block,
            predicted_block,
        )

    print("\n" + "=" * 80)
    print("RESULTS: Batched Continuations by End Field")
    print("=" * 80)
    for _, label in VARIANT_END_FIELDS:
        counts = stats[label]["counts"]
        if not counts:
            print(f"{label:14s}  No samples")
            continue
        total = len(counts)
        stop_rate = stats[label]["stop_token"] / total
        timeout_rate = stats[label]["timeout"] / total
        max_len_rate = stats[label]["max_len"] / total
        print(
            f"{label:14s}  "
            f"Mean tokens: {np.mean(counts):6.2f}  "
            f"Stop: {stop_rate:6.2%}  "
            f"Timeout: {timeout_rate:6.2%}  "
            f"MaxLen: {max_len_rate:6.2%}  "
            f"(n={total})"
        )

    print("\nNotes:")
    print("  - 'Stop' means MEASUREMENT_START was generated.")
    print("  - 'Timeout' means max-new-tokens was reached without a stop token.")
    print("  - 'MaxLen' means the sequence hit the model's max_target_length.")

    print("\n" + "=" * 80)
    print("PREDICTION QUALITY: Expected Field vs First Completion Block")
    print("=" * 80)
    for end_field, label in VARIANT_END_FIELDS:
        bucket = quality[label]
        total = bucket["total"]
        if total == 0:
            print(f"{label:14s}  No samples")
            continue
        field_match_rate = bucket["field_match"] / total
        exact_rate = bucket["exact_tokens"] / total
        stop_first_rate = bucket["stop_first"] / total
        wrong_field_rate = bucket["wrong_field"] / total
        format_error_rate = bucket["format_error"] / total
        print(
            f"{label:14s}  "
            f"FieldMatch: {field_match_rate:6.2%}  "
            f"Exact: {exact_rate:6.2%}  "
            f"StopFirst: {stop_first_rate:6.2%}  "
            f"FormatErr: {format_error_rate:6.2%}  "
            f"WrongField: {wrong_field_rate:6.2%}  "
            f"(n={total})"
        )

        if end_field in ("src", "dst"):
            bit_mean, bit_median = summarize_values(bucket["ip_bit_error_rate"])
            byte_mean, byte_median = summarize_values(bucket["ip_byte_accuracy"])
            if bit_mean is None:
                print("  ip_bit_error_rate: n/a | ip_byte_accuracy: n/a")
            else:
                print(
                    f"  ip_bit_error_rate: mean {bit_mean:.2%}, median {bit_median:.2%} "
                    f"(n={len(bucket['ip_bit_error_rate'])})"
                )
                print(
                    f"  ip_byte_accuracy: mean {byte_mean:.2%}, median {byte_median:.2%}"
                )
            if bucket["ip_version_mismatch"]:
                print(
                    f"  ip_version_mismatch: {bucket['ip_version_mismatch']}"
                )

        if end_field == "timestamp":
            abs_mean, abs_median = summarize_values(bucket["timestamp_abs_errors"])
            bit_mean, bit_median = summarize_values(bucket["timestamp_bit_errors"])
            if abs_mean is None:
                print("  abs_err_s: n/a | bit_diff: n/a")
            else:
                print(
                    f"  abs_err_s: mean {abs_mean:.1f}, median {abs_median:.1f} "
                    f"(n={len(bucket['timestamp_abs_errors'])})"
                )
                print(
                    f"  bit_diff: mean {bit_mean:.1f}, median {bit_median:.1f}"
                )

        if end_field == "rtt":
            abs_mean, abs_median = summarize_values(bucket["rtt_abs_errors"])
            log_mean, log_median = summarize_values(bucket["rtt_log2_errors"])
            if abs_mean is None:
                print("  rtt_abs_err_ms: n/a | rtt_log2_err: n/a")
            else:
                print(
                    f"  rtt_abs_err_ms: mean {abs_mean:.3f}, median {abs_median:.3f} "
                    f"(n={len(bucket['rtt_abs_errors'])})"
                )
                print(
                    f"  rtt_log2_err: mean {log_mean:.3f}, median {log_median:.3f}"
                )
            if bucket["expected_failed"]:
                failed_rate = bucket["failed_match"] / bucket["expected_failed"]
                print(
                    f"  failed_match: {failed_rate:6.2%} (n={bucket['expected_failed']})"
                )

    if print_per_variant > 0:
        print("\n" + "=" * 80)
        print("SAMPLES: Contexts and Predicted Fields")
        print("=" * 80)
        shown = defaultdict(int)
        for idx, meta in enumerate(stream_meta):
            variant = meta["variant"]
            if shown[variant] >= print_per_variant:
                continue
            shown[variant] += 1
            context = initial_streams[idx]
            expected_block = expected_blocks[idx]
            predicted_block = predicted_blocks[idx]
            print(
                f"{variant} sample {shown[variant]} "
                f"(generated={generated_counts[idx]}, stop={stop_reasons[idx]})"
            )
            try:
                context_measurements = decode_tokens_to_measurements(context)
            except Exception:
                context_measurements = []
            if context_measurements and context_measurements[0].get("blocks"):
                print("  context:")
                for block_str in format_measurement_blocks(
                    context_measurements[0]["blocks"], max_tokens=print_max_tokens
                ):
                    print(f"    {block_str}")
            else:
                print(
                    f"  context: {format_pretty_tokens(context, max_tokens=print_max_tokens)}"
                )
            print(
                f"  expected:  {format_block(expected_block, max_tokens=print_max_tokens)}"
            )
            print(
                f"  predicted: {describe_prediction(meta['expected_field'], expected_block, predicted_block, max_tokens=print_max_tokens)}"
            )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate model continuations by end field"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to param-only checkpoint items directory",
    )
    parser.add_argument(
        "--config",
        default="src/MaxText/configs/latency_network.yml",
        help="Config file path",
    )
    parser.add_argument(
        "--data",
        default="data/training_data.parquet",
        help="Path to training data parquet file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of measurements to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate per stream",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="weighted",
        help="Sampling strategy: greedy, weighted, nucleus, topk, or composite",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Print progress every N generation steps (0 to disable)",
    )
    parser.add_argument(
        "--print-per-variant",
        type=int,
        default=2,
        help="How many samples to print per variant (0 to disable)",
    )
    parser.add_argument(
        "--print-max-tokens",
        type=int,
        default=120,
        help="Maximum tokens to display per context/completion",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        config=args.config,
        data=args.data,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        sampling_strategy=args.sampling_strategy,
        progress_interval=args.progress_interval,
        print_per_variant=args.print_per_variant,
        print_max_tokens=args.print_max_tokens,
        seed=args.seed,
    )


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
        timeout=60 * 60 * 2,  # 2 hours
        env={
            # Suppress TensorFlow/CUDA plugin registration warnings
            "TF_CPP_MIN_LOG_LEVEL": "2",
            # Suppress redundant XLA warnings
            "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=1",
        },
    )
    def eval_on_modal(
        checkpoint_path: str = PARAM_ONLY_CHECKPOINT_MODAL,
        config: str = "src/MaxText/configs/latency_network.yml",
        data_file: str = "training_data.parquet",
        num_samples: int = 100,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        sampling_strategy: str = "weighted",
        progress_interval: int = 1,
        print_per_variant: int = 2,
        print_max_tokens: int = 120,
        seed: int = 42,
    ):
        """
        Run eval_ordering_likelihood on Modal with GPU acceleration.

        Args:
            checkpoint_path: Param-only checkpoint items path
            config: Config file path (relative to workspace)
            data_file: Data file name within /mnt/data (e.g., "training_data.parquet")
            num_samples: Number of measurements to evaluate
            max_new_tokens: Maximum tokens to generate per stream
            temperature: Sampling temperature
            progress_interval: Print progress every N generation steps (0 to disable)
            print_per_variant: How many samples to print per variant (0 to disable)
            print_max_tokens: Maximum tokens to display per context/completion
            seed: Random seed
        """
        import sys

        # Set up symlinks for config paths
        os.makedirs(f"{WORKDIR}/outputs", exist_ok=True)
        os.makedirs(f"{WORKDIR}/data", exist_ok=True)

        # Link outputs and data from volume
        if not os.path.exists(f"{WORKDIR}/outputs/latency_network"):
            os.symlink(
                "/mnt/outputs/latency_network", f"{WORKDIR}/outputs/latency_network"
            )
        if not os.path.exists(f"{WORKDIR}/data/{data_file}"):
            os.symlink(f"/mnt/data/{data_file}", f"{WORKDIR}/data/{data_file}")

        # Resolve checkpoint path
        if not checkpoint_path.startswith("/"):
            checkpoint_path = f"{WORKDIR}/{checkpoint_path}"
        data_path = f"{WORKDIR}/data/{data_file}"
        config_path = f"{WORKDIR}/{config}"

        # Prepare argv for main()
        sys.argv = [
            "eval_ordering_likelihood.py",
            "--checkpoint",
            checkpoint_path,
            "--config",
            config_path,
            "--data",
            data_path,
            "--num-samples",
            str(num_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            str(temperature),
            "--sampling-strategy",
            str(sampling_strategy),
            "--progress-interval",
            str(progress_interval),
            "--print-per-variant",
            str(print_per_variant),
            "--print-max-tokens",
            str(print_max_tokens),
            "--seed",
            str(seed),
        ]

        # Set environment to indicate GPU usage
        os.environ["JAX_PLATFORMS"] = "gpu"

        # Change to workspace directory
        os.chdir(WORKDIR)

        # Run main function
        main()
