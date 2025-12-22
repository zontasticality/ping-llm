#!/usr/bin/env python3
"""
Paper metrics collection:
- Writes timestamp, mode, and ping metrics JSON files
- Use eval_paper_metrics_plot.py to render figures
"""

import os
from pathlib import Path


def _in_modal_runtime():
    return (
        bool(os.environ.get("MODAL_IS_REMOTE")) or (Path("/workspace") / "src").exists()
    )


IN_MODAL_RUNTIME = _in_modal_runtime()
if not IN_MODAL_RUNTIME:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import threading
import time
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import entropy
import duckdb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add project root to path
workspace_root = Path("/workspace")
if IN_MODAL_RUNTIME and (workspace_root / "src").exists():
    repo_root = workspace_root
else:
    repo_root = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(repo_root))

plt.style.use("seaborn-v0_8")

try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:
    APP_NAME = "ping-llm-eval-paper-metrics"
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

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "cmake", "ninja-build", "iputils-ping")
        .pip_install("uv")
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
        .run_commands(
            f"cd {WORKDIR} && CC=gcc CXX=g++ uv pip install --system -e '.[cuda12]' --resolution=lowest",
            f"cd {WORKDIR} && install_maxtext_github_deps",
        )
        .uv_pip_install("google-jetstream")
        .uv_pip_install("pandas", "pyarrow", "duckdb", "scipy", "matplotlib")
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    )

    app = modal.App(APP_NAME)
    shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

from src.MaxText import max_utils
try:
    from src.MaxText.input_pipeline._probe_chunk_datasource import ProbeRowDataSource
except ImportError:
    ProbeRowDataSource = None
from src.MaxText.input_pipeline.network_tokenization import (
    DST_IPV4,
    DST_IPV6,
    FAILED,
    MEASUREMENT_START,
    RTT_START,
    SRC_IPV4,
    SRC_IPV6,
    TIMESTAMP_ABS,
    TIMESTAMP_DELTA1,
    TIMESTAMP_DELTA4,
    decode_rtt_exponent_mantissa,
    decode_token_stream_pretty,
    encode_ip_merged,
    encode_measurement,
    encode_rtt_exponent_mantissa,
    encode_timestamp_delta,
    token_to_byte,
)

DEFAULT_REGULAR_IPS = [
    "8.8.8.8",
    "8.8.4.4",
    "1.1.1.1",
    "1.0.0.1",
    "9.9.9.9",
    "149.112.112.112",
    "208.67.222.222",
    "208.67.220.220",
    "64.6.64.6",
    "64.6.65.6",
]

DEFAULT_ANCHOR_IPS = [
    "198.41.0.4",
    "199.9.14.201",
    "192.33.4.12",
    "199.7.91.13",
    "192.203.230.10",
    "192.5.5.241",
    "192.112.36.4",
    "198.97.190.53",
    "192.36.148.17",
    "192.58.128.30",
]

OUTPUT_BASE_MODAL = "/mnt/outputs/latency_network"
OUTPUT_BASE_LOCAL = "outputs/latency_network"
OUTPUT_BASE = OUTPUT_BASE_MODAL if IN_MODAL_RUNTIME else OUTPUT_BASE_LOCAL
DEFAULT_OUTPUT_DIR_MODAL = "/mnt/outputs/paper_metrics/default"
DEFAULT_OUTPUT_DIR_LOCAL = "outputs/paper_metrics/default"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUT_DIR_MODAL if IN_MODAL_RUNTIME else DEFAULT_OUTPUT_DIR_LOCAL
)

PARAM_ONLY_CHECKPOINT_MODAL = (
    f"{OUTPUT_BASE_MODAL}/param_only_checkpoint/checkpoints/0/items"
)
PARAM_ONLY_CHECKPOINT_LOCAL = (
    f"{OUTPUT_BASE_LOCAL}/param_only_checkpoint/checkpoints/0/items"
)
DEFAULT_CHECKPOINT = (
    PARAM_ONLY_CHECKPOINT_MODAL if IN_MODAL_RUNTIME else PARAM_ONLY_CHECKPOINT_LOCAL
)
DEFAULT_ARRAYRECORD = (
    "/mnt/data/probe_rows/test.arrayrecord"
    if IN_MODAL_RUNTIME
    else "data/probe_rows/test.arrayrecord"
)
DEFAULT_PARQUET = (
    "/mnt/data/training_data.parquet"
    if IN_MODAL_RUNTIME
    else "data/training_data.parquet"
)


def find_latest_checkpoint_items(run_name, output_base=OUTPUT_BASE):
    candidates = [
        os.path.join(output_base, run_name, "checkpoints"),
        os.path.join(output_base, run_name, run_name, "checkpoints"),
    ]
    for root in candidates:
        if not os.path.isdir(root):
            continue
        numeric = []
        for entry in os.listdir(root):
            if entry.isdigit():
                step_dir = os.path.join(root, entry)
                items_dir = os.path.join(step_dir, "items")
                if os.path.isdir(items_dir):
                    numeric.append(int(entry))
        if numeric:
            latest = str(max(numeric))
            return os.path.join(root, latest, "items")
    raise FileNotFoundError(
        f"No checkpoints with items/ found for run_name='{run_name}' in: {', '.join(candidates)}"
    )

BASE_FIELDS = ["src", "dst", "rtt", "timestamp"]
VARIANT_END_FIELDS = [
    ("timestamp", "end_timestamp"),
    ("src", "end_src_ip"),
    ("dst", "end_dst_ip"),
    ("rtt", "end_latency"),
]


def build_config(
    checkpoint_path,
    config_path,
    max_prefill_length,
    max_target_length,
    use_gpu=False,
):
    from src.MaxText import pyconfig as maxtext_pyconfig
    checkpoint_path = str(Path(checkpoint_path).resolve())
    config_path = str(resolve_repo_path(config_path))

    argv = [
        "eval_script",
        config_path,
        f"load_parameters_path={checkpoint_path}",
        f"hardware={'gpu' if use_gpu else 'cpu'}",
        "skip_jax_distributed_system=true",
        f"max_prefill_predict_length={max_prefill_length}",
        f"max_target_length={max_target_length}",
        "attention=dot_product",
    ]

    config = maxtext_pyconfig.initialize(argv)
    return config, checkpoint_path


def resolve_repo_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    candidate = workspace_root / path
    if candidate.exists():
        return candidate
    return path


def setup_engine(config):
    from src.MaxText import maxengine

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(0)
    params = engine.load_params(rng=rng)
    return engine, params


def pad_tokens(tokens, target_len):
    seq = tokens[:target_len]
    padded = np.zeros(target_len, dtype=np.int32)
    padded[: len(seq)] = seq
    return padded, len(seq)


def extract_token(result_tokens, slot):
    token_idx = result_tokens.tokens_idx[0]
    return int(np.asarray(result_tokens.data)[slot, token_idx])


def compute_prompt_logprobs(engine, params, config, tokens, rng):
    padded, true_length = pad_tokens(tokens, config.max_prefill_predict_length)
    rng, rng_prefill = jax.random.split(rng)
    prefix, _ = engine.prefill(
        params=params,
        padded_tokens=jnp.array(padded),
        true_length=true_length,
        rng=rng_prefill,
        return_prompt_logp=True,
    )
    prompt_logp = np.array(prefix["prompt_logp"])[0]
    return prompt_logp[:true_length], rng


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


def sample_measurement_window(measurements, target, rng):
    n = len(measurements)
    if n <= target:
        selected = list(measurements)
    else:
        log_size = rng.uniform(0, np.log(n))
        window_size = max(1, int(np.exp(log_size)))
        window_size = min(n, window_size)
        if window_size >= n:
            offset = 0
        else:
            offset = int(rng.integers(0, n - window_size + 1))
        window = measurements[offset : offset + window_size]
        if len(window) <= target:
            selected = list(window)
        else:
            indices = rng.choice(len(window), size=target, replace=False)
            selected = [window[i] for i in indices]
    selected.sort(key=lambda m: m["event_time"])
    return selected


def tokenize_measurements(measurements, mode, rng):
    seeds = rng.integers(0, 2**31, size=len(measurements), dtype=np.int64)
    tokens = []
    if mode == "full":
        prev_timestamp = None
        for meas, seed in zip(measurements, seeds):
            tokens.extend(
                encode_measurement(
                    meas,
                    prev_timestamp=prev_timestamp,
                    include_timestamp=True,
                    shuffle_seed=int(seed),
                )
            )
            prev_timestamp = meas["event_time"]
        return tokens
    if mode == "none":
        pairs = list(zip(measurements, seeds))
        rng.shuffle(pairs)
        for meas, seed in pairs:
            tokens.extend(
                encode_measurement(
                    meas,
                    prev_timestamp=None,
                    include_timestamp=False,
                    shuffle_seed=int(seed),
                )
            )
        return tokens
    raise ValueError(f"Unsupported mode: {mode}")


def build_rtt_prediction_tokens(row, include_timestamp):
    tokens = [MEASUREMENT_START]
    tokens.extend(encode_ip_merged(row["src_addr"], row["ip_version"], is_src=True))
    tokens.extend(encode_ip_merged(row["dst_addr"], row["ip_version"], is_src=False))
    if include_timestamp:
        tokens.extend(encode_timestamp_delta(row["event_time"], prev_time=None))

    if row["rtt"] < 0:
        expected = [FAILED]
    else:
        expected = encode_rtt_exponent_mantissa(row["rtt"])
    return tokens, expected


def sample_rtt_pairs_from_parquet(
    parquet_path,
    num_samples,
    rng,
    sample_rows=50000,
):
    query = (
        "SELECT src_addr, dst_addr, ip_version, rtt, event_time "
        "FROM parquet_scan(?) LIMIT ?"
    )
    df = duckdb.connect().execute(query, [str(parquet_path), sample_rows]).fetch_df()
    if df.empty:
        return []
    df["event_time"] = pd.to_datetime(df["event_time"])
    sample_count = min(num_samples, len(df))
    seed = int(rng.integers(0, 2**31))
    samples = df.sample(n=sample_count, random_state=seed)
    pairs = []
    for row in samples.itertuples(index=False):
        row_dict = row._asdict()
        context_full, expected_full = build_rtt_prediction_tokens(
            row_dict, include_timestamp=True
        )
        context_none, expected_none = build_rtt_prediction_tokens(
            row_dict, include_timestamp=False
        )
        pairs.append((context_full, expected_full, context_none, expected_none))
    return pairs


def create_ordered_measurement(row, field_order, prev_timestamp=None):
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
    variants = []
    for end_field, label in VARIANT_END_FIELDS:
        order = [field for field in BASE_FIELDS if field != end_field] + [end_field]
        tokens, field_boundaries = create_ordered_measurement(
            row_dict, order, prev_timestamp=None
        )
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


def group_key_for_expected(expected_tokens):
    if not expected_tokens:
        return "empty"
    first = expected_tokens[0]
    if first == SRC_IPV4:
        return "src_ipv4"
    if first == SRC_IPV6:
        return "src_ipv6"
    if first == DST_IPV4:
        return "dst_ipv4"
    if first == DST_IPV6:
        return "dst_ipv6"
    if first == RTT_START:
        return "rtt"
    if first == FAILED:
        return "rtt_failed"
    if first == TIMESTAMP_ABS:
        return "timestamp_abs"
    if first == TIMESTAMP_DELTA1:
        return "timestamp_delta1"
    if first == TIMESTAMP_DELTA4:
        return "timestamp_delta4"
    return "unknown"


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


def _format_ping_prefix(label, ping_index, ping_count):
    if label is None:
        label = "unknown"
    if ping_index is None or ping_count is None:
        return f"[ping {label}]"
    return f"[ping {label} {ping_index + 1}/{ping_count}]"


def _log_ping_output(message, log_lock=None):
    if log_lock is None:
        print(message, flush=True)
        return
    with log_lock:
        print(message, flush=True)


def ping_host(
    ip,
    timeout=2,
    label=None,
    ping_index=None,
    ping_count=None,
    show_output=False,
    log_lock=None,
):
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout), ip],
            capture_output=True,
            text=True,
            timeout=timeout + 1,
        )
        prefix = _format_ping_prefix(label, ping_index, ping_count)
        if result.returncode != 0:
            if show_output:
                _log_ping_output(
                    f"{prefix} failed (code={result.returncode})",
                    log_lock=log_lock,
                )
                if result.stdout.strip():
                    _log_ping_output(result.stdout.strip(), log_lock=log_lock)
                if result.stderr.strip():
                    _log_ping_output(result.stderr.strip(), log_lock=log_lock)
            return -1.0
        match = re.search(r"time[=\s]+(\d+\.?\d*)\s*ms", result.stdout)
        if match:
            rtt = float(match.group(1))
            if show_output:
                _log_ping_output(f"{prefix} time={rtt:.3f} ms", log_lock=log_lock)
            return rtt
        if show_output:
            _log_ping_output(f"{prefix} no RTT parsed", log_lock=log_lock)
            if result.stdout.strip():
                _log_ping_output(result.stdout.strip(), log_lock=log_lock)
            if result.stderr.strip():
                _log_ping_output(result.stderr.strip(), log_lock=log_lock)
        return -1.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        if show_output:
            prefix = _format_ping_prefix(label, ping_index, ping_count)
            _log_ping_output(f"{prefix} failed (exception)", log_lock=log_lock)
        return -1.0


def ping_series(label, ip, count, timeout, show_output, log_lock=None):
    rtts = []
    for idx in range(count):
        rtt = ping_host(
            ip,
            timeout=timeout,
            label=label,
            ping_index=idx,
            ping_count=count,
            show_output=show_output,
            log_lock=log_lock,
        )
        rtts.append(rtt)
    return rtts


def parse_ip_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_ping_targets(regular_ips, anchor_ips):
    targets = []
    seen = set()
    for idx, ip in enumerate(regular_ips, start=1):
        if ip in seen:
            continue
        seen.add(ip)
        targets.append(
            {
                "ip": ip,
                "label": f"regular {idx} ({ip})",
                "group": "regular",
            }
        )
    for idx, ip in enumerate(anchor_ips, start=1):
        if ip in seen:
            continue
        seen.add(ip)
        targets.append(
            {
                "ip": ip,
                "label": f"anchor {idx} ({ip})",
                "group": "anchor",
            }
        )
    return targets


def get_src_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        src_ip = sock.getsockname()[0]
        sock.close()
        return src_ip
    except Exception:
        return "127.0.0.1"


def create_conditioning_tokens(src_ip, dst_ip, include_timestamp=False):
    tokens = [MEASUREMENT_START]
    tokens.extend(encode_ip_merged(src_ip, 4, is_src=True))
    tokens.extend(encode_ip_merged(dst_ip, 4, is_src=False))
    if include_timestamp:
        current_time = datetime.now()
        tokens.extend(encode_timestamp_delta(current_time, prev_time=None))
    return tokens


def sample_rtt_from_model(
    engine,
    params,
    config,
    conditioning_tokens,
    num_samples=100,
    temperature=1.0,
    sampling_strategy="weighted",
    rng=None,
):
    rtt_samples = []
    timeout_count = 0
    invalid_count = 0
    if rng is None:
        rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**31)

    max_slots = engine.max_concurrent_decodes
    padded_tokens, true_length = pad_tokens(
        conditioning_tokens, config.max_prefill_predict_length
    )
    padded_tokens = jnp.array(padded_tokens)

    for start in range(0, num_samples, max_slots):
        batch_size = min(max_slots, num_samples - start)
        rng, rng_state = jax.random.split(rng)
        decode_state = engine.init_decode_state(rng=rng_state)

        first_tokens = []
        for slot in range(batch_size):
            rng, rng_prefill = jax.random.split(rng)
            prefix, first = engine.prefill(
                params=params,
                padded_tokens=padded_tokens,
                true_length=true_length,
                rng=rng_prefill,
                slot=slot,
                temperature=temperature,
                algorithm=sampling_strategy,
            )
            decode_state = engine.insert(
                prefix=prefix, decode_state=decode_state, slot=slot
            )
            first_tokens.append(extract_token(first, slot=0))

        rng, rng_gen1 = jax.random.split(rng)
        decode_state, result_tokens = engine.generate(
            params=params,
            decode_state=decode_state,
            rng=rng_gen1,
            temperature=temperature,
            algorithm=sampling_strategy,
        )
        second_tokens = np.asarray(result_tokens.data)[:, result_tokens.tokens_idx[0]]

        rng, rng_gen2 = jax.random.split(rng)
        decode_state, result_tokens = engine.generate(
            params=params,
            decode_state=decode_state,
            rng=rng_gen2,
            temperature=temperature,
            algorithm=sampling_strategy,
        )
        third_tokens = np.asarray(result_tokens.data)[:, result_tokens.tokens_idx[0]]

        for slot in range(batch_size):
            first_token = int(first_tokens[slot])
            second_token = int(second_tokens[slot])
            third_token = int(third_tokens[slot])
            try:
                if first_token == FAILED:
                    timeout_count += 1
                    continue
                if first_token != RTT_START:
                    invalid_count += 1
                    continue
                if second_token == FAILED or third_token == FAILED:
                    invalid_count += 1
                    continue
                byte1 = token_to_byte(second_token)
                byte2 = token_to_byte(third_token)
                rtt_ms = decode_rtt_exponent_mantissa(byte1, byte2)
                rtt_samples.append(rtt_ms)
            except Exception:
                invalid_count += 1
                continue
            finally:
                if (
                    getattr(engine, "page_manager", None) is not None
                    and getattr(engine.config, "attention", None) == "paged"
                ):
                    engine.release_pages(slot=slot)

    return rtt_samples, timeout_count, invalid_count


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
    return float(entropy(p, q))


def plot_ping_hist(
    output_path,
    title_label,
    real_rtts,
    model_rtts,
    bins,
    range_min,
    range_max,
    kl_value,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    real_vals = [r for r in real_rtts if r >= 0]
    model_vals = [r for r in model_rtts if r >= 0]
    if real_vals:
        weights = np.ones(len(real_vals)) / len(real_vals)
        ax.hist(
            real_vals,
            bins=bins,
            range=(range_min, range_max),
            weights=weights,
            alpha=0.6,
            label="real",
        )
    if model_vals:
        weights = np.ones(len(model_vals)) / len(model_vals)
        ax.hist(
            model_vals,
            bins=bins,
            range=(range_min, range_max),
            weights=weights,
            alpha=0.6,
            label="model",
        )
    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("probability mass")
    ax.set_title(f"{title_label} | KL={kl_value:.4f}")
    ax.set_xlim(range_min, range_max)
    if real_vals or model_vals:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ping_grid(
    output_path,
    results,
    bins,
    avg_kl=None,
    src_ip=None,
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
        if row_idx == grid_rows - 1:
            ax.set_xlabel("RTT (ms)")
            ax.set_xticks([range_min, range_max, timeout_center])
            ax.set_xticklabels(
                [f"{range_min:.0f}", f"{range_max:.0f}", "Timeout"]
            )
        else:
            ax.set_xticklabels([])
        if col_idx == 0:
            ax.set_ylabel("probability mass")
        else:
            ax.set_yticklabels([])

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
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_only_arg(value):
    items = {item.strip().lower() for item in value.split(",") if item.strip()}
    if "all" in items or not items:
        return {"timestamps", "modes", "ping"}
    return items


def build_parser():
    parser = argparse.ArgumentParser(description="Paper metrics evaluation")
    parser.add_argument(
        "--param-only-run-name",
        default="param_only_checkpoint",
        help="Run name for param-only checkpoints",
    )
    parser.add_argument(
        "--config",
        default="src/MaxText/configs/latency_network.yml",
        help="Config file path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: outputs/paper_metrics/default)",
    )
    parser.add_argument("--run-name", default=None, help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--only",
        default="all",
        help="Comma list: timestamps,modes,ping,all",
    )

    parser.add_argument(
        "--timestamp-parquet",
        default=DEFAULT_PARQUET,
        help="Parquet path for timestamp evaluation",
    )
    parser.add_argument(
        "--timestamp-contexts",
        type=int,
        default=200,
        help="Number of contexts for timestamp histogram",
    )
    parser.add_argument(
        "--timestamp-sample-rows",
        type=int,
        default=50000,
        help="Rows to sample from parquet when ArrayRecord is unavailable",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max tokens per sample (unused for timestamp RTT mode)",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Histogram bins for logprob plots",
    )

    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET,
        help="Parquet file for prediction-mode evaluation",
    )
    parser.add_argument(
        "--mode-samples",
        type=int,
        default=200,
        help="Number of measurements for prediction-mode evaluation",
    )

    parser.add_argument(
        "--regular-ips",
        default=",".join(DEFAULT_REGULAR_IPS),
        help="Comma-separated list of regular IPs to ping",
    )
    parser.add_argument(
        "--anchor-ips",
        default=",".join(DEFAULT_ANCHOR_IPS),
        help="Comma-separated list of anchor IPs to ping",
    )
    parser.add_argument(
        "--pings-per-ip",
        type=int,
        default=20,
        help="Number of pings per IP",
    )
    parser.add_argument(
        "--model-samples",
        type=int,
        default=1000,
        help="Number of model samples per IP",
    )
    parser.add_argument(
        "--ping-bins",
        type=int,
        default=40,
        help="Histogram bins for ping plots",
    )
    parser.add_argument(
        "--ping-workers",
        type=int,
        default=1,
        help="Parallel workers for pinging targets",
    )
    parser.add_argument(
        "--max-rtt",
        type=int,
        default=1000,
        help="Max RTT (ms) for ping histograms",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for ping evaluation",
    )
    parser.add_argument(
        "--ping-sampling-strategy",
        default="weighted",
        help="Sampling strategy for ping evaluation (e.g., weighted, greedy, nucleus)",
    )
    parser.add_argument(
        "--ping-no-timestamp",
        action="store_true",
        help="Disable timestamp conditioning for ping evaluation",
    )

    return parser


def run(args):
    selected = parse_only_arg(args.only)
    rng = np.random.default_rng(args.seed)
    jax_rng = jax.random.PRNGKey(args.seed)

    run_id = args.run_name or "default"
    output_dir = Path(args.output_dir) if args.output_dir else Path(DEFAULT_OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint = find_latest_checkpoint_items(args.param_only_run_name)
    checkpoint_source = f"param_only:{args.param_only_run_name}"

    metrics = {
        "run": {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "checkpoint": str(resolved_checkpoint),
            "checkpoint_source": checkpoint_source,
            "config": str(args.config),
        }
    }

    mode_variants = []
    max_prefill_len = 1

    rtt_pairs = []
    if "timestamps" in selected:
        parquet_path = Path(args.timestamp_parquet)
        if not parquet_path.exists():
            print(f"Parquet not found: {parquet_path}. Skipping timestamps.")
            selected.discard("timestamps")
        else:
            rtt_pairs = sample_rtt_pairs_from_parquet(
                parquet_path,
                args.timestamp_contexts,
                rng,
                sample_rows=args.timestamp_sample_rows,
            )
            for context_full, expected_full, context_none, expected_none in rtt_pairs:
                max_prefill_len = max(
                    max_prefill_len,
                    len(context_full) + len(expected_full),
                    len(context_none) + len(expected_none),
                )

    if "modes" in selected:
        parquet_path = Path(args.parquet)
        if not parquet_path.exists():
            print(f"Parquet not found: {parquet_path}. Skipping modes.")
            selected.discard("modes")
        else:
            columns = ["src_addr", "dst_addr", "ip_version", "rtt", "event_time"]
            df = pd.read_parquet(parquet_path, columns=columns)
            sample_count = min(args.mode_samples, len(df))
            samples = df.sample(n=sample_count, random_state=args.seed)
            for row in samples.itertuples(index=False):
                row_dict = row._asdict()
                mode_variants.extend(build_measurement_variants(row_dict))
            max_mode_len = max(
                (len(v["context_tokens"]) + len(v["expected_tokens"]) for v in mode_variants),
                default=1,
            )
            max_prefill_len = max(max_prefill_len, max_mode_len)

    ping_include_timestamp = not args.ping_no_timestamp
    ping_src_ip = None
    ping_targets = []
    if "ping" in selected:
        ping_src_ip = get_src_ip()
        regular_ips = parse_ip_list(args.regular_ips)
        anchor_ips = parse_ip_list(args.anchor_ips)
        ping_targets = build_ping_targets(regular_ips, anchor_ips)
        if ping_targets:
            sample_dst = ping_targets[0]["ip"]
            cond_tokens = create_conditioning_tokens(
                ping_src_ip, sample_dst, include_timestamp=ping_include_timestamp
            )
            max_prefill_len = max(max_prefill_len, len(cond_tokens))

    max_target_len = max_prefill_len + 3
    use_gpu = IN_MODAL_RUNTIME or os.environ.get("JAX_PLATFORMS") == "gpu"

    if not selected:
        print("Nothing selected to run.")
        return

    metrics["run"]["selected"] = sorted(selected)

    print(f"Output directory: {output_dir}")
    print(f"Max prefill length: {max_prefill_len}")
    print(f"Checkpoint: {resolved_checkpoint}")

    config, checkpoint_path = build_config(
        resolved_checkpoint,
        args.config,
        max_prefill_len,
        max_target_len,
        use_gpu=use_gpu,
    )

    with max_utils.maybe_get_transformer_engine_context(config):
        engine, params = setup_engine(config)
        print(f"Loaded params from {checkpoint_path}")

        if "timestamps" in selected:
            print("Computing RTT logprobs (timestamp vs no timestamp)...")
            logps_full = []
            logps_none = []
            for context_full, expected_full, context_none, expected_none in rtt_pairs:
                tokens_full = context_full + expected_full
                logp_full, jax_rng = compute_prompt_logprobs(
                    engine, params, config, tokens_full, jax_rng
                )
                start_full = len(context_full)
                end_full = start_full + len(expected_full)
                logps_full.extend(
                    [lp for lp in logp_full[start_full:end_full] if not np.isnan(lp)]
                )

                tokens_none = context_none + expected_none
                logp_none, jax_rng = compute_prompt_logprobs(
                    engine, params, config, tokens_none, jax_rng
                )
                start_none = len(context_none)
                end_none = start_none + len(expected_none)
                logps_none.extend(
                    [lp for lp in logp_none[start_none:end_none] if not np.isnan(lp)]
                )

            flat_full = np.array(logps_full, dtype=np.float32)
            flat_none = np.array(logps_none, dtype=np.float32)
            timestamp_payload = {
                "run": metrics["run"],
                "params": {
                    "timestamp_parquet": str(args.timestamp_parquet),
                    "timestamp_contexts": args.timestamp_contexts,
                    "timestamp_sample_rows": args.timestamp_sample_rows,
                    "hist_bins": args.hist_bins,
                },
                "timestamp_accuracy": {
                    "num_measurements": len(rtt_pairs),
                    "num_tokens_full": int(flat_full.size),
                    "num_tokens_none": int(flat_none.size),
                    "full": summarize_logps(flat_full),
                    "none": summarize_logps(flat_none),
                    "full_logps": flat_full.tolist(),
                    "none_logps": flat_none.tolist(),
                },
            }
            timestamp_path = metrics_dir / "timestamp_metrics.json"
            with timestamp_path.open("w", encoding="utf-8") as f:
                json.dump(timestamp_payload, f)
            metrics["timestamp_metrics_file"] = str(timestamp_path)

        if "modes" in selected:
            print("Computing prediction-mode accuracy...")
            group_logps = {}
            group_labels = {}
            for variant in mode_variants:
                context = variant["context_tokens"]
                expected = variant["expected_tokens"]
                if not expected:
                    continue
                tokens = context + expected
                logp, jax_rng = compute_prompt_logprobs(
                    engine, params, config, tokens, jax_rng
                )
                start = len(context)
                end = start + len(expected)
                expected_logp = logp[start:end]
                group = group_key_for_expected(expected)
                if group == "unknown" or group == "empty":
                    continue
                if group not in group_logps:
                    group_logps[group] = [
                        [] for _ in range(len(expected))
                    ]
                    group_labels[group] = decode_token_stream_pretty(expected)
                if len(expected) != len(group_logps[group]):
                    continue
                for idx, lp in enumerate(expected_logp):
                    if not np.isnan(lp):
                        group_logps[group][idx].append(float(lp))

            group_summary = {}
            for group, per_pos in group_logps.items():
                flat = [lp for pos in per_pos for lp in pos]
                group_summary[group] = summarize_logps(np.array(flat))

            order = [
                "src_ipv4",
                "src_ipv6",
                "dst_ipv4",
                "dst_ipv6",
                "rtt",
                "rtt_failed",
                "timestamp_abs",
                "timestamp_delta1",
                "timestamp_delta4",
            ]
            group_payload = {}
            for group, per_pos in group_logps.items():
                labels = group_labels.get(group, [str(i) for i in range(len(per_pos))])
                group_payload[group] = {
                    "labels": labels,
                    "per_pos_logps": per_pos,
                    "summary": group_summary.get(group, {}),
                }

            mode_payload = {
                "run": metrics["run"],
                "params": {
                    "parquet": str(args.parquet),
                    "mode_samples": args.mode_samples,
                },
                "num_variants": len(mode_variants),
                "group_order": order,
                "groups": group_payload,
            }
            mode_path = metrics_dir / "mode_metrics.json"
            with mode_path.open("w", encoding="utf-8") as f:
                json.dump(mode_payload, f)
            metrics["mode_metrics_file"] = str(mode_path)

        if "ping" in selected:
            print("Running live ping evaluation...")
            print(f"Source IP: {ping_src_ip}")
            ping_sampling_strategy = args.ping_sampling_strategy.lower()
            ping_results = []
            ping_lock = threading.Lock()
            ping_real = {}
            if args.ping_workers > 1:
                print(f"Pinging targets in parallel (workers={args.ping_workers})")
                with ThreadPoolExecutor(max_workers=args.ping_workers) as executor:
                    future_map = {}
                    for target in ping_targets:
                        dst_ip = target["ip"]
                        label = target["label"]
                        future = executor.submit(
                            ping_series,
                            label,
                            dst_ip,
                            args.pings_per_ip,
                            2,
                            True,
                            ping_lock,
                        )
                        future_map[future] = dst_ip
                    for future in as_completed(future_map):
                        dst_ip = future_map[future]
                        ping_real[dst_ip] = future.result()
            else:
                for target in ping_targets:
                    dst_ip = target["ip"]
                    label = target["label"]
                    ping_real[dst_ip] = ping_series(
                        label,
                        dst_ip,
                        args.pings_per_ip,
                        2,
                        True,
                        ping_lock,
                    )

            for target in ping_targets:
                dst_ip = target["ip"]
                label = target["label"]
                group = target.get("group")
                real_rtts = ping_real.get(dst_ip, [])
                success_count = sum(1 for r in real_rtts if r >= 0)
                if success_count == 0:
                    print(f"{label}: all pings failed; keeping for timeout mass")

                cond_tokens = create_conditioning_tokens(
                    ping_src_ip, dst_ip, include_timestamp=ping_include_timestamp
                )
                model_rtts, model_timeout_count, model_invalid_count = sample_rtt_from_model(
                    engine,
                    params,
                    config,
                    cond_tokens,
                    num_samples=args.model_samples,
                    temperature=args.temperature,
                    sampling_strategy=ping_sampling_strategy,
                )
                model_total = (
                    len(model_rtts) + model_timeout_count + model_invalid_count
                )
                model_timeout_rate = (
                    model_timeout_count / model_total if model_total else 0.0
                )
                real_timeout_rate = (
                    (len(real_rtts) - success_count) / len(real_rtts)
                    if real_rtts
                    else 0.0
                )
                real_timeout_count = len(real_rtts) - success_count

                range_min, range_max = compute_rtt_range(
                    real_rtts,
                    model_rtts,
                    args.max_rtt,
                )
                real_dist, _ = discretize_rtt_with_timeout(
                    real_rtts,
                    real_timeout_count,
                    bins=args.ping_bins,
                    range_min=range_min,
                    range_max=range_max,
                )
                model_dist, _ = discretize_rtt_with_timeout(
                    model_rtts,
                    model_timeout_count + model_invalid_count,
                    bins=args.ping_bins,
                    range_min=range_min,
                    range_max=range_max,
                )
                kl_value = kl_divergence(real_dist, model_dist)

                ping_results.append(
                    {
                        "dst_ip": dst_ip,
                        "group": group,
                        "label": label,
                        "kl": kl_value,
                        "real_success_count": success_count,
                        "model_sample_count": len(model_rtts),
                        "model_timeout_count": model_timeout_count,
                        "model_invalid_count": model_invalid_count,
                        "real_timeout_count": real_timeout_count,
                        "real_timeout_rate": real_timeout_rate,
                        "model_timeout_rate": model_timeout_rate,
                        "range_min": range_min,
                        "range_max": range_max,
                        "real_rtts": real_rtts,
                        "model_rtts": model_rtts,
                    }
                )

            avg_kl = (
                float(np.mean([r["kl"] for r in ping_results]))
                if ping_results
                else float("nan")
            )

            ping_payload = {
                "run": metrics["run"],
                "params": {
                    "pings_per_ip": args.pings_per_ip,
                    "model_samples": args.model_samples,
                    "ping_bins": args.ping_bins,
                    "max_rtt": args.max_rtt,
                    "temperature": args.temperature,
                    "sampling_strategy": ping_sampling_strategy,
                    "ping_workers": args.ping_workers,
                    "ping_no_timestamp": args.ping_no_timestamp,
                    "regular_ips": args.regular_ips,
                    "anchor_ips": args.anchor_ips,
                },
                "ping": {
                    "src_ip": ping_src_ip,
                    "include_timestamp": ping_include_timestamp,
                    "targets": [
                        {
                            "dst_ip": r["dst_ip"],
                            "group": r.get("group"),
                            "label": r["label"],
                            "kl": r["kl"],
                            "real_success_count": r["real_success_count"],
                            "model_sample_count": r["model_sample_count"],
                            "model_timeout_count": r["model_timeout_count"],
                            "model_invalid_count": r["model_invalid_count"],
                            "real_timeout_count": r["real_timeout_count"],
                            "real_timeout_rate": r["real_timeout_rate"],
                            "model_timeout_rate": r["model_timeout_rate"],
                            "range_min": r["range_min"],
                            "range_max": r["range_max"],
                            "real_rtts": r["real_rtts"],
                            "model_rtts": r["model_rtts"],
                        }
                        for r in ping_results
                    ],
                    "average_kl": avg_kl,
                },
            }
            ping_path = metrics_dir / "ping_metrics.json"
            with ping_path.open("w", encoding="utf-8") as f:
                json.dump(ping_payload, f)
            metrics["ping_metrics_file"] = str(ping_path)
            if not math.isnan(avg_kl):
                print(f"Average KL divergence: {avg_kl:.4f}")

    run_path = output_dir / "run.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Wrote run summary to {run_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if MODAL_AVAILABLE:

    @app.function(
        image=image,
        gpu="A10G",
        timeout=60 * 60,
        volumes={"/mnt": shared_vol},
    )
    def eval_on_modal(
        param_only_run_name: str | None = None,
        config: str | None = None,
        output_dir: str | None = None,
        run_name: str | None = None,
        seed: int | None = None,
        only: str | None = None,
        timestamp_parquet: str | None = None,
        timestamp_contexts: int | None = None,
        timestamp_sample_rows: int | None = None,
        max_length: int | None = None,
        hist_bins: int | None = None,
        parquet: str | None = None,
        mode_samples: int | None = None,
        regular_ips: str | None = None,
        anchor_ips: str | None = None,
        pings_per_ip: int | None = None,
        model_samples: int | None = None,
        ping_bins: int | None = None,
        ping_workers: int | None = None,
        max_rtt: int | None = None,
        temperature: float | None = None,
        ping_sampling_strategy: str | None = None,
        ping_no_timestamp: bool = False,
    ):
        parser = build_parser()
        args = parser.parse_args([])

        if param_only_run_name is not None:
            args.param_only_run_name = param_only_run_name
        args.config = config or "src/MaxText/configs/latency_network.yml"
        if output_dir is not None:
            args.output_dir = output_dir
        if run_name is not None:
            args.run_name = run_name
        if seed is not None:
            args.seed = seed
        if only is not None:
            args.only = only
        if timestamp_parquet is not None:
            args.timestamp_parquet = timestamp_parquet
        if timestamp_contexts is not None:
            args.timestamp_contexts = timestamp_contexts
        if timestamp_sample_rows is not None:
            args.timestamp_sample_rows = timestamp_sample_rows
        if max_length is not None:
            args.max_length = max_length
        if hist_bins is not None:
            args.hist_bins = hist_bins
        if parquet is not None:
            args.parquet = parquet
        if mode_samples is not None:
            args.mode_samples = mode_samples
        if regular_ips is not None:
            args.regular_ips = regular_ips
        if anchor_ips is not None:
            args.anchor_ips = anchor_ips
        if pings_per_ip is not None:
            args.pings_per_ip = pings_per_ip
        if model_samples is not None:
            args.model_samples = model_samples
        if ping_bins is not None:
            args.ping_bins = ping_bins
        if ping_workers is not None:
            args.ping_workers = ping_workers
        if max_rtt is not None:
            args.max_rtt = max_rtt
        if temperature is not None:
            args.temperature = temperature
        if ping_sampling_strategy is not None:
            args.ping_sampling_strategy = ping_sampling_strategy
        args.ping_no_timestamp = ping_no_timestamp

        run(args)


if __name__ == "__main__":
    main()
