#!/usr/bin/env python3
"""
Paper metrics collection:
- Writes timestamp, mode, ping, and latency-sampling metrics JSON files
- Use eval_paper_metrics_plot.py to render figures
"""

import os
from pathlib import Path


def _in_modal_runtime():
    return (
        bool(os.environ.get("MODAL_IS_REMOTE")) or (Path("/workspace") / "src").exists()
    )


IN_MODAL_RUNTIME = _in_modal_runtime()

import argparse
import json
import math
import re
import socket
import urllib.request
import subprocess
import sys
import threading
import shutil
import ipaddress
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
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

sys.path.insert(0, str(repo_root / "src"))

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
        .add_local_dir("src", f"{WORKDIR}/src", copy=True)
        .run_commands(
            f"cd {WORKDIR} && uv pip install --system -e '.[cuda12]'",
        )
        .uv_pip_install("pandas", "pyarrow", "duckdb", "scipy", "matplotlib")
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    )

    app = modal.App(APP_NAME)
    shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

try:
    from ping_llm.data.datasource import ProbeRowDataSource
except ImportError:
    ProbeRowDataSource = None
from ping_llm.data.tokenization import (
    BYTE_TOKEN_OFFSET,
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
    VOCAB_SIZE,
)
from ping_llm.inference import load_model, generate, get_logits, get_log_probs

DEFAULT_REGULAR_DOMAINS = [
    "umass.edu",
    "berkeley.edu",
    "cam.ac.uk",
    "ethz.ch",
    "iitb.ac.in",
    "u-tokyo.ac.jp",
    "unsw.edu.au",
    "uct.ac.za",
    "ufrj.br",
    "unam.mx",
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


def compute_prompt_logprobs(model, device, tokens):
    """Compute per-token log probs using PyTorch inference.

    Returns an array of length len(tokens) where index 0 is NaN
    (no prediction for the first token) and index i (for i>=1) is
    log P(token[i] | token[0..i-1]).
    """
    log_probs = get_log_probs(model, tokens, device=device)  # length len(tokens)-1
    # Prepend NaN so indices align with the token list
    result = np.empty(len(tokens), dtype=np.float32)
    result[0] = float("nan")
    result[1:] = log_probs.cpu().numpy()
    return result


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
        event_time = row_dict["event_time"]
        context_full, expected_full = build_rtt_prediction_tokens(
            row_dict, include_timestamp=True
        )
        context_none, expected_none = build_rtt_prediction_tokens(
            row_dict, include_timestamp=False
        )
        pairs.append(
            {
                "context_full": context_full,
                "expected_full": expected_full,
                "context_none": context_none,
                "expected_none": expected_none,
                "rtt_ms": float(row_dict["rtt"]),
                "event_hour": int(event_time.hour),
            }
        )
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
    ip_version=4,
):
    try:
        ping_cmd = ["ping", "-c", "1", "-W", str(timeout), ip]
        if ip_version == 6:
            ping_cmd.insert(1, "-6")
        result = subprocess.run(
            ping_cmd,
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


def ping_series(label, ip, count, timeout, show_output, log_lock=None, ip_version=4):
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
            ip_version=ip_version,
        )
        rtts.append(rtt)
    return rtts


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_latency_targets(value):
    if not value:
        return []
    targets = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in ("timeout", "time-out", "failed", "fail"):
            targets.append({"label": "Timeout", "target_ms": None, "is_timeout": True})
            continue
        if item.endswith("ms"):
            item = item[:-2].strip()
        try:
            ms = float(item)
        except ValueError:
            continue
        if ms <= 0:
            continue
        targets.append({"label": f"{ms:g}ms", "target_ms": ms, "is_timeout": False})
    return targets


def is_byte_token(token_id):
    return BYTE_TOKEN_OFFSET <= token_id < VOCAB_SIZE


def extract_dst_ip_from_tokens(tokens, ip_version=None):
    if ip_version == 4:
        role_tokens = {DST_IPV4}
    elif ip_version == 6:
        role_tokens = {DST_IPV6}
    else:
        role_tokens = {DST_IPV4, DST_IPV6}

    for idx, token in enumerate(tokens):
        if token not in role_tokens:
            continue
        expected = 4 if token == DST_IPV4 else 16
        end = idx + 1 + expected
        if end > len(tokens):
            continue
        data = tokens[idx + 1 : end]
        if not all(is_byte_token(t) for t in data):
            continue
        try:
            byte_vals = [token_to_byte(t) for t in data]
            ip_obj = ipaddress.ip_address(bytes(byte_vals))
        except Exception:
            continue
        if (token == DST_IPV4 and ip_obj.version != 4) or (
            token == DST_IPV6 and ip_obj.version != 6
        ):
            continue
        if not ip_obj.is_global:
            continue
        return str(ip_obj)
    return None


def choose_ip_version(rng, ipv6_weight, has_ipv6):
    if not has_ipv6:
        return 4
    if rng.random() < ipv6_weight:
        return 6
    return 4


def create_latency_sampling_context(
    src_ip,
    ip_version,
    target_ms,
    include_timestamp=True,
    current_time=None,
    include_dst_role_token=True,
):
    tokens = [MEASUREMENT_START]
    tokens.extend(encode_ip_merged(src_ip, ip_version, is_src=True))
    if target_ms is None:
        tokens.append(FAILED)
    else:
        tokens.extend(encode_rtt_exponent_mantissa(target_ms))
    if include_timestamp:
        current_time = current_time or datetime.now()
        tokens.extend(encode_timestamp_delta(current_time, prev_time=None))
    if include_dst_role_token:
        tokens.append(DST_IPV6 if ip_version == 6 else DST_IPV4)
    return tokens


def resolve_domains(domains):
    resolved = []
    seen_ips = set()
    for domain in domains:
        domain = domain.strip()
        if not domain:
            continue
        try:
            infos = socket.getaddrinfo(domain, None, family=socket.AF_INET)
        except Exception:
            continue
        ip = None
        for info in infos:
            candidate = info[4][0]
            if candidate:
                ip = candidate
                break
        if ip is None or ip in seen_ips:
            continue
        resolved.append({"domain": domain, "ip": ip})
        seen_ips.add(ip)
    return resolved


def build_ping_targets(regular_domains, anchor_ips):
    targets = []
    seen = set()
    resolved = resolve_domains(regular_domains)
    for idx, entry in enumerate(resolved, start=1):
        ip = entry["ip"]
        if ip in seen:
            continue
        seen.add(ip)
        targets.append(
            {
                "ip": ip,
                "domain": entry["domain"],
                "label": f"{entry['domain']} ({ip})",
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
    public_ip = None
    for url in (
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://ifconfig.me/ip",
    ):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                candidate = resp.read().decode("utf-8").strip()
            if re.match(r"^\\d+\\.\\d+\\.\\d+\\.\\d+$", candidate):
                public_ip = candidate
                break
        except Exception:
            continue
    if public_ip:
        return public_ip
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        src_ip = sock.getsockname()[0]
        sock.close()
        return src_ip
    except Exception:
        return "127.0.0.1"


def get_src_ipv6():
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        sock.connect(("2001:4860:4860::8888", 80))
        src_ip = sock.getsockname()[0]
        sock.close()
        return src_ip
    except Exception:
        return None


def create_conditioning_tokens(src_ip, dst_ip, include_timestamp=False):
    tokens = [MEASUREMENT_START]
    tokens.extend(encode_ip_merged(src_ip, 4, is_src=True))
    tokens.extend(encode_ip_merged(dst_ip, 4, is_src=False))
    if include_timestamp:
        current_time = datetime.now()
        tokens.extend(encode_timestamp_delta(current_time, prev_time=None))
    return tokens


def sample_rtt_from_model(
    model,
    device,
    conditioning_tokens,
    num_samples=100,
    temperature=1.0,
    **_kwargs,
):
    """Sample RTT values by generating 3 tokens (RTT_START + byte1 + byte2)."""
    rtt_samples = []
    timeout_count = 0
    invalid_count = 0

    for _ in range(num_samples):
        generated = generate(
            model,
            conditioning_tokens,
            max_new_tokens=3,
            temperature=temperature,
            device=device,
        )
        new_tokens = generated[len(conditioning_tokens):]
        if len(new_tokens) < 1:
            invalid_count += 1
            continue

        first_token = new_tokens[0]
        try:
            if first_token == FAILED:
                timeout_count += 1
                continue
            if first_token != RTT_START:
                invalid_count += 1
                continue
            if len(new_tokens) < 3:
                invalid_count += 1
                continue
            second_token = new_tokens[1]
            third_token = new_tokens[2]
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

    return rtt_samples, timeout_count, invalid_count


def sample_tokens_from_model(
    model,
    device,
    sequences,
    max_new_tokens,
    temperature=1.0,
    **_kwargs,
):
    """Generate tokens for each prompt sequence using PyTorch inference."""
    if max_new_tokens < 1:
        return [[] for _ in sequences]

    completions = []
    for seq in sequences:
        generated = generate(
            model,
            seq,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        new_tokens = generated[len(seq):]
        completions.append(new_tokens)

    return completions


def sample_destination_ips_for_target(
    model,
    device,
    rng,
    src_ipv4,
    src_ipv6,
    target_ms,
    samples_per_target,
    max_attempts,
    max_new_tokens,
    temperature,
    ipv6_weight,
    include_timestamp=True,
    current_time=None,
    **_kwargs,
):
    samples = []
    seen = set()
    invalid = 0
    duplicate = 0
    attempts = 0
    has_ipv6 = src_ipv6 is not None
    current_time = current_time or datetime.now()

    while len(samples) < samples_per_target and attempts < max_attempts:
        ip_version = choose_ip_version(rng, ipv6_weight, has_ipv6)
        src_ip = src_ipv6 if ip_version == 6 else src_ipv4
        context = create_latency_sampling_context(
            src_ip,
            ip_version,
            target_ms,
            include_timestamp=include_timestamp,
            current_time=current_time,
        )

        completions = sample_tokens_from_model(
            model,
            device,
            [context],
            max_new_tokens,
            temperature=temperature,
        )

        completion = completions[0]
        role_token = DST_IPV6 if ip_version == 6 else DST_IPV4
        ip_str = extract_dst_ip_from_tokens(
            [role_token] + completion, ip_version=ip_version
        )
        if ip_str is None:
            invalid += 1
        else:
            key = (ip_version, ip_str)
            if key in seen:
                duplicate += 1
            else:
                seen.add(key)
                samples.append({"dst_ip": ip_str, "ip_version": ip_version})

        attempts += 1

    meta = {
        "attempted": attempts,
        "invalid": invalid,
        "duplicate": duplicate,
    }
    return samples, meta


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
    range_min = max(0.1, range_min)
    if range_max <= range_min:
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
        return {"timestamps", "modes", "ping", "latency_sampling"}
    return items


def build_parser():
    parser = argparse.ArgumentParser(description="Paper metrics evaluation")
    parser.add_argument(
        "--param-only-run-name",
        default="param_only_checkpoint",
        help="Run name for param-only checkpoints",
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
        help="Comma list: timestamps,modes,ping,latency_sampling,all",
    )

    parser.add_argument(
        "--timestamp-parquet",
        default=DEFAULT_PARQUET,
        help="Parquet path for timestamp evaluation",
    )
    parser.add_argument(
        "--timestamp-contexts",
        type=int,
        default=5000,
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
        "--regular-domains",
        default=",".join(DEFAULT_REGULAR_DOMAINS),
        help="Comma-separated list of regular domains to resolve and ping",
    )
    parser.add_argument(
        "--regular-ips",
        dest="regular_domains",
        default=None,
        help=argparse.SUPPRESS,
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
        help="Sampling temperature for ping and latency sampling",
    )
    parser.add_argument(
        "--ping-sampling-strategy",
        default="weighted",
        help="Sampling strategy for ping and latency sampling (e.g., weighted, greedy, nucleus)",
    )
    parser.add_argument(
        "--ping-no-timestamp",
        action="store_true",
        help="Disable timestamp conditioning for ping evaluation",
    )

    parser.add_argument(
        "--latency-targets",
        default="1,10,50,100,500,1000,timeout",
        help="Comma-separated target RTTs in ms plus 'timeout'",
    )
    parser.add_argument(
        "--latency-samples-per-target",
        type=int,
        default=15,
        help="Destination IP samples per target bucket",
    )
    parser.add_argument(
        "--latency-sample-max-attempts",
        type=int,
        default=200,
        help="Max model sampling attempts per target bucket",
    )
    parser.add_argument(
        "--latency-sample-max-new-tokens",
        type=int,
        default=24,
        help="Max new tokens when sampling destination IP blocks",
    )
    parser.add_argument(
        "--latency-ipv6-weight",
        type=float,
        default=0.5,
        help="Probability of sampling IPv6 when available",
    )
    parser.add_argument(
        "--latency-ping-timeout",
        type=int,
        default=2,
        help="Ping timeout (seconds) for latency sampling",
    )

    return parser


def run(args):
    selected = parse_only_arg(args.only)
    rng = np.random.default_rng(args.seed)

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
        }
    }

    mode_variants = []

    rtt_pairs = []
    latency_targets = []
    latency_src_ipv4 = None
    latency_src_ipv6 = None
    latency_context_time = None
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

    ping_include_timestamp = not args.ping_no_timestamp
    ping_src_ip = None
    ping_targets = []
    if "ping" in selected:
        ping_src_ip = get_src_ip()
        regular_domains = parse_list(args.regular_domains)
        anchor_ips = parse_list(args.anchor_ips)
        ping_targets = build_ping_targets(regular_domains, anchor_ips)

    if "latency_sampling" in selected:
        latency_targets = parse_latency_targets(args.latency_targets)
        if not latency_targets:
            print("No latency targets parsed; skipping latency_sampling.")
            selected.discard("latency_sampling")
        else:
            latency_src_ipv4 = get_src_ip()
            latency_src_ipv6 = get_src_ipv6()
            latency_context_time = datetime.now()

    if not selected:
        print("Nothing selected to run.")
        return

    metrics["run"]["selected"] = sorted(selected)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {resolved_checkpoint}")
    print(f"Device: {device}")

    model, model_cfg = load_model(resolved_checkpoint, device=device)
    print(f"Loaded model from {resolved_checkpoint}")

    if "timestamps" in selected:
        print("Computing RTT logprobs (timestamp vs no timestamp)...")
        logps_full = []
        logps_none = []
        full_rtt_ms = []
        none_rtt_ms = []
        full_event_hour = []
        none_event_hour = []
        for pair in rtt_pairs:
            context_full = pair["context_full"]
            expected_full = pair["expected_full"]
            context_none = pair["context_none"]
            expected_none = pair["expected_none"]
            rtt_ms = pair["rtt_ms"]
            event_hour = pair["event_hour"]
            tokens_full = context_full + expected_full
            logp_full = compute_prompt_logprobs(model, device, tokens_full)
            start_full = len(context_full)
            end_full = start_full + len(expected_full)
            for lp in logp_full[start_full:end_full]:
                if np.isnan(lp):
                    continue
                logps_full.append(lp)
                full_rtt_ms.append(rtt_ms)
                full_event_hour.append(event_hour)

            tokens_none = context_none + expected_none
            logp_none = compute_prompt_logprobs(model, device, tokens_none)
            start_none = len(context_none)
            end_none = start_none + len(expected_none)
            for lp in logp_none[start_none:end_none]:
                if np.isnan(lp):
                    continue
                logps_none.append(lp)
                none_rtt_ms.append(rtt_ms)
                none_event_hour.append(event_hour)

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
                "full_rtt_ms": full_rtt_ms,
                "none_rtt_ms": none_rtt_ms,
                "full_event_hour": full_event_hour,
                "none_event_hour": none_event_hour,
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
            logp = compute_prompt_logprobs(model, device, tokens)
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
            domain = target.get("domain")
            real_rtts = ping_real.get(dst_ip, [])
            success_count = sum(1 for r in real_rtts if r >= 0)
            if success_count == 0:
                print(f"{label}: all pings failed; keeping for timeout mass")

            cond_tokens = create_conditioning_tokens(
                ping_src_ip, dst_ip, include_timestamp=ping_include_timestamp
            )
            model_rtts, model_timeout_count, model_invalid_count = sample_rtt_from_model(
                model,
                device,
                cond_tokens,
                num_samples=args.model_samples,
                temperature=args.temperature,
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
                    "domain": domain,
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
                "regular_domains": args.regular_domains,
                "anchor_ips": args.anchor_ips,
            },
            "ping": {
                "src_ip": ping_src_ip,
                "include_timestamp": ping_include_timestamp,
                "targets": [
                    {
                        "dst_ip": r["dst_ip"],
                        "domain": r.get("domain"),
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

    if "latency_sampling" in selected:
        print("Running latency-conditioned destination sampling...")
        latency_sampling_strategy = args.ping_sampling_strategy.lower()
        latency_ping_timeout_s = args.latency_ping_timeout
        latency_ping_timeout_ms = float(latency_ping_timeout_s) * 1000.0
        latency_src_ipv4 = latency_src_ipv4 or get_src_ip()
        latency_src_ipv6 = latency_src_ipv6 or get_src_ipv6()
        latency_context_time = latency_context_time or datetime.now()
        print(f"Source IPv4: {latency_src_ipv4}")
        if latency_src_ipv6:
            print(f"Source IPv6: {latency_src_ipv6}")
        else:
            print("Source IPv6: unavailable (sampling IPv4 only)")

        latency_results = []
        ping_lock = threading.Lock()

        for target in latency_targets:
            label = target["label"]
            target_ms = target["target_ms"]
            is_timeout = target["is_timeout"]
            print(f"Sampling destinations for target {label}...")

            samples, sample_meta = sample_destination_ips_for_target(
                model,
                device,
                rng,
                latency_src_ipv4,
                latency_src_ipv6,
                target_ms,
                args.latency_samples_per_target,
                args.latency_sample_max_attempts,
                args.latency_sample_max_new_tokens,
                args.temperature,
                args.latency_ipv6_weight,
                include_timestamp=True,
                current_time=latency_context_time,
            )

            if not samples:
                print(f"{label}: no valid destination samples")

            sample_rtts = [None] * len(samples)
            if samples and args.ping_workers > 1:
                print(f"{label}: pinging samples in parallel (workers={args.ping_workers})")
                with ThreadPoolExecutor(max_workers=args.ping_workers) as executor:
                    future_map = {}
                    for idx, sample in enumerate(samples):
                        dst_ip = sample["dst_ip"]
                        ip_version = sample["ip_version"]
                        ping_label = f"{label} {idx + 1}/{len(samples)}"
                        future = executor.submit(
                            ping_series,
                            ping_label,
                            dst_ip,
                            args.pings_per_ip,
                            latency_ping_timeout_s,
                            False,
                            ping_lock,
                            ip_version,
                        )
                        future_map[future] = idx
                    for future in as_completed(future_map):
                        idx = future_map[future]
                        sample_rtts[idx] = future.result()
            else:
                for idx, sample in enumerate(samples):
                    dst_ip = sample["dst_ip"]
                    ip_version = sample["ip_version"]
                    ping_label = f"{label} {idx + 1}/{len(samples)}"
                    sample_rtts[idx] = ping_series(
                        ping_label,
                        dst_ip,
                        args.pings_per_ip,
                        latency_ping_timeout_s,
                        False,
                        ping_lock,
                        ip_version,
                    )

            errors = []
            medians = []
            timeout_rates = []
            ipv4_count = 0
            ipv6_count = 0
            samples_payload = []
            for sample, rtts in zip(samples, sample_rtts):
                ip_version = sample["ip_version"]
                if ip_version == 6:
                    ipv6_count += 1
                else:
                    ipv4_count += 1
                rtts = rtts or []
                success_count = sum(1 for r in rtts if r >= 0)
                timeout_count = len(rtts) - success_count
                timeout_rate = timeout_count / len(rtts) if rtts else 1.0
                median_rtt = compute_rtt_median(rtts)
                if median_rtt is None:
                    median_rtt = latency_ping_timeout_ms

                if is_timeout:
                    if success_count == 0:
                        error_log2 = 0.0
                    else:
                        ratio = max(median_rtt, 1e-3) / latency_ping_timeout_ms
                        error_log2 = abs(math.log2(ratio))
                else:
                    ratio = max(median_rtt, 1e-3) / max(target_ms, 1e-3)
                    error_log2 = abs(math.log2(ratio))

                errors.append(error_log2)
                medians.append(median_rtt)
                timeout_rates.append(timeout_rate)
                samples_payload.append(
                    {
                        "dst_ip": sample["dst_ip"],
                        "ip_version": ip_version,
                        "rtts": rtts,
                        "median_rtt_ms": median_rtt,
                        "success_count": success_count,
                        "timeout_count": timeout_count,
                        "timeout_rate": timeout_rate,
                        "error_log2": error_log2,
                    }
                )

            summary = {
                "sampled_count": len(samples),
                "ipv4_count": ipv4_count,
                "ipv6_count": ipv6_count,
                "mean_error_log2": float(np.mean(errors)) if errors else None,
                "median_error_log2": float(np.median(errors)) if errors else None,
                "mean_median_rtt_ms": float(np.mean(medians)) if medians else None,
                "median_median_rtt_ms": float(np.median(medians)) if medians else None,
                "mean_timeout_rate": float(np.mean(timeout_rates)) if timeout_rates else None,
            }

            latency_results.append(
                {
                    "label": label,
                    "target_ms": target_ms,
                    "is_timeout": is_timeout,
                    "requested_samples": args.latency_samples_per_target,
                    "sampled_count": len(samples),
                    "sampling_attempts": sample_meta["attempted"],
                    "sampling_invalid": sample_meta["invalid"],
                    "sampling_duplicate": sample_meta["duplicate"],
                    "samples": samples_payload,
                    "summary": summary,
                }
            )

        latency_payload = {
            "run": metrics["run"],
            "params": {
                "targets": args.latency_targets,
                "samples_per_target": args.latency_samples_per_target,
                "sampling_max_attempts": args.latency_sample_max_attempts,
                "sampling_max_new_tokens": args.latency_sample_max_new_tokens,
                "temperature": args.temperature,
                "sampling_strategy": latency_sampling_strategy,
                "ipv6_weight": args.latency_ipv6_weight,
                "pings_per_ip": args.pings_per_ip,
                "ping_timeout_s": latency_ping_timeout_s,
                "include_timestamp": True,
            },
            "latency_sampling": {
                "src_ipv4": latency_src_ipv4,
                "src_ipv6": latency_src_ipv6,
                "targets": latency_results,
            },
        }
        latency_path = metrics_dir / "latency_sampling_metrics.json"
        with latency_path.open("w", encoding="utf-8") as f:
            json.dump(latency_payload, f)
        metrics["latency_sampling_metrics_file"] = str(latency_path)

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
        regular_domains: str | None = None,
        anchor_ips: str | None = None,
        pings_per_ip: int | None = None,
        model_samples: int | None = None,
        ping_bins: int | None = None,
        ping_workers: int | None = None,
        max_rtt: int | None = None,
        temperature: float | None = None,
        ping_sampling_strategy: str | None = None,
        ping_no_timestamp: bool = False,
        latency_targets: str | None = None,
        latency_samples_per_target: int | None = None,
        latency_sample_max_attempts: int | None = None,
        latency_sample_max_new_tokens: int | None = None,
        latency_ipv6_weight: float | None = None,
        latency_ping_timeout: int | None = None,
    ):
        parser = build_parser()
        args = parser.parse_args([])

        if param_only_run_name is not None:
            args.param_only_run_name = param_only_run_name
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
        if regular_domains is not None:
            args.regular_domains = regular_domains
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
        if latency_targets is not None:
            args.latency_targets = latency_targets
        if latency_samples_per_target is not None:
            args.latency_samples_per_target = latency_samples_per_target
        if latency_sample_max_attempts is not None:
            args.latency_sample_max_attempts = latency_sample_max_attempts
        if latency_sample_max_new_tokens is not None:
            args.latency_sample_max_new_tokens = latency_sample_max_new_tokens
        if latency_ipv6_weight is not None:
            args.latency_ipv6_weight = latency_ipv6_weight
        if latency_ping_timeout is not None:
            args.latency_ping_timeout = latency_ping_timeout

        run(args)


if __name__ == "__main__":
    main()
