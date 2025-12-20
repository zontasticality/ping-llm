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
    return bool(os.environ.get("MODAL_IS_REMOTE")) or (Path("/workspace") / "src").exists()


IN_MODAL_RUNTIME = _in_modal_runtime()
if not IN_MODAL_RUNTIME:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import argparse
import sys
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

from src.MaxText.input_pipeline.network_tokenization import MEASUREMENT_START, FAILED
from src.MaxText import pyconfig, model_creation_utils, maxtext_utils, max_utils
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
    APP_NAME = "ping-llm-eval-ordering"
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
        .apt_install("git", "build-essential", "cmake", "ninja-build")
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
        .pip_install("pandas", "pyarrow")
        # Stage 3: Copy code
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    )

    app = modal.App(APP_NAME)
    shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


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
        List of token IDs with fields in specified order
    """
    from src.MaxText.input_pipeline.network_tokenization import (
        encode_ip_merged, encode_rtt_exponent_mantissa,
        encode_timestamp_delta
    )

    # Build field blocks
    field_blocks = {}
    field_blocks['src'] = encode_ip_merged(row['src_addr'], row['ip_version'], is_src=True)
    field_blocks['dst'] = encode_ip_merged(row['dst_addr'], row['ip_version'], is_src=False)

    if row['rtt'] < 0:
        field_blocks['rtt'] = [FAILED]
    else:
        field_blocks['rtt'] = encode_rtt_exponent_mantissa(row['rtt'])

    field_blocks['timestamp'] = encode_timestamp_delta(row['event_time'], prev_timestamp)

    # Build token sequence in specified order
    tokens = [MEASUREMENT_START]
    for field in field_order:
        if field in field_blocks:
            tokens.extend(field_blocks[field])

    return tokens


def build_measurement_variants(row_dict):
    """Return (label, tokens) pairs for the 4 end-field variants."""
    variants = []
    for end_field, label in VARIANT_END_FIELDS:
        order = [field for field in BASE_FIELDS if field != end_field] + [end_field]
        tokens = create_ordered_measurement(row_dict, order, prev_timestamp=None)
        variants.append((label, tokens))
    return variants


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


def build_batch_inputs(sequences, max_len):
    """Pad variable-length sequences into a batch for model.apply."""
    batch_size = len(sequences)
    input_tokens = np.zeros((batch_size, max_len), dtype=np.int32)
    positions = np.zeros((batch_size, max_len), dtype=np.int32)
    segment_ids = np.zeros((batch_size, max_len), dtype=np.int32)
    lengths = np.zeros((batch_size,), dtype=np.int32)

    for i, seq in enumerate(sequences):
        seq_trim = seq[:max_len]
        length = len(seq_trim)
        if length == 0:
            continue
        lengths[i] = length
        input_tokens[i, :length] = seq_trim
        positions[i, :length] = np.arange(length, dtype=np.int32)
        segment_ids[i, :length] = 1

    return (
        jnp.array(input_tokens),
        jnp.array(positions),
        jnp.array(segment_ids),
        jnp.array(lengths),
    )


def batched_generate(
    model,
    params,
    sequences,
    config,
    max_new_tokens,
    stop_token,
    rng,
    temperature=1.0,
):
    """Generate tokens in batch until stop token or timeout."""
    generated_counts = [0] * len(sequences)
    stop_reasons = [None] * len(sequences)
    active = [True] * len(sequences)
    max_len = config.max_target_length

    for _ in range(max_new_tokens):
        active_indices = [i for i, is_active in enumerate(active) if is_active]
        if not active_indices:
            break

        pruned_indices = []
        for idx in active_indices:
            if len(sequences[idx]) >= max_len:
                active[idx] = False
                stop_reasons[idx] = "max_len"
            else:
                pruned_indices.append(idx)
        active_indices = pruned_indices
        if not active_indices:
            continue

        batch_sequences = [sequences[i] for i in active_indices]
        input_tokens, positions, segment_ids, lengths = build_batch_inputs(batch_sequences, max_len)

        logits, _ = model.apply(
            params,
            input_tokens,
            positions,
            decoder_segment_ids=segment_ids,
            enable_dropout=False,
            rngs={"dropout": rng, "params": rng},
            mutable=["intermediates"],
        )

        rng, step_rng = jax.random.split(rng)
        next_logits = logits[jnp.arange(len(active_indices)), lengths - 1, :]
        next_tokens = jax.random.categorical(step_rng, next_logits / temperature)
        next_tokens = np.asarray(next_tokens, dtype=np.int32).tolist()

        for idx, token in zip(active_indices, next_tokens):
            sequences[idx].append(int(token))
            generated_counts[idx] += 1
            if int(token) == stop_token:
                active[idx] = False
                stop_reasons[idx] = "stop_token"

    for idx, is_active in enumerate(active):
        if is_active:
            stop_reasons[idx] = "timeout"
            active[idx] = False

    return generated_counts, stop_reasons, sequences


def run_eval(checkpoint, config, data, num_samples=100, seed=42, max_new_tokens=20, temperature=1.0):
    """
    Core evaluation logic (can be called from CLI or Modal).

    Args:
        checkpoint: Path to checkpoint directory
        config: Path to config file
        data: Path to training data parquet file
        num_samples: Number of measurements to evaluate
        seed: Random seed
    """
    print("Loading model and checkpoint...")
    use_gpu = IN_MODAL_RUNTIME or os.environ.get("JAX_PLATFORMS") == "gpu"
    config_obj, checkpoint_path = build_config(checkpoint, config, use_gpu=use_gpu)
    with max_utils.maybe_get_transformer_engine_context(config_obj):
        model, params, mesh = load_model_and_params(checkpoint_path, config_obj)
        print(f"✓ Model loaded from {checkpoint_path}")

        print(f"\nLoading data from {data}...")
        columns = ["src_addr", "dst_addr", "ip_version", "rtt", "event_time"]
        df = pd.read_parquet(data, columns=columns)
        print(f"✓ Loaded {len(df)} measurements")

        samples = df.sample(n=min(num_samples, len(df)), random_state=seed)
        streams = []
        stream_meta = []

        print("\nPreparing measurement variants...")
        for idx, row in enumerate(samples.itertuples(index=False)):
            if idx % 10 == 0:
                print(f"  Processing sample {idx+1}/{len(samples)}...")
            row_dict = row._asdict()
            for label, tokens in build_measurement_variants(row_dict):
                streams.append(tokens)
                stream_meta.append({"variant": label, "sample_index": idx})

        print(f"\nGenerating up to {max_new_tokens} tokens per stream...")
        rng = jax.random.PRNGKey(seed)
        with mesh, nn_partitioning.axis_rules(config_obj.logical_axis_rules):
            generated_counts, stop_reasons, _ = batched_generate(
                model,
                params,
                streams,
                config_obj,
                max_new_tokens=max_new_tokens,
                stop_token=MEASUREMENT_START,
                rng=rng,
                temperature=temperature,
            )

    stats = defaultdict(lambda: {"counts": [], "stop_token": 0, "timeout": 0, "max_len": 0})
    for meta, count, reason in zip(stream_meta, generated_counts, stop_reasons):
        variant = meta["variant"]
        stats[variant]["counts"].append(count)
        stats[variant][reason] += 1

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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate model continuations by end field")
    parser.add_argument("--checkpoint",
                       default="outputs/latency_network/full_run/full_run/checkpoints/2000",
                       help="Path to checkpoint directory")
    parser.add_argument("--config", default="src/MaxText/configs/latency_network.yml",
                       help="Config file path")
    parser.add_argument("--data", default="data/training_data.parquet",
                       help="Path to training data parquet file")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of measurements to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                       help="Maximum tokens to generate per stream")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        config=args.config,
        data=args.data,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
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
        checkpoint_dir: str = "full_run/full_run/checkpoints/2000",
        config: str = "src/MaxText/configs/latency_network.yml",
        data_file: str = "training_data.parquet",
        num_samples: int = 100,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        """
        Run eval_ordering_likelihood on Modal with GPU acceleration.

        Args:
            checkpoint_dir: Relative path within /mnt/outputs (e.g., "full_run/full_run/checkpoints/2000")
            config: Config file path (relative to workspace)
            data_file: Data file name within /mnt/data (e.g., "training_data.parquet")
            num_samples: Number of measurements to evaluate
            max_new_tokens: Maximum tokens to generate per stream
            temperature: Sampling temperature
            seed: Random seed
        """
        import sys

        # Set up symlinks for config paths
        os.makedirs(f"{WORKDIR}/outputs", exist_ok=True)
        os.makedirs(f"{WORKDIR}/data", exist_ok=True)

        # Link outputs and data from volume
        if not os.path.exists(f"{WORKDIR}/outputs/latency_network"):
            os.symlink("/mnt/outputs/latency_network", f"{WORKDIR}/outputs/latency_network")
        if not os.path.exists(f"{WORKDIR}/data/{data_file}"):
            os.symlink(f"/mnt/data/{data_file}", f"{WORKDIR}/data/{data_file}")

        # Build checkpoint path
        checkpoint_path = f"/mnt/outputs/latency_network/{checkpoint_dir}"
        data_path = f"{WORKDIR}/data/{data_file}"
        config_path = f"{WORKDIR}/{config}"

        # Prepare argv for main()
        sys.argv = [
            "eval_ordering_likelihood.py",
            "--checkpoint", checkpoint_path,
            "--config", config_path,
            "--data", data_path,
            "--num-samples", str(num_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--temperature", str(temperature),
            "--seed", str(seed),
        ]

        # Set environment to indicate GPU usage
        os.environ["JAX_PLATFORMS"] = "gpu"

        # Change to workspace directory
        os.chdir(WORKDIR)

        # Run main function
        main()
