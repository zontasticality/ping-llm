#!/usr/bin/env python3
"""
Evaluate next-token prediction accuracy and show predicted vs actual tokens.

This script:
1. Loads a trained checkpoint
2. Samples sequences from evaluation data
3. For each position in sequence, shows predicted vs actual next token
4. Displays results in pretty-printed format
5. Reports accuracy metrics

Usage (local CPU):
    python scripts/eval_next_token_predictions.py \
        --checkpoint checkpoints/full_run/checkpoints/2000 \
        --data data/eval_probe_rows.arrayrecord \
        --num-sequences 5 \
        --max-length 100

Usage (Modal GPU):
    modal run scripts/eval_next_token_predictions.py::eval_on_modal \
        --num-sequences 10
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
from typing import List, Tuple
import random

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
workspace_root = Path("/workspace")
if IN_MODAL_RUNTIME and (workspace_root / "src").exists():
    sys.path.insert(0, str(workspace_root))
else:
    sys.path.insert(0, str(repo_root))

import jax
import jax.numpy as jnp
import numpy as np

from src.MaxText.input_pipeline.network_tokenization import (
    decode_token_stream_pretty,
    TOKEN_NAMES,
    token_to_byte,
    BYTE_TOKEN_OFFSET,
    VOCAB_SIZE,
)
from src.MaxText.input_pipeline._probe_chunk_datasource import ProbeRowDataSource, ProbeRowSampler

# ============================================================================
# Modal Setup (for GPU acceleration)
# ============================================================================
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

if MODAL_AVAILABLE:
    APP_NAME = "ping-llm-eval-next-token"
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
        )
        .uv_pip_install("google-jetstream")
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
DEFAULT_DATA_MODAL = "/mnt/data/probe_rows/test.arrayrecord"
DEFAULT_DATA_LOCAL = "data/probe_rows/test.arrayrecord"
DEFAULT_DATA = DEFAULT_DATA_MODAL if IN_MODAL_RUNTIME else DEFAULT_DATA_LOCAL

# ============================================================================
# Token Pretty Printing
# ============================================================================

def format_token_comparison(pos: int, actual: int, predicted: int, correct: bool) -> str:
    """Format a single token comparison for display."""
    actual_str = _token_to_str(actual)
    predicted_str = _token_to_str(predicted)

    status = "✓" if correct else "✗"
    color = "\033[92m" if correct else "\033[91m"  # Green if correct, red if wrong
    reset = "\033[0m"

    return f"{color}{status}{reset} Pos {pos:3d}: Actual={actual_str:20s} | Predicted={predicted_str:20s}"


def _token_to_str(token: int) -> str:
    """Convert a single token to a readable string."""
    if token in TOKEN_NAMES:
        return TOKEN_NAMES[token]
    elif BYTE_TOKEN_OFFSET <= token < VOCAB_SIZE:
        val = token_to_byte(token)
        return f"Byte(0x{val:02X}/{val:3d})"
    else:
        return f"Unknown({token})"


# ============================================================================
# Model Evaluation
# ============================================================================

def build_config(checkpoint_path, config_path, use_gpu=False, max_length=None):
    """Build a MaxText config with resolved paths and eval overrides."""
    from src.MaxText import pyconfig as maxtext_pyconfig

    checkpoint_path = str(Path(checkpoint_path).resolve())
    config_path = str(Path(config_path).resolve())

    argv = [
        "eval_script",
        config_path,
        f"load_parameters_path={checkpoint_path}",
        f"hardware={'gpu' if use_gpu else 'cpu'}",
        "skip_jax_distributed_system=true",
    ]

    if max_length is not None:
        # max_target_length must be > max_prefill_predict_length
        # Set prefill to max_length and target to max_length + a buffer
        argv.append(f"max_prefill_predict_length={max_length}")
        argv.append(f"max_target_length={max_length + 20}")

    # Use dot_product attention for inference
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


def get_logits_for_sequence(engine, params, config, tokens: np.ndarray) -> np.ndarray:
    """
    Get logits for all positions in a sequence.

    Args:
        engine: MaxEngine instance
        params: Model parameters
        config: Config object
        tokens: Token sequence (np.array of shape [seq_len])

    Returns:
        np.ndarray of shape [seq_len, vocab_size] with logits
    """
    from flax.linen import partitioning as nn_partitioning
    from MaxText.common_types import MODEL_MODE_PREFILL, DECODING_ACTIVE_SEQUENCE_INDICATOR

    # Pad sequence
    padded_tokens, true_length = pad_tokens(tokens, config.max_prefill_predict_length)

    # Prepare inputs for model (add batch dimension)
    input_tokens = jnp.expand_dims(jnp.array(padded_tokens), 0)  # [1, seq_len]
    positions = jnp.expand_dims(jnp.arange(input_tokens.shape[1]), 0)  # [1, seq_len]

    # Create sequence indicator (marks valid tokens)
    ones_to_keep = jnp.arange(input_tokens.shape[1]) < true_length
    sequence_indicator = jnp.expand_dims(ones_to_keep * DECODING_ACTIVE_SEQUENCE_INDICATOR, 0)

    # Call model.apply directly to get ALL logits (not just last token)
    rng = jax.random.PRNGKey(0)
    rng, new_rng = jax.random.split(rng)

    with engine._mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        flat_logits, _ = engine.model.apply(
            params,
            input_tokens,
            positions,
            decoder_segment_ids=sequence_indicator,
            enable_dropout=False,
            model_mode=MODEL_MODE_PREFILL,
            rngs={"params": new_rng},
            mutable=["cache"],
            true_length=true_length,
        )

    # Extract logits for valid positions only
    # flat_logits has shape [1, seq_len, vocab_size]
    logits = np.array(flat_logits[0, :true_length, :])  # Shape: [true_length, vocab_size]

    return logits


def evaluate_sequence(
    engine,
    params,
    config,
    tokens: np.ndarray,
    max_positions: int = None
) -> Tuple[List[Tuple[int, int, int, bool]], float]:
    """
    Evaluate next-token prediction for a sequence.

    Args:
        engine: MaxEngine instance
        params: Model parameters
        config: Config
        tokens: Token sequence (np.array)
        max_positions: Maximum positions to evaluate (None = all)

    Returns:
        (comparisons, accuracy) where comparisons is list of (pos, actual, predicted, correct)
    """
    seq_len = len(tokens)
    if max_positions is not None:
        seq_len = min(seq_len, max_positions)

    # Get logits for the sequence
    logits = get_logits_for_sequence(engine, params, config, tokens[:seq_len])

    # Compare predictions with actuals
    comparisons = []
    correct_count = 0

    for pos in range(seq_len - 1):  # Don't predict beyond sequence
        actual_next = int(tokens[pos + 1])
        predicted_next = int(np.argmax(logits[pos]))
        correct = (actual_next == predicted_next)

        comparisons.append((pos, actual_next, predicted_next, correct))
        if correct:
            correct_count += 1

    accuracy = correct_count / len(comparisons) if comparisons else 0.0

    return comparisons, accuracy


# ============================================================================
# Data Loading
# ============================================================================

def load_sequences_from_arrayrecord(arrayrecord_path: str, num_sequences: int, seed: int = 42) -> List[np.ndarray]:
    """Load sequences from arrayrecord file."""
    datasource = ProbeRowDataSource(arrayrecord_path)
    sampler = ProbeRowSampler(
        crop_size=1024,
        avg_tokens_per_measurement=30,
        max_contexts_per_row=1,
        seed=seed,
    )

    sequences = []
    random.seed(seed)
    row_indices = random.sample(range(len(datasource)), min(num_sequences * 2, len(datasource)))

    for idx in row_indices:
        if len(sequences) >= num_sequences:
            break

        row = datasource[idx]
        contexts = sampler.flat_map(row)

        if contexts:
            tokens = contexts[0]["inputs"]
            segmentation = contexts[0]["inputs_segmentation"]

            # Extract non-padding tokens
            valid_len = int(np.sum(segmentation))
            if valid_len > 10:  # Skip very short sequences
                sequences.append(tokens[:valid_len])

    return sequences


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_eval(
    checkpoint: str,
    config_path: str,
    data: str,
    num_sequences: int = 5,
    max_length: int = 100,
    seed: int = 42,
):
    """
    Main evaluation logic.

    Args:
        checkpoint: Path to checkpoint directory
        config_path: Path to config file
        data: Path to evaluation data (arrayrecord file)
        num_sequences: Number of sequences to evaluate
        max_length: Maximum sequence length to evaluate
        seed: Random seed
    """
    from src.MaxText import max_utils

    print(f"\n{'='*80}")
    print("NEXT-TOKEN PREDICTION EVALUATION")
    print(f"{'='*80}\n")

    print(f"Loading data from {data}...")
    sequences = load_sequences_from_arrayrecord(data, num_sequences, seed)
    print(f"✓ Loaded {len(sequences)} sequences")

    print(f"\nLoading engine and checkpoint...")
    use_gpu = IN_MODAL_RUNTIME or os.environ.get("JAX_PLATFORMS") == "gpu"
    config, checkpoint_path = build_config(checkpoint, config_path, use_gpu=use_gpu, max_length=max_length)

    with max_utils.maybe_get_transformer_engine_context(config):
        engine, params = setup_engine(config)
        print(f"✓ Engine initialized with params from {checkpoint_path}\n")

        # Evaluate each sequence
        all_accuracies = []

        for seq_idx, tokens in enumerate(sequences):
            print(f"\n{'='*80}")
            print(f"SEQUENCE {seq_idx + 1}/{len(sequences)}")
            print(f"{'='*80}")
            print(f"Length: {len(tokens)} tokens")

            # Pretty-print the sequence
            pretty_tokens = decode_token_stream_pretty(tokens[:max_length])
            print(f"\nSequence: {' '.join(pretty_tokens[:20])}")
            if len(pretty_tokens) > 20:
                print(f"          ... (+{len(pretty_tokens) - 20} more tokens)")

            # Evaluate
            print(f"\nEvaluating next-token predictions...")
            comparisons, accuracy = evaluate_sequence(
                engine, params, config, tokens, max_positions=max_length
            )
            all_accuracies.append(accuracy)

            print(f"\nAccuracy: {accuracy*100:.1f}% ({sum(1 for _, _, _, c in comparisons if c)}/{len(comparisons)} correct)")

            # Show first N predictions
            print(f"\nFirst 20 predictions:")
            for pos, actual, predicted, correct in comparisons[:20]:
                print(f"  {format_token_comparison(pos, actual, predicted, correct)}")

            if len(comparisons) > 20:
                print(f"\n  ... (+{len(comparisons) - 20} more positions)")

                # Show last few
                print(f"\nLast 10 predictions:")
                for pos, actual, predicted, correct in comparisons[-10:]:
                    print(f"  {format_token_comparison(pos, actual, predicted, correct)}")

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Sequences evaluated: {len(sequences)}")
        print(f"Average accuracy: {np.mean(all_accuracies)*100:.1f}%")
        print(f"Min accuracy: {np.min(all_accuracies)*100:.1f}%")
        print(f"Max accuracy: {np.max(all_accuracies)*100:.1f}%")
        print(f"Std accuracy: {np.std(all_accuracies)*100:.1f}%")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate next-token predictions")
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
        default=DEFAULT_DATA,
        help="Path to evaluation data (arrayrecord file)",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of sequences to evaluate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum sequence length to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        config_path=args.config,
        data=args.data,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
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
        data_file: str = "probe_rows/test.arrayrecord",
        num_sequences: int = 5,
        max_length: int = 100,
        seed: int = 42,
    ):
        """
        Run eval_next_token_predictions on Modal with GPU acceleration.

        Args:
            checkpoint_path: Param-only checkpoint items path
            config: Config file path (relative to workspace)
            data_file: Data file path within /mnt/data (e.g., "probe_rows/test.arrayrecord")
            num_sequences: Number of sequences to evaluate
            max_length: Maximum sequence length to evaluate
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

        # Create nested directory structure if needed
        data_parent = os.path.dirname(data_file)
        if data_parent and not os.path.exists(f"{WORKDIR}/data/{data_parent}"):
            os.makedirs(f"{WORKDIR}/data/{data_parent}", exist_ok=True)
            if os.path.exists(f"/mnt/data/{data_parent}"):
                # Link the parent directory
                for item in os.listdir(f"/mnt/data/{data_parent}"):
                    src = f"/mnt/data/{data_parent}/{item}"
                    dst = f"{WORKDIR}/data/{data_parent}/{item}"
                    if not os.path.exists(dst):
                        os.symlink(src, dst)
        elif not os.path.exists(f"{WORKDIR}/data/{data_file}"):
            os.symlink(f"/mnt/data/{data_file}", f"{WORKDIR}/data/{data_file}")

        # Resolve checkpoint path
        if not checkpoint_path.startswith("/"):
            checkpoint_path = f"{WORKDIR}/{checkpoint_path}"
        data_path = f"{WORKDIR}/data/{data_file}"
        config_path = f"{WORKDIR}/{config}"

        # Prepare argv for main()
        sys.argv = [
            "eval_next_token_predictions.py",
            "--checkpoint",
            checkpoint_path,
            "--config",
            config_path,
            "--data",
            data_path,
            "--num-sequences",
            str(num_sequences),
            "--max-length",
            str(max_length),
            "--seed",
            str(seed),
        ]

        # Set environment to indicate GPU usage
        os.environ["JAX_PLATFORMS"] = "gpu"

        # Change to workspace directory
        os.chdir(WORKDIR)

        # Run main function
        main()
