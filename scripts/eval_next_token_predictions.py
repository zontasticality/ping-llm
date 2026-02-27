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

import argparse
import sys
from typing import List, Tuple
import random

# Add project src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import torch

from ping_llm.data.tokenization import (
    decode_token_stream_pretty,
    TOKEN_NAMES,
    token_to_byte,
    BYTE_TOKEN_OFFSET,
    VOCAB_SIZE,
)
from ping_llm.data.datasource import (
    ProbeRowDataSource,
    ProbeRowSampler,
)
from ping_llm.inference import load_model, get_logits

# ============================================================================
# Modal Setup (for GPU acceleration)
# ============================================================================
try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

IN_MODAL_RUNTIME = bool(os.environ.get("MODAL_IS_REMOTE"))

if MODAL_AVAILABLE:
    APP_NAME = "ping-llm-eval-next-token"
    WORKDIR = "/workspace"
    VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")

    IGNORE_PATTERNS = [
        ".git", ".venv", "__pycache__", "outputs", "logs", "data",
        "archive", "*.parquet", "*.arrayrecord", ".DS_Store",
    ]

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12",
        )
        .entrypoint([])
        .apt_install("git", "build-essential")
        .pip_install("uv")
        .run_commands("uv pip install --system torch pyarrow numpy grain array_record")
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    )

    app = modal.App(APP_NAME)
    shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DEFAULT_CHECKPOINT_MODAL = "/mnt/outputs/checkpoints/default/latest.pt"
DEFAULT_CHECKPOINT_LOCAL = "outputs/checkpoints/default/latest.pt"
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_MODAL if IN_MODAL_RUNTIME else DEFAULT_CHECKPOINT_LOCAL
DEFAULT_DATA_MODAL = "/mnt/data/probe_rows/test.arrayrecord"
DEFAULT_DATA_LOCAL = "data/probe_rows/test.arrayrecord"
DEFAULT_DATA = DEFAULT_DATA_MODAL if IN_MODAL_RUNTIME else DEFAULT_DATA_LOCAL

# ============================================================================
# Token Pretty Printing
# ============================================================================


def format_token_comparison(
    pos: int, actual: int, predicted: int, correct: bool
) -> str:
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


def get_logits_for_sequence(model, tokens: np.ndarray, device: str) -> np.ndarray:
    """
    Get logits for all positions in a sequence.

    Args:
        model: GPT model
        tokens: Token sequence (np.array of shape [seq_len])
        device: Device string

    Returns:
        np.ndarray of shape [seq_len, vocab_size] with logits
    """
    logits = get_logits(model, tokens.tolist(), device=device)
    return logits.cpu().numpy()


def evaluate_sequence(
    model, tokens: np.ndarray, device: str, max_positions: int = None
) -> Tuple[List[Tuple[int, int, int, bool]], float]:
    """
    Evaluate next-token prediction for a sequence.

    Args:
        model: GPT model
        tokens: Token sequence (np.array)
        device: Device string
        max_positions: Maximum positions to evaluate (None = all)

    Returns:
        (comparisons, accuracy) where comparisons is list of (pos, actual, predicted, correct)
    """
    seq_len = len(tokens)
    if max_positions is not None:
        seq_len = min(seq_len, max_positions)

    logits = get_logits_for_sequence(model, tokens[:seq_len], device)

    comparisons = []
    correct_count = 0

    for pos in range(seq_len - 1):
        actual_next = int(tokens[pos + 1])
        predicted_next = int(np.argmax(logits[pos]))
        correct = actual_next == predicted_next

        comparisons.append((pos, actual_next, predicted_next, correct))
        if correct:
            correct_count += 1

    accuracy = correct_count / len(comparisons) if comparisons else 0.0

    return comparisons, accuracy


# ============================================================================
# Data Loading
# ============================================================================


def load_sequences_from_arrayrecord(
    arrayrecord_path: str, num_sequences: int, seed: int = 42
) -> List[np.ndarray]:
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
    row_indices = random.sample(
        range(len(datasource)), min(num_sequences * 2, len(datasource))
    )

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
    data: str,
    num_sequences: int = 5,
    max_length: int = 100,
    seed: int = 42,
):
    """
    Main evaluation logic.

    Args:
        checkpoint: Path to .pt checkpoint file
        data: Path to evaluation data (arrayrecord file)
        num_sequences: Number of sequences to evaluate
        max_length: Maximum sequence length to evaluate
        seed: Random seed
    """
    print(f"\n{'='*80}")
    print("NEXT-TOKEN PREDICTION EVALUATION")
    print(f"{'='*80}\n")

    print(f"Loading data from {data}...")
    sequences = load_sequences_from_arrayrecord(data, num_sequences, seed)
    print(f"Loaded {len(sequences)} sequences")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model from {checkpoint} (device={device})...")
    model, model_cfg = load_model(checkpoint, device=device)
    print(f"Model loaded: {model.num_params:,} params\n")

    all_accuracies = []

    for seq_idx, tokens in enumerate(sequences):
        print(f"\n{'='*80}")
        print(f"SEQUENCE {seq_idx + 1}/{len(sequences)}")
        print(f"{'='*80}")
        print(f"Length: {len(tokens)} tokens")

        pretty_tokens = decode_token_stream_pretty(tokens[:max_length])
        print(f"\nSequence: {' '.join(pretty_tokens[:20])}")
        if len(pretty_tokens) > 20:
            print(f"          ... (+{len(pretty_tokens) - 20} more tokens)")

        print(f"\nEvaluating next-token predictions...")
        comparisons, accuracy = evaluate_sequence(
            model, tokens, device, max_positions=max_length
        )
        all_accuracies.append(accuracy)

        print(
            f"\nAccuracy: {accuracy*100:.1f}% ({sum(1 for _, _, _, c in comparisons if c)}/{len(comparisons)} correct)"
        )

        print(f"\nFirst 20 predictions:")
        for pos, actual, predicted, correct in comparisons[:20]:
            print(f"  {format_token_comparison(pos, actual, predicted, correct)}")

        if len(comparisons) > 20:
            print(f"\n  ... (+{len(comparisons) - 20} more positions)")

            print(f"\nLast 10 predictions:")
            for pos, actual, predicted, correct in comparisons[-10:]:
                print(
                    f"  {format_token_comparison(pos, actual, predicted, correct)}"
                )

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
        help="Path to .pt checkpoint file",
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
        gpu="A100",
        cpu=4,
        volumes={"/mnt": shared_vol},
        timeout=60 * 60 * 2,
    )
    def eval_on_modal(
        checkpoint_path: str = DEFAULT_CHECKPOINT_MODAL,
        data_file: str = "probe_rows/test.arrayrecord",
        num_sequences: int = 5,
        max_length: int = 100,
        seed: int = 42,
    ):
        """Run eval on Modal with GPU."""
        os.symlink("/mnt/data", f"{WORKDIR}/data")
        os.symlink("/mnt/outputs", f"{WORKDIR}/outputs")

        if not checkpoint_path.startswith("/"):
            checkpoint_path = f"{WORKDIR}/{checkpoint_path}"
        data_path = f"/mnt/data/{data_file}"

        sys.argv = [
            "eval_next_token_predictions.py",
            "--checkpoint", checkpoint_path,
            "--data", data_path,
            "--num-sequences", str(num_sequences),
            "--max-length", str(max_length),
            "--seed", str(seed),
        ]
        os.chdir(WORKDIR)
        main()
