#!/usr/bin/env python3
"""
Unified eval runner for ping-llm.

Usage:
    python -m ping_llm.eval.run_all \
        --checkpoint path/to/latest.pt \
        --test-data path/to/test.arrayrecord \
        --tests loss_breakdown,history_ping,baselines
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ping_llm.inference import load_model


def load_test_sequences(arrayrecord_path, num_sequences=200, seed=42):
    """Load tokenized sequences from arrayrecord test data."""
    from ping_llm.data.datasource import ProbeRowDataSource, ProbeRowSampler
    import random

    print(f"Loading test data from {arrayrecord_path}...")
    datasource = ProbeRowDataSource(arrayrecord_path)
    sampler = ProbeRowSampler(
        crop_size=1024,
        avg_tokens_per_measurement=30,
        max_contexts_per_row=1,
        seed=seed,
    )

    sequences = []
    random.seed(seed)
    n_rows = len(datasource)
    row_indices = random.sample(range(n_rows), min(max(num_sequences * 2, n_rows), n_rows))

    for idx in row_indices:
        if len(sequences) >= num_sequences:
            break
        row = datasource[idx]
        contexts = sampler.flat_map(row)
        if contexts:
            tokens = contexts[0]["inputs"]
            seg = contexts[0]["inputs_segmentation"]
            valid_len = int(np.sum(seg))
            if valid_len > 20:
                sequences.append(tokens[:valid_len])

    print(f"Loaded {len(sequences)} sequences (avg length: {np.mean([len(s) for s in sequences]):.0f} tokens)")
    return sequences


def run_loss_breakdown(model, sequences, device, args):
    """Run Test 1: Loss breakdown by token type."""
    from ping_llm.eval.loss_breakdown import eval_loss_breakdown, print_loss_breakdown

    print("\n" + "#" * 80)
    print("# TEST 1: LOSS BREAKDOWN BY TOKEN TYPE")
    print("#" * 80)

    results = eval_loss_breakdown(
        model, sequences, device=device,
        max_sequences=args.loss_sequences,
    )
    print_loss_breakdown(results)
    return results


def run_history_ping(model, device, args):
    """Run Test 2: History-conditioned live ping."""
    from ping_llm.eval.history_ping import eval_history_ping, print_history_ping

    print("\n" + "#" * 80)
    print("# TEST 2: HISTORY-CONDITIONED LIVE PING")
    print("#" * 80)

    results = eval_history_ping(
        model, device=device,
        pings_per_target=args.pings_per_target,
        model_samples=args.model_samples,
        temperature=args.temperature,
        history_sizes=tuple(args.history_sizes),
    )
    print_history_ping(results)
    return results


def run_baselines(model, sequences, device, args):
    """Run Test 3: Baseline comparison."""
    from ping_llm.eval.baselines import eval_baselines, print_baselines

    print("\n" + "#" * 80)
    print("# TEST 3: BASELINE COMPARISON")
    print("#" * 80)

    results = eval_baselines(
        model, sequences, device=device,
        max_sequences=args.baseline_sequences,
    )
    print_baselines(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="ping-llm evaluation suite")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test-data", required=True, help="Path to test.arrayrecord")
    parser.add_argument("--output-dir", default=None, help="Directory to save JSON results")
    parser.add_argument(
        "--tests", default="loss_breakdown,history_ping,baselines",
        help="Comma-separated list of tests to run",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Test 1 options
    parser.add_argument("--loss-sequences", type=int, default=200,
                        help="Number of sequences for loss breakdown")

    # Test 2 options
    parser.add_argument("--pings-per-target", type=int, default=50)
    parser.add_argument("--model-samples", type=int, default=200,
                        help="RTT samples per model query for history ping")
    parser.add_argument("--history-sizes", type=int, nargs="+", default=[0, 1, 2, 3, 5])

    # Test 3 options
    parser.add_argument("--baseline-sequences", type=int, default=100,
                        help="Number of sequences for baseline comparison")

    args = parser.parse_args()
    selected_tests = set(args.tests.split(","))

    print("=" * 80)
    print("ping-llm Evaluation Suite")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Tests: {', '.join(sorted(selected_tests))}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model, model_cfg = load_model(args.checkpoint, device=device)
    print(f"Model loaded: {model.num_params:,} params ({time.time()-t0:.1f}s)")

    # Load sequences if needed
    sequences = None
    needs_sequences = selected_tests & {"loss_breakdown", "baselines"}
    if needs_sequences:
        max_needed = max(
            args.loss_sequences if "loss_breakdown" in selected_tests else 0,
            args.baseline_sequences if "baselines" in selected_tests else 0,
        )
        sequences = load_test_sequences(
            args.test_data, num_sequences=max_needed, seed=args.seed,
        )

    all_results = {}
    total_start = time.time()

    # Run tests
    if "loss_breakdown" in selected_tests:
        t0 = time.time()
        all_results["loss_breakdown"] = run_loss_breakdown(model, sequences, device, args)
        print(f"  (completed in {time.time()-t0:.1f}s)")

    if "history_ping" in selected_tests:
        t0 = time.time()
        all_results["history_ping"] = run_history_ping(model, device, args)
        print(f"  (completed in {time.time()-t0:.1f}s)")

    if "baselines" in selected_tests:
        t0 = time.time()
        all_results["baselines"] = run_baselines(model, sequences, device, args)
        print(f"  (completed in {time.time()-t0:.1f}s)")

    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"All tests completed in {total_time:.0f}s")
    print(f"{'='*80}")

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "eval_results.json"

        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
