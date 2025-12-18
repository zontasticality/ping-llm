#!/usr/bin/env python3
"""
Test PLAN_3 data loading pipeline.

This script:
1. Loads a probe row dataset
2. Samples a few batches
3. Validates output shapes and segmentation masks
4. Prints statistics about K contexts and timestamp modes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.MaxText.input_pipeline._probe_chunk_datasource import (
    ProbeRowDataSource,
    ProbeRowSampler,
)
import numpy as np
from collections import Counter


def test_data_source(arrayrecord_path: str, num_samples: int = 3):
    """Test ProbeRowDataSource."""
    print("=" * 80)
    print("Testing ProbeRowDataSource")
    print("=" * 80)

    source = ProbeRowDataSource(arrayrecord_path)
    print(f"\nTotal rows: {len(source):,}")

    # Sample a few rows
    for i in range(min(num_samples, len(source))):
        print(f"\n--- Row {i} ---")
        row = source[i]
        print(f"  src_id: {row['src_id']}")
        print(f"  n_measurements: {row['n_measurements']}")
        print(f"  measurements (first 2): {row['measurements'][:2]}")
        print(f"  time_span: {row['metadata']['time_span_seconds']:.1f}s")


def test_sampler(arrayrecord_path: str, num_batches: int = 3):
    """Test ProbeRowSampler."""
    print("\n" + "=" * 80)
    print("Testing ProbeRowSampler")
    print("=" * 80)

    source = ProbeRowDataSource(arrayrecord_path)
    sampler = ProbeRowSampler(crop_size=1024, seed=42)

    # Test K contexts calculation
    K_values = []
    padding_percentages = []
    mode_counter = Counter()

    print("\nSampling contexts from first 10 rows...")
    for i in range(min(10, len(source))):
        row = source[i]
        contexts = sampler.map(row)

        n_meas = row['n_measurements']
        K = len(contexts)
        K_values.append(K)

        print(f"\nRow {i}: n_measurements={n_meas}, K={K}")

        # Check first context
        ctx = contexts[0]
        segmentation = ctx['inputs_segmentation']
        real_tokens = np.sum(segmentation)
        padding = 1024 - real_tokens
        padding_pct = (padding / 1024) * 100
        padding_percentages.append(padding_pct)

        print(f"  First context: real_tokens={real_tokens}, padding={padding} ({padding_pct:.1f}%)")

    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"\nK contexts per row:")
    print(f"  Min: {min(K_values)}")
    print(f"  Max: {max(K_values)}")
    print(f"  Mean: {np.mean(K_values):.1f}")
    print(f"  Total contexts from 10 rows: {sum(K_values)}")

    print(f"\nPadding percentage:")
    print(f"  Min: {min(padding_percentages):.1f}%")
    print(f"  Max: {max(padding_percentages):.1f}%")
    print(f"  Mean: {np.mean(padding_percentages):.1f}%")


def test_full_pipeline(arrayrecord_path: str, num_batches: int = 2):
    """Test full pipeline with batching."""
    print("\n" + "=" * 80)
    print("Testing Full Pipeline with Batching")
    print("=" * 80)

    import grain.python as grain
    from src.MaxText.input_pipeline.probe_chunk_pipeline import build_probe_chunk_dataset

    dataset = build_probe_chunk_dataset(
        arrayrecord_path=arrayrecord_path,
        batch_size=4,
        crop_size=1024,
        shuffle=False,
        num_workers=0,  # Disable threading for testing
    )

    print(f"\nSampling {num_batches} batches...")
    for batch_idx, batch in enumerate(dataset):
        if batch_idx >= num_batches:
            break

        print(f"\n--- Batch {batch_idx} ---")
        print(f"  inputs shape: {batch['inputs'].shape}")
        print(f"  targets shape: {batch['targets'].shape}")
        print(f"  segmentation shape: {batch['inputs_segmentation'].shape}")

        # Check that inputs == targets (autoregressive)
        assert np.array_equal(batch['inputs'], batch['targets']), "inputs != targets"

        # Check padding statistics
        for i in range(batch['inputs'].shape[0]):
            real_tokens = np.sum(batch['inputs_segmentation'][i])
            padding = 1024 - real_tokens
            padding_pct = (padding / 1024) * 100
            print(f"  Sample {i}: real_tokens={real_tokens}, padding={padding} ({padding_pct:.1f}%)")

    print("\n✓ All tests passed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test PLAN_3 data loading pipeline")
    parser.add_argument(
        "--arrayrecord",
        default="data/probe_rows_test/train.arrayrecord",
        help="Path to ArrayRecord file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to test"
    )

    args = parser.parse_args()

    if not Path(args.arrayrecord).exists():
        print(f"Error: ArrayRecord file not found: {args.arrayrecord}")
        print("Run preprocessing first:")
        print(f"  python scripts/data/create_probe_rows.py --input <parquet> --output data/probe_rows_test")
        return 1

    # Run tests
    test_data_source(args.arrayrecord, num_samples=args.num_samples)
    test_sampler(args.arrayrecord)
    test_full_pipeline(args.arrayrecord)

    return 0


if __name__ == "__main__":
    exit(main())
