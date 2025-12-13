#!/usr/bin/env python3
"""
Shard the single training_data.parquet file into multiple smaller files.

This script:
1. Reads data/training_data.parquet (100M rows)
2. Splits into train/val/test (80/10/10 split)
3. Shards each split into multiple files for better I/O parallelism
4. Stratifies by time to ensure balanced temporal distribution

Output structure:
    data/sharded/
        train/shard_0000.parquet, shard_0001.parquet, ...
        val/shard_0000.parquet, ...
        test/shard_0000.parquet, ...
"""

import argparse
import os
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc


def shard_parquet(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shards_per_split: dict = None,
):
    """
    Shard a large Parquet file into train/val/test splits.

    Args:
        input_file: Path to input Parquet file
        output_dir: Output directory for sharded files
        train_ratio: Fraction of data for training (default 0.8)
        val_ratio: Fraction of data for validation (default 0.1)
        test_ratio: Fraction of data for testing (default 0.1)
        shards_per_split: Dict with shard counts, e.g., {'train': 200, 'val': 25, 'test': 25}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    if shards_per_split is None:
        shards_per_split = {
            'train': 200,  # ~400k rows/shard
            'val': 25,     # ~400k rows/shard
            'test': 25,    # ~400k rows/shard
        }

    print(f"Reading input file: {input_file}")
    table = pq.read_table(input_file)
    total_rows = len(table)
    print(f"Total rows: {total_rows:,}")

    # Sort by event_time to enable stratified splitting
    print("Sorting by event_time...")
    sorted_indices = pc.sort_indices(table, sort_keys=[("event_time", "ascending")])
    table = pc.take(table, sorted_indices)

    # Calculate split boundaries
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    splits = {
        'train': table.slice(0, train_end),
        'val': table.slice(train_end, val_end - train_end),
        'test': table.slice(val_end, total_rows - val_end),
    }

    # Create output directories
    output_path = Path(output_dir)
    for split_name in splits.keys():
        (output_path / split_name).mkdir(parents=True, exist_ok=True)

    # Shard each split
    for split_name, split_table in splits.items():
        split_rows = len(split_table)
        num_shards = shards_per_split[split_name]
        rows_per_shard = split_rows // num_shards

        print(f"\nSharding {split_name} split: {split_rows:,} rows into {num_shards} shards")
        print(f"  ~{rows_per_shard:,} rows per shard")

        for shard_idx in range(num_shards):
            start_idx = shard_idx * rows_per_shard
            # Last shard gets any remaining rows
            if shard_idx == num_shards - 1:
                end_idx = split_rows
            else:
                end_idx = start_idx + rows_per_shard

            shard_table = split_table.slice(start_idx, end_idx - start_idx)

            # Write shard
            output_file = output_path / split_name / f"shard_{shard_idx:04d}.parquet"
            pq.write_table(shard_table, output_file, compression='snappy')

            if (shard_idx + 1) % 20 == 0 or shard_idx == num_shards - 1:
                print(f"  Written {shard_idx + 1}/{num_shards} shards")

    print(f"\nSharding complete. Output directory: {output_dir}")
    print("\nSummary:")
    for split_name in splits.keys():
        split_dir = output_path / split_name
        num_files = len(list(split_dir.glob("*.parquet")))
        print(f"  {split_name}: {num_files} shards in {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Shard Parquet file into train/val/test splits")
    parser.add_argument(
        "--input",
        type=str,
        default="data/training_data.parquet",
        help="Input Parquet file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sharded",
        help="Output directory for sharded files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=200,
        help="Number of training shards (default: 200)"
    )
    parser.add_argument(
        "--val-shards",
        type=int,
        default=25,
        help="Number of validation shards (default: 25)"
    )
    parser.add_argument(
        "--test-shards",
        type=int,
        default=25,
        help="Number of test shards (default: 25)"
    )

    args = parser.parse_args()

    shards_per_split = {
        'train': args.train_shards,
        'val': args.val_shards,
        'test': args.test_shards,
    }

    shard_parquet(
        input_file=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shards_per_split=shards_per_split,
    )


if __name__ == "__main__":
    main()
