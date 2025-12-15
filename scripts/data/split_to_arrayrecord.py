#!/usr/bin/env python3
"""
Simple Parquet to ArrayRecord converter with train/test split.

Reads a single Parquet file and creates train.arrayrecord and test.arrayrecord.
Memory-efficient: streams batches, never loads full dataset.

Usage:
    python split_to_arrayrecord.py \
      --input data/training_data.parquet \
      --output-dir data/arrayrecord \
      --test-split 0.1 \
      --batch-size 50000
"""

import argparse
import sys
from pathlib import Path
import time
import random

import pyarrow.parquet as pq
import numpy as np

try:
    from array_record.python.array_record_module import ArrayRecordWriter
    ARRAY_RECORD_AVAILABLE = True
except ImportError:
    print("ERROR: array_record not installed!")
    print("Install with: pip install array_record")
    ARRAY_RECORD_AVAILABLE = False


def serialize_batch(table) -> list[bytes]:
    """
    Serialize a batch of rows efficiently.

    Format: [src_len:2][src][dst_len:2][dst][ip:1][rtt:4][time:8]
    """
    num_rows = len(table)

    # Convert to Python/numpy for fast access
    src_addrs = table['src_addr'].to_pylist()
    dst_addrs = table['dst_addr'].to_pylist()
    ip_versions = table['ip_version'].to_numpy()
    rtts = table['rtt'].to_numpy(zero_copy_only=False).astype(np.float32)

    # Convert datetime64[us] to Unix timestamp (seconds)
    event_times_us = table['event_time'].to_numpy()  # datetime64[us]
    event_times_sec = (event_times_us.astype('int64') // 1_000_000).astype('int64')

    serialized = []
    for i in range(num_rows):
        # Handle None values in string fields (use empty string)
        src = src_addrs[i] if src_addrs[i] is not None else ""
        dst = dst_addrs[i] if dst_addrs[i] is not None else ""

        src_bytes = src.encode('utf-8')
        dst_bytes = dst.encode('utf-8')

        parts = [
            len(src_bytes).to_bytes(2, 'little'),
            src_bytes,
            len(dst_bytes).to_bytes(2, 'little'),
            dst_bytes,
            ip_versions[i].tobytes(),
            rtts[i].tobytes(),
            event_times_sec[i].tobytes(),
        ]

        serialized.append(b''.join(parts))

    return serialized


def split_parquet_to_arrayrecord(
    input_file: str,
    output_dir: str,
    test_split: float = 0.1,
    batch_size: int = 50_000,
    seed: int = 42,
):
    """
    Convert single Parquet file to train/test ArrayRecord files.

    Strategy:
    - Read Parquet in batches (streaming, low memory)
    - Randomly assign rows to train/test based on test_split
    - Write to two separate ArrayRecord files
    - Memory usage: O(batch_size), not O(total_rows)

    Args:
        input_file: Path to input Parquet file
        output_dir: Output directory for ArrayRecord files
        test_split: Fraction of data for test (e.g., 0.1 = 10%)
        batch_size: Rows to process per batch
        seed: Random seed for reproducible splits
    """
    if not ARRAY_RECORD_AVAILABLE:
        raise ImportError("array_record package not available")

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print("="*80)
    print("PARQUET TO ARRAYRECORD CONVERTER (with train/test split)")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Test split: {test_split*100:.1f}%")
    print(f"Batch size: {batch_size:,} rows")
    print()

    # Get metadata
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    total_row_groups = parquet_file.num_row_groups

    print(f"Input file statistics:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Row groups: {total_row_groups}")
    print(f"  Estimated train rows: {int(total_rows * (1-test_split)):,}")
    print(f"  Estimated test rows: {int(total_rows * test_split):,}")
    print()

    # Create output files
    train_file = output_path / "train.arrayrecord"
    test_file = output_path / "test.arrayrecord"

    train_writer = ArrayRecordWriter(str(train_file), 'group_size:1')
    test_writer = ArrayRecordWriter(str(test_file), 'group_size:1')

    # Set random seed for reproducible splits
    rng = random.Random(seed)

    start_time = time.time()
    train_count = 0
    test_count = 0
    total_processed = 0

    print("Processing batches...")

    # Stream through Parquet file in batches
    for batch_idx, batch_table in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        # Serialize batch
        serialized = serialize_batch(batch_table)

        # Randomly split into train/test
        for data in serialized:
            if rng.random() < test_split:
                test_writer.write(data)
                test_count += 1
            else:
                train_writer.write(data)
                train_count += 1

        total_processed += len(batch_table)

        # Progress update
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            progress = total_processed / total_rows * 100 if total_rows > 0 else 0
            print(f"  Batch {batch_idx+1}: {total_processed:,}/{total_rows:,} rows "
                  f"({progress:.1f}%, {rate:,.0f} rows/sec)")

    # Close writers
    train_writer.close()
    test_writer.close()

    elapsed = time.time() - start_time

    # Final stats
    train_size_mb = train_file.stat().st_size / (1024**2)
    test_size_mb = test_file.stat().st_size / (1024**2)
    total_size_gb = (train_size_mb + test_size_mb) / 1024

    print()
    print("="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Total rows processed: {total_processed:,}")
    print(f"Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Throughput: {total_processed/elapsed:,.0f} rows/second")
    print()
    print(f"Train split:")
    print(f"  File: {train_file}")
    print(f"  Rows: {train_count:,} ({train_count/total_processed*100:.1f}%)")
    print(f"  Size: {train_size_mb:.2f} MB")
    print()
    print(f"Test split:")
    print(f"  File: {test_file}")
    print(f"  Rows: {test_count:,} ({test_count/total_processed*100:.1f}%)")
    print(f"  Size: {test_size_mb:.2f} MB")
    print()
    print(f"Total size: {total_size_gb:.2f} GB")
    print("="*80)

    return {
        'train_rows': train_count,
        'test_rows': test_count,
        'train_file': str(train_file),
        'test_file': str(test_file),
        'elapsed_seconds': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet to train/test ArrayRecord splits"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Parquet file (e.g., data/training_data.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/arrayrecord",
        help="Output directory (default: data/arrayrecord)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for test split (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per batch (affects memory usage, default: 50,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )

    args = parser.parse_args()

    if not ARRAY_RECORD_AVAILABLE:
        print("\n" + "="*80)
        print("ERROR: array_record not installed!")
        print("="*80)
        print("\nInstall with: pip install array_record")
        print("="*80)
        sys.exit(1)

    try:
        stats = split_parquet_to_arrayrecord(
            input_file=args.input,
            output_dir=args.output_dir,
            test_split=args.test_split,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        print()
        print("Next steps:")
        print()
        print("1. Update your MaxText config:")
        print(f"   grain_file_type: 'arrayrecord'")
        print(f"   grain_train_files: '{stats['train_file']}'")
        print(f"   grain_eval_files: '{stats['test_file']}'")
        print()
        print("2. Grain will stream from these files during training (fast random access!)")
        print()

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
