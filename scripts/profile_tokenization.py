#!/usr/bin/env python3
"""
Profile tokenization throughput.

This script measures how fast we can tokenize measurements, which helps
determine if we need pre-tokenization or if on-the-fly is sufficient.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq
from tokenization import encode_measurement


def profile_throughput(input_file: str, num_samples: int = 10000):
    """Profile tokenization throughput."""
    print("=" * 80)
    print("TOKENIZATION THROUGHPUT PROFILING")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Profiling {num_samples:,} samples\n")

    # Read data
    print("Reading Parquet...")
    start = time.time()
    table = pq.read_table(input_file)
    df = table.to_pandas().head(num_samples)
    read_time = time.time() - start
    print(f"Read time: {read_time:.2f}s ({num_samples / read_time:,.0f} rows/sec)\n")

    # Tokenize
    print("Tokenizing measurements...")
    start = time.time()

    total_tokens = 0
    ipv4_count = 0
    ipv6_count = 0

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        tokens = encode_measurement(row_dict)
        total_tokens += len(tokens)

        if row_dict['ip_version'] == 4:
            ipv4_count += 1
        else:
            ipv6_count += 1

        # Progress indicator
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            print(f"  Processed {idx + 1:,} / {num_samples:,} ({rate:,.0f} rows/sec)", end='\r')

    tokenize_time = time.time() - start
    print()  # New line after progress

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Samples processed: {num_samples:,}")
    print(f"  IPv4: {ipv4_count:,} ({100 * ipv4_count / num_samples:.1f}%)")
    print(f"  IPv6: {ipv6_count:,} ({100 * ipv6_count / num_samples:.1f}%)")
    print(f"\nTiming:")
    print(f"  Tokenization time: {tokenize_time:.2f}s")
    print(f"  Throughput: {num_samples / tokenize_time:,.0f} rows/sec")
    print(f"  Token throughput: {total_tokens / tokenize_time:,.0f} tokens/sec")
    print(f"\nTokens:")
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Average tokens/row: {total_tokens / num_samples:.1f}")
    print(f"\nProjections for full dataset (100M rows):")
    full_dataset_time = 100_000_000 / (num_samples / tokenize_time)
    print(f"  Estimated time: {full_dataset_time / 3600:.1f} hours")
    print(f"  Estimated total tokens: {(total_tokens / num_samples) * 100_000_000 / 1e9:.1f}B tokens")


def main():
    parser = argparse.ArgumentParser(description="Profile tokenization throughput")
    parser.add_argument(
        "--input",
        type=str,
        default="data/training_data.parquet",
        help="Input Parquet file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples to profile (default: 10000)"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    profile_throughput(args.input, args.samples)


if __name__ == "__main__":
    main()
