#!/usr/bin/env python3
"""
Local Grain smoke test for Parquet + tokenization integration.

This script tests the complete data loading pipeline:
1. Load Parquet file using Grain (or fallback to direct Parquet reading)
2. Apply tokenization transform
3. Validate token outputs

Usage:
    python scripts/local_grain_smoke.py --input data/training_data.parquet --samples 5

Note: This is a smoke test. Full Grain integration with MaxText will happen in Phase 2.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for tokenization import
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq
from tokenization import encode_measurement, validate_tokens, VOCAB_SIZE


def test_tokenization_direct(input_file: str, num_samples: int = 5):
    """
    Test tokenization by directly reading Parquet and applying encode_measurement.

    This is a fallback approach that doesn't require Grain to be installed.
    It validates that tokenization works correctly on actual data.

    Args:
        input_file: Path to Parquet file
        num_samples: Number of samples to test
    """
    print("=" * 80)
    print("GRAIN SMOKE TEST (Direct Parquet reading)")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Testing {num_samples} samples\n")

    # Read Parquet file
    print("Reading Parquet file...")
    table = pq.read_table(input_file)
    total_rows = len(table)
    print(f"Total rows in dataset: {total_rows:,}\n")

    # Convert to pandas for easier row access
    df = table.to_pandas()

    # Test tokenization on first N samples
    print("-" * 80)
    print("TOKENIZATION TEST")
    print("-" * 80)

    token_lengths = []
    ipv4_count = 0
    ipv6_count = 0

    for i in range(min(num_samples, total_rows)):
        row = df.iloc[i].to_dict()

        # Encode measurement
        tokens = encode_measurement(row)

        # Validate tokens
        is_valid = validate_tokens(tokens)
        assert is_valid, f"Invalid tokens found in row {i}"

        # Track statistics
        token_lengths.append(len(tokens))
        if row['ip_version'] == 4:
            ipv4_count += 1
        else:
            ipv6_count += 1

        # Print sample
        print(f"\nSample {i + 1}:")
        print(f"  msm_id: {row['msm_id']}")
        print(f"  event_time: {row['event_time']}")
        print(f"  src_addr: {row['src_addr']} (IPv{row['ip_version']})")
        print(f"  dst_addr: {row['dst_addr']}")
        print(f"  rtt: {row['rtt']:.2f} ms")
        print(f"  size: {row['size']} bytes")
        print(f"  packet_error_count: {row['packet_error_count']}")
        print(f"  Token count: {len(tokens)}")
        print(f"  First 20 tokens: {tokens[:20]}")
        print(f"  Token range: [{min(tokens)}, {max(tokens)}]")
        print(f"  All tokens valid: {is_valid}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Samples tested: {len(token_lengths)}")
    print(f"IPv4 samples: {ipv4_count}")
    print(f"IPv6 samples: {ipv6_count}")
    print(f"Token lengths: min={min(token_lengths)}, max={max(token_lengths)}, "
          f"avg={sum(token_lengths) / len(token_lengths):.1f}")
    print(f"All tokens within valid range [0, {VOCAB_SIZE}): ✓")
    print("\n✅ Tokenization smoke test PASSED")


def test_with_grain(input_file: str, num_samples: int = 5):
    """
    Test tokenization with Grain library (if available).

    This tests the full Grain + tokenization pipeline that will be used
    in MaxText training.

    Args:
        input_file: Path to Parquet file
        num_samples: Number of samples to test
    """
    try:
        import grain.python as grain
    except ImportError:
        print("⚠️  Grain library not installed. Falling back to direct Parquet reading.")
        print("   To install Grain: pip install grain-python")
        print()
        test_tokenization_direct(input_file, num_samples)
        return

    print("=" * 80)
    print("GRAIN SMOKE TEST (Using Grain library)")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Testing {num_samples} samples\n")

    # Create Grain dataset
    # Note: This is a simplified version. Full MaxText integration will use
    # grain.experimental.ParquetIterDataset or similar
    print("⚠️  Full Grain integration not yet implemented.")
    print("   This requires MaxText-specific Grain setup.")
    print("   Falling back to direct Parquet reading for now.\n")

    test_tokenization_direct(input_file, num_samples)


def main():
    parser = argparse.ArgumentParser(description="Grain smoke test for tokenization")
    parser.add_argument(
        "--input",
        type=str,
        default="data/training_data.parquet",
        help="Input Parquet file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--use-grain",
        action="store_true",
        help="Try to use Grain library (if installed)"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.use_grain:
        test_with_grain(args.input, args.samples)
    else:
        test_tokenization_direct(args.input, args.samples)


if __name__ == "__main__":
    main()
