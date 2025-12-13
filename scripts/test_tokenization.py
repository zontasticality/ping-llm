#!/usr/bin/env python3
"""
Simple tokenization test using DuckDB (already available in the project).

This script validates tokenization works correctly without requiring pyarrow.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for tokenization import
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from tokenization import encode_measurement, validate_tokens, VOCAB_SIZE


def test_tokenization(input_file: str = "data/training_data.parquet", num_samples: int = 3):
    """Test tokenization on sample data using DuckDB."""

    print("=" * 80)
    print("TOKENIZATION TEST")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Testing {num_samples} samples\n")

    # Connect to DuckDB and read sample data
    con = duckdb.connect()
    query = f"""
        SELECT *
        FROM '{input_file}'
        LIMIT {num_samples}
    """

    df = con.sql(query).df()
    print(f"Loaded {len(df)} rows\n")

    # Test tokenization
    print("-" * 80)
    print("ENCODING TESTS")
    print("-" * 80)

    token_lengths = []
    ipv4_count = 0
    ipv6_count = 0

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        # Encode measurement
        tokens = encode_measurement(row_dict)

        # Validate tokens
        is_valid = validate_tokens(tokens)
        if not is_valid:
            print(f"❌ ERROR: Invalid tokens in row {i}")
            print(f"   Tokens: {tokens}")
            sys.exit(1)

        # Track statistics
        token_lengths.append(len(tokens))
        if row_dict['ip_version'] == 4:
            ipv4_count += 1
        else:
            ipv6_count += 1

        # Print sample
        print(f"\nSample {i + 1}:")
        print(f"  msm_id: {row_dict['msm_id']}")
        print(f"  event_time: {row_dict['event_time']}")
        print(f"  src_addr: {row_dict['src_addr']} (IPv{row_dict['ip_version']})")
        print(f"  dst_addr: {row_dict['dst_addr']}")
        print(f"  rtt: {row_dict['rtt']:.2f} ms")
        print(f"  size: {row_dict['size']} bytes")
        print(f"  packet_error_count: {row_dict['packet_error_count']}")
        print(f"  Token count: {len(tokens)}")
        print(f"  First 30 tokens: {tokens[:30]}")
        print(f"  Token range: [{min(tokens)}, {max(tokens)}]")
        print(f"  All tokens valid: ✓")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Samples tested: {len(token_lengths)}")
    print(f"IPv4 samples: {ipv4_count}")
    print(f"IPv6 samples: {ipv6_count}")
    print(f"Token lengths: min={min(token_lengths)}, max={max(token_lengths)}, "
          f"avg={sum(token_lengths) / len(token_lengths):.1f}")
    print(f"Expected range for IPv4: ~42 tokens")
    print(f"Expected range for IPv6: ~66 tokens")
    print(f"All tokens within valid range [0, {VOCAB_SIZE}): ✓")

    # Test deterministic shuffling
    print("\n" + "-" * 80)
    print("DETERMINISTIC SHUFFLING TEST")
    print("-" * 80)
    row_dict = df.iloc[0].to_dict()
    tokens1 = encode_measurement(row_dict)
    tokens2 = encode_measurement(row_dict)
    if tokens1 == tokens2:
        print("✓ Same row produces identical token sequences (deterministic)")
    else:
        print("❌ ERROR: Same row produces different tokens (non-deterministic)")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_tokenization()
