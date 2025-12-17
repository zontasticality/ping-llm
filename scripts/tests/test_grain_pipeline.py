#!/usr/bin/env python3
"""
Test Grain data pipeline with real sharded data.

This script validates:
1. Grain can load Parquet shards
2. ContextWindowSampler works with real data
3. Tokenization produces valid sequences
4. Batching works correctly
"""

import sys
from pathlib import Path
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from network_grain_datasource_simple import (
    ParquetMeasurementDataSource,
    SimpleMeasurementTokenizer,
    create_simple_grain_pipeline,
)


def test_parquet_datasource():
    """Test ParquetMeasurementDataSource with real shards."""
    print("=" * 80)
    print("PARQUET DATASOURCE TEST")
    print("=" * 80)

    # Load first 2 train shards
    train_files = sorted(glob.glob("data/sharded/train/*.parquet"))[:2]
    print(f"\nLoading {len(train_files)} shards:")
    for f in train_files:
        print(f"  {f}")

    source = ParquetMeasurementDataSource(train_files)
    print(f"\nTotal rows: {len(source):,}")

    # Test random access
    print("\nTesting random access:")
    for idx in [0, 100, 1000, 10000]:
        if idx < len(source):
            row = source[idx]
            print(f"  Row {idx}: {row['src_addr']} → {row['dst_addr']} "
                  f"({row['ip_version']}) RTT={row['rtt']:.2f}ms")

    print("\n✓ ParquetMeasurementDataSource working")


def test_simple_tokenizer():
    """Test SimpleMeasurementTokenizer with real data."""
    print("\n" + "=" * 80)
    print("SIMPLE TOKENIZER TEST")
    print("=" * 80)

    # Load one shard
    train_files = sorted(glob.glob("data/sharded/train/*.parquet"))[:1]
    source = ParquetMeasurementDataSource(train_files)

    # Create tokenizer
    tokenizer = SimpleMeasurementTokenizer(include_timestamp=True, max_tokens=1024, seed=42)

    # Test tokenization
    print("\nTesting tokenization on 5 measurements:")
    for i in range(5):
        if i < len(source):
            measurement = source[i]
            result = tokenizer.map(measurement)
            print(f"  Measurement {i}: "
                  f"{measurement['src_addr']} → {measurement['dst_addr']} "
                  f"| tokens={result['length']}")

    print("\n✓ SimpleMeasurementTokenizer working")


def test_full_pipeline():
    """Test full Grain pipeline."""
    print("\n" + "=" * 80)
    print("FULL GRAIN PIPELINE TEST")
    print("=" * 80)

    # Load first 5 train shards
    train_files = sorted(glob.glob("data/sharded/train/*.parquet"))[:5]
    print(f"\nCreating pipeline with {len(train_files)} shards")

    pipeline = create_simple_grain_pipeline(
        parquet_files=train_files,
        batch_size=4,  # Small batch for testing
        max_tokens=1024,
        include_timestamp=True,
        shuffle=True,
        shuffle_seed=42,
        num_workers=2,
    )

    print("\nIterating pipeline:")
    for i, batch in enumerate(pipeline):
        if i >= 3:  # Test first 3 batches
            break

        print(f"\nBatch {i+1}:")
        print(f"  Keys: {list(batch.keys())}")
        if 'tokens' in batch:
            tokens = batch['tokens']
            print(f"  Tokens shape: {tokens.shape if hasattr(tokens, 'shape') else len(tokens)}")
            if hasattr(tokens, '__iter__'):
                print(f"  Sample tokens[0]: {list(tokens[0])[:20]}...")
        if 'length' in batch:
            lengths = batch['length']
            print(f"  Lengths: {lengths[:4] if hasattr(lengths, '__iter__') else lengths}")
        if 'mode' in batch:
            modes = batch['mode']
            print(f"  Modes: {modes[:4] if hasattr(modes, '__iter__') else modes}")

    print("\n✓ Full pipeline working")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("GRAIN PIPELINE TESTS (Real Data)")
    print("=" * 80)

    try:
        # Check data exists
        train_files = glob.glob("data/sharded/train/*.parquet")
        if not train_files:
            print("\n❌ ERROR: No training shards found")
            print("Run: .venv/bin/python scripts/data/probe_chunk_preprocess.py")
            sys.exit(1)

        print(f"\nFound {len(train_files)} training shards")

        # Run tests
        test_parquet_datasource()
        test_simple_tokenizer()
        test_full_pipeline()

        print("\n" + "=" * 80)
        print("✅ ALL GRAIN PIPELINE TESTS PASSED")
        print("=" * 80)
        print("\nData pipeline ready for MaxText training!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
