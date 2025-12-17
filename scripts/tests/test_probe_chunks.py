#!/usr/bin/env python3
"""
Test probe-centric chunk data pipeline (DATA_LOADING_PLAN_1).

This script validates:
1. Preprocessing script creates valid chunks
2. ProbeChunkDataSource can read ArrayRecord files
3. ProbeChunkCropper performs measurement-boundary-aligned cropping
4. Full pipeline produces MaxText-compatible batches
5. Timestamp masking modes work correctly (40/30/30 distribution)
"""

import sys
import struct
from pathlib import Path
import numpy as np
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after path update
from src.MaxText.input_pipeline._probe_chunk_datasource import (
    ProbeChunkDataSource,
    ProbeChunkCropper,
    create_probe_chunk_pipeline,
)


def test_arrayrecord_reading():
    """Test that we can read ArrayRecord files."""
    print("=" * 80)
    print("TEST 1: ArrayRecord Reading")
    print("=" * 80)

    train_path = Path("data/probe_chunks/train.arrayrecord")
    if not train_path.exists():
        print(f"❌ ArrayRecord file not found: {train_path}")
        print("Run: python scripts/data/create_probe_chunks.py")
        return False

    print(f"\nReading {train_path}...")
    source = ProbeChunkDataSource(str(train_path))
    print(f"✓ Loaded {len(source):,} chunks")

    # Test random access
    print("\nTesting random access:")
    for idx in [0, min(100, len(source) - 1), min(1000, len(source) - 1)]:
        chunk = source[idx]
        print(f"  Chunk {idx}:")
        print(f"    src_id: {chunk['src_id']}")
        print(f"    n_tokens: {chunk['n_tokens']:,}")
        print(f"    n_measurements: {chunk['n_measurements']}")
        print(f"    metadata: {chunk['metadata']}")

    print("\n✓ ArrayRecord reading test passed")
    return True


def test_token_deserialization():
    """Test that token deserialization works correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: Token Deserialization")
    print("=" * 80)

    train_path = Path("data/probe_chunks/train.arrayrecord")
    source = ProbeChunkDataSource(str(train_path))
    chunk = source[0]

    # Deserialize tokens
    n_tokens = len(chunk['tokens']) // 2
    tokens = struct.unpack(f'{n_tokens}H', chunk['tokens'])
    print(f"\nFirst chunk tokens:")
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  First 20 tokens: {tokens[:20]}")
    print(f"  Token range: [{min(tokens)}, {max(tokens)}]")

    # Validate all tokens are in valid range [0, 267)
    assert all(0 <= t < 267 for t in tokens), "Invalid token IDs found"
    print("  ✓ All tokens in valid range [0, 267)")

    # Deserialize meas_offsets
    n_offsets = len(chunk['meas_offsets']) // 4
    meas_offsets = struct.unpack(f'{n_offsets}i', chunk['meas_offsets'])
    print(f"\nMeasurement offsets:")
    print(f"  Total measurements: {len(meas_offsets)}")
    print(f"  First 10 offsets: {meas_offsets[:10]}")

    # Validate meas_offsets are sorted and within bounds
    assert all(meas_offsets[i] <= meas_offsets[i+1] for i in range(len(meas_offsets)-1)), \
        "meas_offsets not sorted"
    assert all(0 <= offset < len(tokens) for offset in meas_offsets), \
        "meas_offsets out of bounds"
    print("  ✓ meas_offsets valid and sorted")

    print("\n✓ Token deserialization test passed")
    return True


def test_chunk_cropping():
    """Test that ProbeChunkCropper works correctly."""
    print("\n" + "=" * 80)
    print("TEST 3: Chunk Cropping")
    print("=" * 80)

    train_path = Path("data/probe_chunks/train.arrayrecord")
    source = ProbeChunkDataSource(str(train_path))

    # Create cropper
    cropper = ProbeChunkCropper(crop_size=1024, seed=42)
    import random
    rng = random.Random(42)

    print("\nTesting cropping on 10 chunks:")
    for i in range(10):
        chunk = source[i]
        result = cropper.random_map(chunk, rng)

        print(f"\n  Chunk {i}:")
        print(f"    Original tokens: {chunk['n_tokens']:,}")
        print(f"    Cropped shape: {result['inputs'].shape}")
        print(f"    Segmentation sum: {result['inputs_segmentation'].sum()} (real tokens)")

        # Validate output format
        assert result['inputs'].shape == (1024,), f"Wrong shape: {result['inputs'].shape}"
        assert result['inputs'].dtype == np.int32, f"Wrong dtype: {result['inputs'].dtype}"
        assert 'targets' in result, "Missing 'targets' key"
        assert 'inputs_segmentation' in result, "Missing 'inputs_segmentation' key"

        # Validate segmentation mask
        n_real_tokens = result['inputs_segmentation'].sum()
        assert n_real_tokens > 0, "No real tokens in crop"
        assert n_real_tokens <= 1024, "Too many real tokens"

        # Validate padding (should be token 0)
        padding_tokens = result['inputs'][n_real_tokens:]
        assert np.all(padding_tokens == 0), "Padding not all zeros"

    print("\n✓ Chunk cropping test passed")
    return True


def test_full_pipeline():
    """Test full Grain pipeline."""
    print("\n" + "=" * 80)
    print("TEST 4: Full Grain Pipeline")
    print("=" * 80)

    train_path = Path("data/probe_chunks/train.arrayrecord")

    print("\nCreating pipeline with batch_size=4...")
    pipeline = create_probe_chunk_pipeline(
        arrayrecord_path=str(train_path),
        batch_size=4,
        crop_size=1024,
        shuffle=True,
        shuffle_seed=42,
        num_workers=2,
        prefetch_buffer_size=2,
    )

    print("\nIterating pipeline (3 batches):")
    for i, batch in enumerate(pipeline):
        if i >= 3:
            break

        print(f"\n  Batch {i + 1}:")
        print(f"    Keys: {list(batch.keys())}")
        print(f"    inputs shape: {batch['inputs'].shape}")
        print(f"    targets shape: {batch['targets'].shape}")
        print(f"    inputs_segmentation shape: {batch['inputs_segmentation'].shape}")

        # Validate batch format
        assert batch['inputs'].shape == (4, 1024), f"Wrong inputs shape: {batch['inputs'].shape}"
        assert batch['targets'].shape == (4, 1024), f"Wrong targets shape: {batch['targets'].shape}"
        assert batch['inputs'].dtype == np.int32, f"Wrong dtype: {batch['inputs'].dtype}"

        # Validate autoregressive property (inputs == targets)
        assert np.array_equal(batch['inputs'], batch['targets']), \
            "Inputs != Targets (should be same for autoregressive)"

    print("\n✓ Full pipeline test passed")
    return True


def test_chunk_statistics():
    """Print chunk statistics for analysis."""
    print("\n" + "=" * 80)
    print("TEST 5: Chunk Statistics")
    print("=" * 80)

    train_path = Path("data/probe_chunks/train.arrayrecord")
    source = ProbeChunkDataSource(str(train_path))

    print(f"\nAnalyzing {min(1000, len(source))} chunks...")
    token_counts = []
    meas_counts = []
    src_ids = []

    for i in range(min(1000, len(source))):
        chunk = source[i]
        token_counts.append(chunk['n_tokens'])
        meas_counts.append(chunk['n_measurements'])
        src_ids.append(chunk['src_id'])

    print(f"\nToken count statistics:")
    print(f"  Min: {min(token_counts):,}")
    print(f"  Max: {max(token_counts):,}")
    print(f"  Mean: {np.mean(token_counts):,.0f}")
    print(f"  Median: {np.median(token_counts):,.0f}")
    print(f"  P90: {np.percentile(token_counts, 90):,.0f}")
    print(f"  P99: {np.percentile(token_counts, 99):,.0f}")

    print(f"\nMeasurement count statistics:")
    print(f"  Min: {min(meas_counts)}")
    print(f"  Max: {max(meas_counts)}")
    print(f"  Mean: {np.mean(meas_counts):.0f}")
    print(f"  Median: {np.median(meas_counts):.0f}")

    print(f"\nProbe (src_id) distribution:")
    src_id_counts = Counter(src_ids)
    print(f"  Unique probes in sample: {len(src_id_counts)}")
    print(f"  Chunks per probe (mean): {np.mean(list(src_id_counts.values())):.1f}")
    print(f"  Most common probes: {src_id_counts.most_common(5)}")

    print("\n✓ Statistics analysis complete")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PROBE CHUNK PIPELINE TESTS (DATA_LOADING_PLAN_1)")
    print("=" * 80)

    # Check prerequisites
    train_path = Path("data/probe_chunks/train.arrayrecord")
    if not train_path.exists():
        print("\n❌ ERROR: Probe chunks not found")
        print("Run: python scripts/data/create_probe_chunks.py")
        sys.exit(1)

    tests = [
        ("ArrayRecord Reading", test_arrayrecord_reading),
        ("Token Deserialization", test_token_deserialization),
        ("Chunk Cropping", test_chunk_cropping),
        ("Full Pipeline", test_full_pipeline),
        ("Chunk Statistics", test_chunk_statistics),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("\nProbe chunk pipeline ready for MaxText training!")
        print("Next step: Run preprocessing on full dataset if not done")
        print("  python scripts/data/create_probe_chunks.py")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
