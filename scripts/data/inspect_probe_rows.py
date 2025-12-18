#!/usr/bin/env python3
"""
Inspect PLAN_3 ArrayRecord probe row files.

This utility helps debug and validate probe row data by:
1. Showing summary statistics
2. Displaying row size distributions
3. Pretty-printing sample measurements
"""

import argparse
import pyarrow.ipc as ipc
import numpy as np
from pathlib import Path

try:
    import array_record.python.array_record_module as array_record_module
except ImportError:
    raise ImportError(
        "array_record not installed. Install with: pip install array_record"
    )


def deserialize_measurements(measurements_bytes: bytes) -> list:
    """Deserialize measurements from PyArrow IPC format."""
    reader = ipc.open_stream(measurements_bytes)
    table = reader.read_all()
    reader.close()
    return table.to_pylist()


def inspect_arrayrecord(arrayrecord_path: str, num_samples: int = 3):
    """
    Inspect an ArrayRecord file and print statistics.

    Args:
        arrayrecord_path: Path to ArrayRecord file
        num_samples: Number of sample rows to display
    """
    print(f"Inspecting: {arrayrecord_path}")
    print("=" * 80)

    reader = array_record_module.ArrayRecordReader(arrayrecord_path)
    total_rows = reader.num_records()

    print(f"\nTotal rows: {total_rows:,}")

    if total_rows == 0:
        print("No rows found!")
        return

    # Collect statistics
    row_sizes = []
    measurements_counts = []
    time_spans = []
    total_measurements = 0

    print("\nScanning rows...")
    for i in range(total_rows):
        if (i + 1) % 100 == 0 or i == total_rows - 1:
            print(f"Progress: {i + 1}/{total_rows}", end='\r')

        record_bytes = reader.read([i])[0]
        row_sizes.append(len(record_bytes))

        # Deserialize row
        reader_ipc = ipc.open_stream(record_bytes)
        batch = reader_ipc.read_next_batch()
        reader_ipc.close()

        n_meas = batch.column('n_measurements')[0].as_py()
        measurements_counts.append(n_meas)
        total_measurements += n_meas
        time_spans.append(batch.column('time_span_seconds')[0].as_py())

    print()  # Newline after progress

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    print(f"\nRow sizes (bytes):")
    print(f"  Min: {np.min(row_sizes):,}")
    print(f"  Max: {np.max(row_sizes):,}")
    print(f"  Mean: {np.mean(row_sizes):,.0f}")
    print(f"  Median: {np.median(row_sizes):,.0f}")

    print(f"\nMeasurements per row:")
    print(f"  Min: {np.min(measurements_counts):,}")
    print(f"  Max: {np.max(measurements_counts):,}")
    print(f"  Mean: {np.mean(measurements_counts):,.1f}")
    print(f"  Median: {np.median(measurements_counts):,.0f}")
    print(f"  Total: {total_measurements:,}")

    print(f"\nTime span per row (seconds):")
    print(f"  Min: {np.min(time_spans):,.1f}")
    print(f"  Max: {np.max(time_spans):,.1f}")
    print(f"  Mean: {np.mean(time_spans):,.1f}")
    print(f"  Median: {np.median(time_spans):,.1f}")

    # Print percentiles
    print(f"\nMeasurements per row percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(measurements_counts, p):,.0f}")

    # Sample rows
    print("\n" + "=" * 80)
    print(f"SAMPLE ROWS (showing up to {num_samples})")
    print("=" * 80)

    sample_indices = np.linspace(0, total_rows - 1, min(num_samples, total_rows), dtype=int)

    for idx in sample_indices:
        print(f"\n--- Row {idx} ---")

        record_bytes = reader.read([idx])[0]

        # Deserialize row metadata
        reader_ipc = ipc.open_stream(record_bytes)
        batch = reader_ipc.read_next_batch()
        reader_ipc.close()

        src_id = batch.column('src_id')[0].as_py()
        measurements_bytes = batch.column('measurements')[0].as_py()
        n_measurements = batch.column('n_measurements')[0].as_py()
        time_span = batch.column('time_span_seconds')[0].as_py()
        first_ts = batch.column('first_timestamp')[0].as_py()
        last_ts = batch.column('last_timestamp')[0].as_py()

        print(f"  src_id: {src_id}")
        print(f"  n_measurements: {n_measurements:,}")
        print(f"  time_span: {time_span:.1f}s")
        print(f"  first_timestamp: {first_ts}")
        print(f"  last_timestamp: {last_ts}")
        print(f"  row_size: {len(record_bytes):,} bytes")

        # Deserialize measurements
        measurements = deserialize_measurements(measurements_bytes)

        print(f"\n  First 3 measurements:")
        for i, meas in enumerate(measurements[:3]):
            print(f"    [{i}] {meas}")

        if len(measurements) > 3:
            print(f"    ... ({len(measurements) - 3} more)")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PLAN_3 ArrayRecord probe row files"
    )
    parser.add_argument(
        "arrayrecord_path",
        help="Path to ArrayRecord file to inspect"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample rows to display (default: 3)"
    )

    args = parser.parse_args()

    if not Path(args.arrayrecord_path).exists():
        print(f"Error: File not found: {args.arrayrecord_path}")
        return 1

    inspect_arrayrecord(args.arrayrecord_path, args.samples)
    return 0


if __name__ == "__main__":
    exit(main())
