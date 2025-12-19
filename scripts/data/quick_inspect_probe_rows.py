#!/usr/bin/env python3
"""
Quick local inspection of probe row files.

Lighter weight version for quick checks during development.

Usage:
    python scripts/data/quick_inspect_probe_rows.py data/probe_rows/train.arrayrecord
    python scripts/data/quick_inspect_probe_rows.py data/probe_rows/test.arrayrecord --samples 10
"""

import argparse
import array_record.python.array_record_module as array_record_module
import pyarrow.ipc as ipc
import numpy as np
from pathlib import Path
from collections import defaultdict


def deserialize_measurements(measurements_bytes: bytes) -> list:
    """Deserialize measurements from PyArrow IPC format."""
    reader = ipc.open_stream(measurements_bytes)
    table = reader.read_all()
    reader.close()
    return table.to_pylist()


def deserialize_row_metadata(record_bytes: bytes) -> dict:
    """Deserialize row metadata."""
    reader = ipc.open_stream(record_bytes)
    batch = reader.read_next_batch()
    reader.close()

    return {
        'src_id': batch.column('src_id')[0].as_py(),
        'measurements_bytes': batch.column('measurements')[0].as_py(),
        'n_measurements': batch.column('n_measurements')[0].as_py(),
        'time_span_seconds': batch.column('time_span_seconds')[0].as_py(),
        'first_timestamp': batch.column('first_timestamp')[0].as_py(),
        'last_timestamp': batch.column('last_timestamp')[0].as_py(),
    }


def format_timestamp(ts):
    """Format timestamp for display."""
    if ts is None:
        return "None"
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if hasattr(ts, 'strftime') else str(ts)


def inspect_file(file_path: str, num_samples: int = 5, quick: bool = False, debug: bool = False):
    """Inspect a single ArrayRecord file."""

    print(f"=" * 80)
    print(f"Inspecting: {file_path}")
    print(f"=" * 80)

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    reader = array_record_module.ArrayRecordReader(file_path)
    total_rows = reader.num_records()

    print(f"\n📊 Total rows: {total_rows:,}")

    if total_rows == 0:
        print("⚠️  No rows found!")
        return

    # Quick scan
    print("\nScanning rows...")
    row_sizes = []
    measurement_counts = []
    time_spans = []
    src_ids = set()
    src_id_to_rows = defaultdict(list)
    total_measurements = 0

    scan_limit = min(1000, total_rows) if quick else total_rows
    sample_step = max(1, total_rows // scan_limit)

    for i in range(0, total_rows, sample_step):
        if (i + 1) % 100 == 0 or i == total_rows - 1:
            print(f"  Progress: {i + 1:,}/{total_rows:,}", end='\r')

        record_bytes = reader.read([i])[0]
        row_sizes.append(len(record_bytes))

        meta = deserialize_row_metadata(record_bytes)
        measurement_counts.append(meta['n_measurements'])
        time_spans.append(meta['time_span_seconds'])
        src_ids.add(meta['src_id'])
        src_id_to_rows[meta['src_id']].append(i)
        total_measurements += meta['n_measurements']

    print()

    # Statistics
    print(f"\n📈 Statistics:")
    print(f"  Row sizes:         {np.min(row_sizes) / 1024**2:.2f} - {np.max(row_sizes) / 1024**2:.2f} MB (avg: {np.mean(row_sizes) / 1024**2:.2f} MB)")
    print(f"  Measurements/row:  {np.min(measurement_counts):,} - {np.max(measurement_counts):,} (avg: {np.mean(measurement_counts):.1f})")
    print(f"  Time spans:        {np.min(time_spans):.1f} - {np.max(time_spans):.1f} seconds (avg: {np.mean(time_spans):.1f}s)")
    print(f"  Total measurements:{total_measurements:,}")
    print(f"  Unique src_ids:    {len(src_ids):,}")

    # Check for split probes
    split_probes = {sid: rows for sid, rows in src_id_to_rows.items() if len(rows) > 1}
    if split_probes:
        print(f"  ✂️  Split probes:     {len(split_probes):,} ({100 * len(split_probes) / len(src_ids):.2f}%)")
        print(f"      Max splits:      {max(len(rows) for rows in split_probes.values())}")
    else:
        print(f"  ✓ No split probes")

    # Samples
    print(f"\n📋 Sample rows:")

    sample_indices = []
    if total_rows > 0:
        sample_indices.append(0)  # First
    if total_rows > 1:
        sample_indices.append(total_rows // 2)  # Middle
    if total_rows > 2:
        sample_indices.append(total_rows - 1)  # Last
    if total_rows > 3 and row_sizes:
        sample_indices.append(int(np.argmax(row_sizes)))  # Largest

    # Random samples
    remaining = num_samples - len(sample_indices)
    if remaining > 0 and total_rows > len(sample_indices):
        available = [i for i in range(total_rows) if i not in sample_indices]
        if available:
            random_samples = np.random.choice(available, size=min(remaining, len(available)), replace=False)
            sample_indices.extend(random_samples)

    for idx in sorted(set(sample_indices))[:num_samples]:
        print(f"\n  {'─' * 76}")
        print(f"  Row {idx:,}:")

        record_bytes = reader.read([idx])[0]
        meta = deserialize_row_metadata(record_bytes)
        measurements = deserialize_measurements(meta['measurements_bytes'])

        print(f"    src_id:         {meta['src_id']}")
        print(f"    measurements:   {meta['n_measurements']:,}")
        print(f"    time_span:      {meta['time_span_seconds']:.1f}s ({meta['time_span_seconds'] / 3600:.2f}h)")
        print(f"    size:           {len(record_bytes) / 1024**2:.2f} MB")
        print(f"    timestamps:     {format_timestamp(meta['first_timestamp'])} -> {format_timestamp(meta['last_timestamp'])}")

        if meta['src_id'] in split_probes:
            row_list = src_id_to_rows[meta['src_id']]
            print(f"    ✂️  Split:        This src_id has {len(row_list)} rows")

        # Detailed ordering check
        violations = []
        for i in range(len(measurements) - 1):
            if measurements[i]['event_time'] > measurements[i+1]['event_time']:
                violations.append((i, measurements[i]['event_time'], measurements[i+1]['event_time']))

        ordered = len(violations) == 0
        print(f"    Ordering:       {'✓ OK' if ordered else f'❌ {len(violations)} VIOLATIONS'}")

        if debug and violations:
            print(f"\n    VIOLATION DETAILS (first 10):")
            for i, (idx_v, prev_t, curr_t) in enumerate(violations[:10]):
                print(f"      [{idx_v}->]{idx_v+1}]: {format_timestamp(prev_t)} > {format_timestamp(curr_t)} (delta: {(prev_t - curr_t).total_seconds():.3f}s)")

        # Check src_addr consistency
        unique_addrs = set(m['src_addr'] for m in measurements)
        print(f"    Consistency:    {'✓ OK' if len(unique_addrs) == 1 else f'❌ {len(unique_addrs)} different src_addr'}")

        # Show first and last measurements
        print(f"    First:          {format_timestamp(measurements[0]['event_time'])} | {measurements[0]['src_addr']} -> {measurements[0]['dst_addr']} | {measurements[0]['rtt']:.3f}ms")
        if len(measurements) > 1:
            print(f"    Last:           {format_timestamp(measurements[-1]['event_time'])} | {measurements[-1]['src_addr']} -> {measurements[-1]['dst_addr']} | {measurements[-1]['rtt']:.3f}ms")

        if debug and len(measurements) > 1:
            # Show detailed timestamp progression
            print(f"\n    TIMESTAMP PROGRESSION (showing indices around violations):")
            if violations:
                for vi, (idx_v, _, _) in enumerate(violations[:3]):
                    start = max(0, idx_v - 2)
                    end = min(len(measurements), idx_v + 3)
                    print(f"      Violation {vi+1} context:")
                    for i in range(start, end):
                        marker = " ⚠️ " if i == idx_v or i == idx_v + 1 else "    "
                        print(f"        {marker}[{i}] {format_timestamp(measurements[i]['event_time'])}")
            else:
                # Show first 5 and last 5 timestamps
                print(f"      First 5:")
                for i in range(min(5, len(measurements))):
                    print(f"        [{i}] {format_timestamp(measurements[i]['event_time'])}")
                if len(measurements) > 10:
                    print(f"      ...")
                    print(f"      Last 5:")
                    for i in range(max(0, len(measurements) - 5), len(measurements)):
                        print(f"        [{i}] {format_timestamp(measurements[i]['event_time'])}")

    reader.close()

    print(f"\n{'=' * 80}")
    print("✓ Inspection complete")


def main():
    parser = argparse.ArgumentParser(
        description="Quick inspection of probe row ArrayRecord files"
    )
    parser.add_argument(
        "file_path",
        help="Path to ArrayRecord file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to display (default: 5)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only scan first 1000 rows for stats"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: show detailed violation information and timestamp progressions"
    )

    args = parser.parse_args()

    inspect_file(args.file_path, args.samples, args.quick, args.debug)


if __name__ == "__main__":
    main()
