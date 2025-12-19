"""
Modal script to inspect and validate probe row ArrayRecord files.

This script provides comprehensive stats and samples for human review of the
probe row transformation process.

Usage:
    modal run scripts/data/modal_inspect_probe_rows.py
"""

import modal
from pathlib import Path
import sys

app = modal.App("probe-rows-inspector")

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow",
        "array_record",
        "numpy",
        "pandas",
    )
)

# Mount the volume
volume = modal.Volume.from_name("ping-llm", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=3600,  # 1 hour
    cpu=4.0,
    memory=16 * 1024,  # 16GB
)
def inspect_probe_rows(
    data_dir: str = "/mnt/data/probe_rows",
    num_samples_per_split: int = 5,
    check_ordering: bool = True,
    check_splitting: bool = True,
):
    """
    Inspect probe row files and output comprehensive stats.

    Args:
        data_dir: Directory containing train.arrayrecord and test.arrayrecord
        num_samples_per_split: Number of sample rows to display per split
        check_ordering: Verify timestamp ordering within probes
        check_splitting: Check if large probes are split correctly
    """
    import array_record.python.array_record_module as array_record_module
    import pyarrow.ipc as ipc
    import numpy as np
    from collections import defaultdict
    from datetime import datetime

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
        return ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, 'strftime') else str(ts)

    def check_measurement_ordering(measurements: list) -> dict:
        """Check if measurements are ordered by event_time."""
        if len(measurements) <= 1:
            return {'ordered': True, 'violations': 0, 'details': []}

        violations = []
        for i in range(1, len(measurements)):
            if measurements[i]['event_time'] < measurements[i-1]['event_time']:
                violations.append({
                    'index': i,
                    'prev_time': measurements[i-1]['event_time'],
                    'curr_time': measurements[i]['event_time'],
                })

        return {
            'ordered': len(violations) == 0,
            'violations': len(violations),
            'details': violations[:5],  # Show first 5 violations
        }

    def check_src_addr_consistency(measurements: list) -> dict:
        """Check if all measurements have the same src_addr."""
        if not measurements:
            return {'consistent': True, 'unique_addrs': 0}

        unique_addrs = set(m['src_addr'] for m in measurements)
        return {
            'consistent': len(unique_addrs) == 1,
            'unique_addrs': len(unique_addrs),
            'addresses': list(unique_addrs)[:5],  # Show first 5
        }

    print("=" * 100)
    print("PROBE ROW INSPECTION REPORT")
    print("=" * 100)
    print(f"Data directory: {data_dir}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    base_dir = Path(data_dir)

    # Check for files
    train_path = base_dir / "train.arrayrecord"
    test_path = base_dir / "test.arrayrecord"

    if not train_path.exists():
        print(f"❌ ERROR: Train file not found: {train_path}")
        return
    if not test_path.exists():
        print(f"⚠️  WARNING: Test file not found: {test_path}")
        test_path = None

    # Process each split
    for split_name, split_path in [("TRAIN", train_path), ("TEST", test_path)]:
        if split_path is None:
            continue

        print("\n" + "=" * 100)
        print(f"{split_name} SET ANALYSIS")
        print("=" * 100)

        reader = array_record_module.ArrayRecordReader(str(split_path))
        total_rows = reader.num_records()

        print(f"\n📊 BASIC STATISTICS")
        print(f"{'─' * 100}")
        print(f"Total rows: {total_rows:,}")

        if total_rows == 0:
            print("⚠️  No rows found!")
            continue

        # Collect statistics
        print("\nScanning all rows for statistics...")
        row_sizes = []
        measurement_counts = []
        time_spans = []
        src_ids = set()
        src_id_to_rows = defaultdict(list)  # Track which rows belong to which src_id
        total_measurements = 0

        for i in range(total_rows):
            if (i + 1) % 1000 == 0 or i == total_rows - 1:
                print(f"  Progress: {i + 1:,}/{total_rows:,} ({100 * (i + 1) / total_rows:.1f}%)", end='\r')

            record_bytes = reader.read([i])[0]
            row_sizes.append(len(record_bytes))

            meta = deserialize_row_metadata(record_bytes)
            measurement_counts.append(meta['n_measurements'])
            time_spans.append(meta['time_span_seconds'])
            src_ids.add(meta['src_id'])
            src_id_to_rows[meta['src_id']].append(i)
            total_measurements += meta['n_measurements']

        print()  # Newline after progress

        # Statistics summary
        print(f"\n📈 SIZE DISTRIBUTION")
        print(f"{'─' * 100}")
        print(f"  Min row size:    {np.min(row_sizes):>12,} bytes  ({np.min(row_sizes) / (1024**2):>6.2f} MB)")
        print(f"  Max row size:    {np.max(row_sizes):>12,} bytes  ({np.max(row_sizes) / (1024**2):>6.2f} MB)")
        print(f"  Mean row size:   {np.mean(row_sizes):>12,.0f} bytes  ({np.mean(row_sizes) / (1024**2):>6.2f} MB)")
        print(f"  Median row size: {np.median(row_sizes):>12,.0f} bytes  ({np.median(row_sizes) / (1024**2):>6.2f} MB)")

        # Check for rows near 8MB limit
        large_rows = sum(1 for s in row_sizes if s > 7.5 * 1024 * 1024)
        if large_rows > 0:
            print(f"  ⚠️  Rows > 7.5MB:  {large_rows:>12,} ({100 * large_rows / total_rows:.2f}%)")

        print(f"\n📊 MEASUREMENT DISTRIBUTION")
        print(f"{'─' * 100}")
        print(f"  Total measurements:     {total_measurements:>15,}")
        print(f"  Min measurements/row:   {np.min(measurement_counts):>15,}")
        print(f"  Max measurements/row:   {np.max(measurement_counts):>15,}")
        print(f"  Mean measurements/row:  {np.mean(measurement_counts):>15,.1f}")
        print(f"  Median measurements/row:{np.median(measurement_counts):>15,.0f}")
        print()
        print(f"  Percentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    P{p:>2}: {np.percentile(measurement_counts, p):>15,.0f}")

        print(f"\n⏱️  TIME SPAN DISTRIBUTION")
        print(f"{'─' * 100}")
        print(f"  Min time span:    {np.min(time_spans):>12,.1f} seconds  ({np.min(time_spans) / 3600:>8.2f} hours)")
        print(f"  Max time span:    {np.max(time_spans):>12,.1f} seconds  ({np.max(time_spans) / 3600:>8.2f} hours)")
        print(f"  Mean time span:   {np.mean(time_spans):>12,.1f} seconds  ({np.mean(time_spans) / 3600:>8.2f} hours)")
        print(f"  Median time span: {np.median(time_spans):>12,.1f} seconds  ({np.median(time_spans) / 3600:>8.2f} hours)")

        print(f"\n🔑 PROBE (src_id) STATISTICS")
        print(f"{'─' * 100}")
        print(f"  Unique src_ids:           {len(src_ids):>12,}")
        print(f"  Rows per unique src_id:   {total_rows / len(src_ids):>12,.2f}")

        # Check for split probes (src_id appearing in multiple rows)
        split_probes = {sid: rows for sid, rows in src_id_to_rows.items() if len(rows) > 1}
        if split_probes:
            print(f"  ✂️  Split probes (>1 row):   {len(split_probes):>12,} ({100 * len(split_probes) / len(src_ids):.2f}%)")
            print(f"      Max rows per probe:     {max(len(rows) for rows in split_probes.values()):>12,}")

            # Show examples of split probes
            print(f"\n      Top 5 most-split probes:")
            split_sorted = sorted(split_probes.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            for src_id, row_indices in split_sorted:
                print(f"        src_id {src_id:>20}: {len(row_indices):>3} rows (indices: {row_indices[:10]}{'...' if len(row_indices) > 10 else ''})")
        else:
            print(f"  ✓ No split probes (all probes fit in single row)")

        # Data quality checks
        print(f"\n🔍 DATA QUALITY CHECKS")
        print(f"{'─' * 100}")

        # Sample random rows for quality checks
        sample_size = min(100, total_rows)
        sample_indices = np.random.choice(total_rows, size=sample_size, replace=False)

        ordering_issues = 0
        consistency_issues = 0

        if check_ordering or check_splitting:
            print(f"Checking {sample_size} random rows...")
            for idx in sample_indices:
                record_bytes = reader.read([int(idx)])[0]
                meta = deserialize_row_metadata(record_bytes)
                measurements = deserialize_measurements(meta['measurements_bytes'])

                if check_ordering:
                    order_result = check_measurement_ordering(measurements)
                    if not order_result['ordered']:
                        ordering_issues += 1

                consistency_result = check_src_addr_consistency(measurements)
                if not consistency_result['consistent']:
                    consistency_issues += 1

        if check_ordering:
            if ordering_issues == 0:
                print(f"  ✓ Timestamp ordering: PASS (0 violations in {sample_size} samples)")
            else:
                print(f"  ❌ Timestamp ordering: FAIL ({ordering_issues}/{sample_size} samples have violations)")

        if consistency_issues == 0:
            print(f"  ✓ src_addr consistency: PASS (all measurements in each row have same src_addr)")
        else:
            print(f"  ❌ src_addr consistency: FAIL ({consistency_issues}/{sample_size} samples have inconsistent src_addr)")

        # Sample display
        print(f"\n📋 SAMPLE ROWS (showing {min(num_samples_per_split, total_rows)})")
        print(f"{'─' * 100}")

        # Sample strategy: beginning, middle, end, and largest
        sample_indices = []
        if total_rows > 0:
            sample_indices.append(0)  # First
        if total_rows > 1:
            sample_indices.append(total_rows // 2)  # Middle
        if total_rows > 2:
            sample_indices.append(total_rows - 1)  # Last

        # Add largest row
        if total_rows > 3:
            largest_idx = int(np.argmax(row_sizes))
            if largest_idx not in sample_indices:
                sample_indices.append(largest_idx)

        # Add random samples
        remaining = num_samples_per_split - len(sample_indices)
        if remaining > 0:
            available = [i for i in range(total_rows) if i not in sample_indices]
            if available:
                random_samples = np.random.choice(available, size=min(remaining, len(available)), replace=False)
                sample_indices.extend(random_samples)

        for sample_num, idx in enumerate(sorted(sample_indices)[:num_samples_per_split], 1):
            print(f"\n{'─' * 100}")
            print(f"Sample {sample_num} - Row Index {idx:,} (Position: {100 * idx / total_rows:.1f}%)")
            print(f"{'─' * 100}")

            record_bytes = reader.read([idx])[0]
            meta = deserialize_row_metadata(record_bytes)
            measurements = deserialize_measurements(meta['measurements_bytes'])

            print(f"  src_id:            {meta['src_id']}")
            print(f"  n_measurements:    {meta['n_measurements']:,}")
            print(f"  time_span:         {meta['time_span_seconds']:.1f} seconds ({meta['time_span_seconds'] / 3600:.2f} hours)")
            print(f"  first_timestamp:   {format_timestamp(meta['first_timestamp'])}")
            print(f"  last_timestamp:    {format_timestamp(meta['last_timestamp'])}")
            print(f"  row_size:          {len(record_bytes):,} bytes ({len(record_bytes) / (1024**2):.2f} MB)")

            # Check if this src_id appears in multiple rows
            if meta['src_id'] in split_probes:
                row_list = src_id_to_rows[meta['src_id']]
                print(f"  ✂️  SPLIT PROBE:      This src_id appears in {len(row_list)} rows: {row_list}")

            # Show measurement samples
            print(f"\n  First 3 measurements:")
            for i, meas in enumerate(measurements[:3]):
                print(f"    [{i}] {format_timestamp(meas['event_time'])} | {meas['src_addr']} -> {meas['dst_addr']} | RTT: {meas['rtt']:.3f}ms | IPv{meas['ip_version']}")

            if len(measurements) > 6:
                print(f"    ... ({len(measurements) - 6} measurements omitted) ...")

            if len(measurements) > 3:
                print(f"\n  Last 3 measurements:")
                for i, meas in enumerate(measurements[-3:], len(measurements) - 3):
                    print(f"    [{i}] {format_timestamp(meas['event_time'])} | {meas['src_addr']} -> {meas['dst_addr']} | RTT: {meas['rtt']:.3f}ms | IPv{meas['ip_version']}")

            # Detailed checks for this sample
            order_check = check_measurement_ordering(measurements)
            consistency_check = check_src_addr_consistency(measurements)

            print(f"\n  Quality checks:")
            if order_check['ordered']:
                print(f"    ✓ Timestamps are ordered")
            else:
                print(f"    ❌ Timestamps NOT ordered ({order_check['violations']} violations)")
                if order_check['details']:
                    print(f"       First violation at index {order_check['details'][0]['index']}")

            if consistency_check['consistent']:
                print(f"    ✓ All measurements have same src_addr")
            else:
                print(f"    ❌ Inconsistent src_addr: {consistency_check['addresses']}")

        reader.close()

    # Summary
    print("\n" + "=" * 100)
    print("INSPECTION COMPLETE")
    print("=" * 100)
    print("\n✅ What to look for:")
    print("  1. ✓ All quality checks should PASS")
    print("  2. ✓ Row sizes should be < 8MB (with some close to it)")
    print("  3. ✓ Measurements should be ordered by timestamp within each row")
    print("  4. ✓ All measurements in a row should have the same src_addr")
    print("  5. ✓ Split probes (if any) should have reasonable split counts")
    print("  6. ✓ Sample data should look realistic (valid timestamps, RTTs, IPs)")
    print("\n" + "=" * 100)


@app.local_entrypoint()
def main():
    """Run inspection from command line."""
    print("Starting probe row inspection on Modal...")
    inspect_probe_rows.remote()
    print("\n✓ Inspection complete!")
