#!/usr/bin/env python3
"""
Convert sharded Parquet files to PLAN_3 ArrayRecord format.

This script:
1. Reads all parquet files with network measurements
2. Groups measurements by src_addr (probe)
3. Serializes measurements to compact binary (PyArrow IPC)
4. Splits large probes into multiple rows (8MB threshold)
5. Writes to ArrayRecord for efficient random access

Output schema:
{
    'src_id': int64,                    # Probe identifier (hash of src_addr)
    'measurements': binary,              # PyArrow IPC serialized RecordBatch
    'n_measurements': int32,             # Count of measurements
    'time_span_seconds': float64,       # last_time - first_time
    'first_timestamp': timestamp('us'), # For metadata/debugging
    'last_timestamp': timestamp('us'),  # For metadata/debugging
}

Measurements RecordBatch schema (inside binary blob):
{
    'event_time': timestamp('us'),
    'src_addr': string,             # Needed for tokenization
    'dst_addr': string,
    'ip_version': int8,
    'rtt': float32,
}
"""

import argparse
import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from typing import List, Tuple
import hashlib

try:
    import array_record.python.array_record_module as array_record_module
except ImportError:
    raise ImportError(
        "array_record not installed. Install with: pip install array_record"
    )


def hash_src_addr(src_addr: str) -> int:
    """Convert src_addr to stable integer ID (within int64 range)."""
    # Use first 15 hex chars to stay within int64 range (2^63 - 1)
    hash_hex = hashlib.md5(src_addr.encode()).hexdigest()[:15]
    return int(hash_hex, 16)


def serialize_measurements_to_ipc(measurements: List[dict]) -> bytes:
    """
    Serialize list of measurements to PyArrow IPC format.

    Args:
        measurements: List of dicts with {event_time, src_addr, dst_addr, ip_version, rtt}

    Returns:
        Serialized bytes in IPC stream format
    """
    # Define schema
    schema = pa.schema([
        ('event_time', pa.timestamp('us')),
        ('src_addr', pa.string()),
        ('dst_addr', pa.string()),
        ('ip_version', pa.int8()),
        ('rtt', pa.float32()),
    ])

    # Convert to PyArrow Table
    table = pa.Table.from_pylist(measurements, schema=schema)

    # Serialize to IPC format
    sink = pa.BufferOutputStream()
    writer = ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()

    return sink.getvalue().to_pybytes()


def create_arrayrecord_entry(
    src_id: int,
    measurements_bytes: bytes,
    n_measurements: int,
    first_timestamp,
    last_timestamp,
) -> bytes:
    """
    Create single ArrayRecord entry (serialized as PyArrow RecordBatch).

    Returns:
        Serialized RecordBatch in IPC format
    """
    # Calculate time span
    time_span = (last_timestamp - first_timestamp).total_seconds() if last_timestamp != first_timestamp else 0.0

    # Create single-row RecordBatch
    schema = pa.schema([
        ('src_id', pa.int64()),
        ('measurements', pa.binary()),
        ('n_measurements', pa.int32()),
        ('time_span_seconds', pa.float64()),
        ('first_timestamp', pa.timestamp('us')),
        ('last_timestamp', pa.timestamp('us')),
    ])

    arrays = [
        pa.array([src_id], type=pa.int64()),
        pa.array([measurements_bytes], type=pa.binary()),
        pa.array([n_measurements], type=pa.int32()),
        pa.array([time_span], type=pa.float64()),
        pa.array([first_timestamp], type=pa.timestamp('us')),
        pa.array([last_timestamp], type=pa.timestamp('us')),
    ]

    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)

    # Serialize to IPC
    sink = pa.BufferOutputStream()
    writer = ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()

    return sink.getvalue().to_pybytes()


def write_probe_to_arrayrecord(
    writer,
    src_addr: str,
    measurements: List[dict],
    max_size_bytes: int,
) -> int:
    """
    Write probe measurements to ArrayRecord, splitting if needed.

    Args:
        writer: ArrayRecordWriter instance
        src_addr: Source IP address
        measurements: List of measurement dicts (sorted by event_time)
        max_size_bytes: Maximum size per row (8MB)

    Returns:
        Number of rows written
    """
    if not measurements:
        return 0

    src_id = hash_src_addr(src_addr)
    rows_written = 0

    # Try to write all measurements as single row
    measurements_bytes = serialize_measurements_to_ipc(measurements)

    if len(measurements_bytes) <= max_size_bytes:
        # Single row
        entry = create_arrayrecord_entry(
            src_id=src_id,
            measurements_bytes=measurements_bytes,
            n_measurements=len(measurements),
            first_timestamp=measurements[0]['event_time'],
            last_timestamp=measurements[-1]['event_time'],
        )
        writer.write(entry)
        return 1

    # Need to split into multiple rows
    # Binary search for split point
    left = 0
    right = len(measurements)

    while left < len(measurements):
        # Find largest chunk that fits
        lo, hi = 1, min(right - left, len(measurements) - left)
        best_size = 1

        while lo <= hi:
            mid = (lo + hi) // 2
            chunk = measurements[left:left + mid]
            chunk_bytes = serialize_measurements_to_ipc(chunk)

            if len(chunk_bytes) <= max_size_bytes:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Write chunk
        chunk = measurements[left:left + best_size]
        chunk_bytes = serialize_measurements_to_ipc(chunk)

        entry = create_arrayrecord_entry(
            src_id=src_id,
            measurements_bytes=chunk_bytes,
            n_measurements=len(chunk),
            first_timestamp=chunk[0]['event_time'],
            last_timestamp=chunk[-1]['event_time'],
        )
        writer.write(entry)
        rows_written += 1
        left += best_size

    return rows_written


def process_parquet_to_probe_rows(
    input_pattern: str,
    output_dir: Path,
    max_row_size_mb: float = 8.0,
    train_ratio: float = 0.9,
):
    """
    Main processing function.

    Args:
        input_pattern: Glob pattern for input parquet files
        output_dir: Output directory for ArrayRecord files
        max_row_size_mb: Maximum row size in MB
        train_ratio: Ratio of probes for training set
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    max_size_bytes = int(max_row_size_mb * 1024 * 1024)

    print(f"Reading parquet files: {input_pattern}")

    # Connect to DuckDB
    con = duckdb.connect(':memory:')

    # Create view of all measurements
    con.execute(f"""
        CREATE VIEW all_measurements AS
        SELECT * FROM read_parquet('{input_pattern}')
    """)

    # Get total count
    total_count = con.execute("SELECT COUNT(*) FROM all_measurements").fetchone()[0]
    print(f"Total measurements: {total_count:,}")

    # Get unique source addresses
    src_addrs = con.execute("""
        SELECT DISTINCT src_addr FROM all_measurements
        ORDER BY src_addr
    """).fetchall()
    src_addrs = [row[0] for row in src_addrs]

    print(f"Unique probes: {len(src_addrs):,}")

    # Split into train/test by probe
    n_train = int(len(src_addrs) * train_ratio)
    train_addrs = set(src_addrs[:n_train])
    test_addrs = set(src_addrs[n_train:])

    print(f"Train probes: {len(train_addrs):,}")
    print(f"Test probes: {len(test_addrs):,}")

    # Create writers
    train_path = str(output_dir / "train.arrayrecord")
    test_path = str(output_dir / "test.arrayrecord")

    train_writer = array_record_module.ArrayRecordWriter(train_path, 'group_size:1')
    test_writer = array_record_module.ArrayRecordWriter(test_path, 'group_size:1')

    train_rows = 0
    test_rows = 0
    train_measurements_total = 0
    test_measurements_total = 0

    # Process each probe
    for idx, src_addr in enumerate(src_addrs):
        if (idx + 1) % 100 == 0 or idx == len(src_addrs) - 1:
            print(f"Processing probe {idx + 1}/{len(src_addrs)}...", end='\r')

        # Fetch probe's measurements
        result = con.execute("""
            SELECT event_time, src_addr, dst_addr, ip_version, rtt
            FROM all_measurements
            WHERE src_addr = ?
            ORDER BY event_time
        """, [src_addr]).fetchall()

        # Convert to list of dicts
        measurements = [
            {
                'event_time': row[0],
                'src_addr': row[1],
                'dst_addr': row[2],
                'ip_version': row[3],
                'rtt': row[4],
            }
            for row in result
        ]

        # Write to appropriate split
        if src_addr in train_addrs:
            rows = write_probe_to_arrayrecord(
                train_writer, src_addr, measurements, max_size_bytes
            )
            train_rows += rows
            train_measurements_total += len(measurements)
        else:
            rows = write_probe_to_arrayrecord(
                test_writer, src_addr, measurements, max_size_bytes
            )
            test_rows += rows
            test_measurements_total += len(measurements)

    print()  # Newline after progress

    # Close writers
    train_writer.close()
    test_writer.close()

    # Print summary
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print(f"\nTrain set:")
    print(f"  Rows written: {train_rows:,}")
    print(f"  Measurements: {train_measurements_total:,}")
    print(f"  Avg measurements/row: {train_measurements_total/train_rows:.1f}")
    print(f"  Output: {train_path}")

    print(f"\nTest set:")
    print(f"  Rows written: {test_rows:,}")
    print(f"  Measurements: {test_measurements_total:,}")
    print(f"  Avg measurements/row: {test_measurements_total/test_rows:.1f}")
    print(f"  Output: {test_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to PLAN_3 ArrayRecord format"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern for input parquet files (e.g., 'data/sharded/**/*.parquet')"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for ArrayRecord files"
    )
    parser.add_argument(
        "--max-row-size-mb",
        type=float,
        default=8.0,
        help="Maximum row size in MB (default: 8.0)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of probes for training set (default: 0.9)"
    )

    args = parser.parse_args()

    process_parquet_to_probe_rows(
        input_pattern=args.input,
        output_dir=Path(args.output),
        max_row_size_mb=args.max_row_size_mb,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
