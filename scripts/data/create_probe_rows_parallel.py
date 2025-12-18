#!/usr/bin/env python3
"""
Parallelized probe row preprocessing for PLAN_3.

Implements three optimization strategies:
1. DuckDB GROUP BY - single query instead of N queries
2. Multiprocessing - parallel serialization across CPU cores
3. Batch writing - improved I/O efficiency

Expected speedup: 10-50x over sequential version
"""

import argparse
import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from typing import List, Tuple
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial
import time

try:
    import array_record.python.array_record_module as array_record_module
except ImportError:
    raise ImportError(
        "array_record not installed. Install with: pip install array_record"
    )


def hash_src_addr(src_addr: str) -> int:
    """Convert src_addr to stable integer ID (within int64 range)."""
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

    while left < len(measurements):
        # Find largest chunk that fits
        lo, hi = 1, len(measurements) - left
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


def process_probe_batch(args):
    """
    Worker function to process a batch of probes.

    Args:
        args: Tuple of (probe_batch, output_path, max_size_bytes, worker_id)

    Returns:
        Tuple of (output_path, rows_written, measurements_total, worker_id)
    """
    probe_batch, output_path, max_size_bytes, worker_id = args

    writer = array_record_module.ArrayRecordWriter(output_path, 'group_size:1')

    rows_written = 0
    measurements_total = 0

    for src_addr, measurements_struct_list in probe_batch:
        # Convert struct list to list of dicts
        measurements = [
            {
                'event_time': m['event_time'],
                'src_addr': m['src_addr'],
                'dst_addr': m['dst_addr'],
                'ip_version': m['ip_version'],
                'rtt': m['rtt'],
            }
            for m in measurements_struct_list
        ]

        rows = write_probe_to_arrayrecord(writer, src_addr, measurements, max_size_bytes)
        rows_written += rows
        measurements_total += len(measurements)

    writer.close()

    print(f"  Worker {worker_id} complete: {len(probe_batch)} probes, {rows_written} rows, {measurements_total} measurements")
    return output_path, rows_written, measurements_total, worker_id


def merge_arrayrecords(input_paths: List[str], output_path: str):
    """
    Merge multiple ArrayRecord files into one.

    Args:
        input_paths: List of partial ArrayRecord file paths
        output_path: Output merged ArrayRecord path
    """
    print(f"\nMerging {len(input_paths)} partial files...")

    writer = array_record_module.ArrayRecordWriter(output_path, 'group_size:1')

    total_rows = 0
    for i, input_path in enumerate(input_paths):
        print(f"  Merging {i+1}/{len(input_paths)}: {input_path}", end='\r')

        reader = array_record_module.ArrayRecordReader(input_path)
        n_records = reader.num_records()

        # Read and write all records
        for j in range(n_records):
            record_bytes = reader.read([j])[0]
            writer.write(record_bytes)
            total_rows += 1

        reader.close()

    writer.close()
    print(f"\n  Merged {total_rows:,} rows into {output_path}")


def process_parquet_to_probe_rows_parallel(
    input_pattern: str,
    output_dir: Path,
    max_row_size_mb: float = 8.0,
    train_ratio: float = 0.9,
    num_workers: int = None,
):
    """
    Main processing function with parallel execution.

    Args:
        input_pattern: Glob pattern for input parquet files
        output_dir: Output directory for ArrayRecord files
        max_row_size_mb: Maximum row size in MB
        train_ratio: Ratio of probes for training set
        num_workers: Number of worker processes (default: CPU count)
    """
    if num_workers is None:
        num_workers = cpu_count()

    output_dir.mkdir(parents=True, exist_ok=True)
    max_size_bytes = int(max_row_size_mb * 1024 * 1024)

    print(f"=" * 80)
    print(f"PARALLELIZED PROBE ROW PREPROCESSING")
    print(f"=" * 80)
    print(f"Workers: {num_workers}")
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")
    print(f"Max row size: {max_row_size_mb}MB")
    print(f"Train ratio: {train_ratio}")
    print()

    start_time = time.time()

    # STRATEGY 1: DuckDB GROUP BY (single query)
    print("Step 1: Loading and grouping data with DuckDB...")
    con = duckdb.connect(':memory:')

    # Create view
    con.execute(f"""
        CREATE VIEW all_measurements AS
        SELECT * FROM read_parquet('{input_pattern}')
    """)

    # Get total count
    total_count = con.execute("SELECT COUNT(*) FROM all_measurements").fetchone()[0]
    print(f"  Total measurements: {total_count:,}")

    # GROUP BY in single query (leverage DuckDB parallelism)
    print("  Grouping by src_addr (this may take a minute)...")
    group_start = time.time()

    probe_groups = con.execute("""
        SELECT
            src_addr,
            LIST(STRUCT_PACK(
                event_time := event_time,
                src_addr := src_addr,
                dst_addr := dst_addr,
                ip_version := ip_version,
                rtt := rtt
            ) ORDER BY event_time) as measurements
        FROM all_measurements
        GROUP BY src_addr
        ORDER BY src_addr
    """).fetchall()

    group_time = time.time() - group_start
    print(f"  Grouped {len(probe_groups):,} probes in {group_time:.1f}s")

    # Split into train/test
    n_train = int(len(probe_groups) * train_ratio)
    train_probes = probe_groups[:n_train]
    test_probes = probe_groups[n_train:]

    print(f"  Train probes: {len(train_probes):,}")
    print(f"  Test probes: {len(test_probes):,}")

    # STRATEGY 2: Multiprocessing workers
    print(f"\nStep 2: Processing probes in parallel ({num_workers} workers)...")

    # Partition train probes across workers
    train_batch_size = len(train_probes) // num_workers
    test_batch_size = len(test_probes) // num_workers if len(test_probes) > 0 else 0

    # Prepare work batches
    train_batches = []
    test_batches = []

    print("  Partitioning work...")
    for i in range(num_workers):
        # Train batches
        start_idx = i * train_batch_size
        end_idx = start_idx + train_batch_size if i < num_workers - 1 else len(train_probes)
        if start_idx < len(train_probes):
            batch = train_probes[start_idx:end_idx]
            output_path = str(output_dir / f"train_part_{i}.arrayrecord")
            train_batches.append((batch, output_path, max_size_bytes, i))

        # Test batches
        if len(test_probes) > 0:
            start_idx = i * test_batch_size
            end_idx = start_idx + test_batch_size if i < num_workers - 1 else len(test_probes)
            if start_idx < len(test_probes):
                batch = test_probes[start_idx:end_idx]
                output_path = str(output_dir / f"test_part_{i}.arrayrecord")
                test_batches.append((batch, output_path, max_size_bytes, i))

    # Process train set in parallel
    print(f"\n  Processing TRAIN set ({len(train_probes):,} probes)...")
    process_start = time.time()

    with Pool(num_workers) as pool:
        train_results = pool.map(process_probe_batch, train_batches)

    train_time = time.time() - process_start
    print(f"\n  Train processing complete in {train_time:.1f}s")

    # Process test set in parallel
    if test_batches:
        print(f"\n  Processing TEST set ({len(test_probes):,} probes)...")
        test_start = time.time()

        with Pool(num_workers) as pool:
            test_results = pool.map(process_probe_batch, test_batches)

        test_time = time.time() - test_start
        print(f"\n  Test processing complete in {test_time:.1f}s")
    else:
        test_results = []

    # STRATEGY 3: Efficient merging (batch I/O)
    print("\nStep 3: Merging partial files...")

    # Merge train files
    train_partial_files = [result[0] for result in train_results]
    train_output = str(output_dir / "train.arrayrecord")
    merge_arrayrecords(train_partial_files, train_output)

    # Merge test files
    if test_results:
        test_partial_files = [result[0] for result in test_results]
        test_output = str(output_dir / "test.arrayrecord")
        merge_arrayrecords(test_partial_files, test_output)

    # Clean up partial files
    print("\nStep 4: Cleaning up partial files...")
    for partial_file in train_partial_files + ([result[0] for result in test_results] if test_results else []):
        Path(partial_file).unlink()
        print(f"  Deleted {partial_file}")

    # Calculate statistics
    train_rows_total = sum(result[1] for result in train_results)
    train_measurements_total = sum(result[2] for result in train_results)

    test_rows_total = sum(result[1] for result in test_results) if test_results else 0
    test_measurements_total = sum(result[2] for result in test_results) if test_results else 0

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  DuckDB grouping: {group_time:.1f}s")
    print(f"  Train processing: {train_time:.1f}s")
    if test_results:
        print(f"  Test processing: {test_time:.1f}s")
    print(f"  Throughput: {total_count/total_time:,.0f} measurements/sec")

    print(f"\nTrain set:")
    print(f"  Rows written: {train_rows_total:,}")
    print(f"  Measurements: {train_measurements_total:,}")
    print(f"  Avg measurements/row: {train_measurements_total/train_rows_total:.1f}")
    print(f"  Output: {train_output}")

    if test_results:
        print(f"\nTest set:")
        print(f"  Rows written: {test_rows_total:,}")
        print(f"  Measurements: {test_measurements_total:,}")
        print(f"  Avg measurements/row: {test_measurements_total/test_rows_total:.1f}")
        print(f"  Output: {test_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallelized Parquet to PLAN_3 ArrayRecord conversion"
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )

    args = parser.parse_args()

    process_parquet_to_probe_rows_parallel(
        input_pattern=args.input,
        output_dir=Path(args.output),
        max_row_size_mb=args.max_row_size_mb,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
