#!/usr/bin/env python3
"""
Memory-efficient parallelized probe row preprocessing for PLAN_3.

This version uses streaming processing to avoid OOM errors on large datasets.

Improvements over create_probe_rows_parallel.py:
1. DuckDB memory limits (configurable)
2. Streaming GROUP BY to disk (avoids loading 200M rows into RAM)
3. Batch processing from disk
4. Works on Modal with limited memory
"""

import argparse
import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from typing import List, Tuple
import hashlib
from multiprocessing import Pool, cpu_count
import time
import tempfile
import shutil
import psutil

try:
    import array_record.python.array_record_module as array_record_module
except ImportError:
    raise ImportError(
        "array_record not installed. Install with: pip install array_record"
    )


def get_available_memory_gb():
    """Get available system memory in GB."""
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except:
        return 4.0  # Default fallback


def hash_src_addr(src_addr: str) -> int:
    """Convert src_addr to stable integer ID (within int64 range)."""
    hash_hex = hashlib.md5(src_addr.encode()).hexdigest()[:15]
    return int(hash_hex, 16)


def serialize_measurements_to_ipc(measurements: List[dict]) -> bytes:
    """Serialize list of measurements to PyArrow IPC format."""
    schema = pa.schema([
        ('event_time', pa.timestamp('us')),
        ('src_addr', pa.string()),
        ('dst_addr', pa.string()),
        ('ip_version', pa.int8()),
        ('rtt', pa.float32()),
    ])
    table = pa.Table.from_pylist(measurements, schema=schema)
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
    """Create single ArrayRecord entry."""
    time_span = (last_timestamp - first_timestamp).total_seconds() if last_timestamp != first_timestamp else 0.0
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
    """Write probe measurements to ArrayRecord, splitting if needed."""
    if not measurements:
        return 0

    src_id = hash_src_addr(src_addr)
    measurements_bytes = serialize_measurements_to_ipc(measurements)

    if len(measurements_bytes) <= max_size_bytes:
        entry = create_arrayrecord_entry(
            src_id=src_id,
            measurements_bytes=measurements_bytes,
            n_measurements=len(measurements),
            first_timestamp=measurements[0]['event_time'],
            last_timestamp=measurements[-1]['event_time'],
        )
        writer.write(entry)
        return 1

    # Split into multiple rows
    left = 0
    rows_written = 0

    while left < len(measurements):
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


def process_parquet_file_worker(args):
    """Worker to read parquet and extract probe groups to disk."""
    parquet_file, output_parquet, worker_id = args

    print(f"  Worker {worker_id}: Processing {parquet_file}")

    # Create DuckDB connection with memory limit
    con = duckdb.connect(':memory:')

    # Export grouped data to parquet (disk-based, no memory limit)
    con.execute(f"""
        COPY (
            SELECT
                src_addr,
                LIST(STRUCT_PACK(
                    event_time := event_time,
                    src_addr := src_addr,
                    dst_addr := dst_addr,
                    ip_version := ip_version,
                    rtt := rtt
                ) ORDER BY event_time) as measurements
            FROM read_parquet('{parquet_file}')
            GROUP BY src_addr
            ORDER BY src_addr
        ) TO '{output_parquet}' (FORMAT PARQUET)
    """)

    con.close()
    return output_parquet, worker_id


def process_probe_batch_from_parquet(args):
    """Worker to process probe batches from intermediate parquet files."""
    parquet_file, row_start, row_end, output_path, max_size_bytes, worker_id = args

    con = duckdb.connect(':memory:')

    # Read batch of rows
    probe_batch = con.execute(f"""
        SELECT src_addr, measurements
        FROM read_parquet('{parquet_file}')
        LIMIT {row_end - row_start} OFFSET {row_start}
    """).fetchall()

    con.close()

    # Process batch
    writer = array_record_module.ArrayRecordWriter(output_path, 'group_size:1')
    rows_written = 0
    measurements_total = 0

    for src_addr, measurements_struct_list in probe_batch:
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

        # Sort measurements by timestamp (in-memory, per-row sorting)
        # This is memory-efficient since we only sort one probe at a time
        measurements.sort(key=lambda m: m['event_time'])

        rows = write_probe_to_arrayrecord(writer, src_addr, measurements, max_size_bytes)
        rows_written += rows
        measurements_total += len(measurements)

    writer.close()
    print(f"  Worker {worker_id}: Completed batch [{row_start}:{row_end}] -> {rows_written} rows")
    return output_path, rows_written, measurements_total


def merge_arrayrecords(input_paths: List[str], output_path: str):
    """Merge multiple ArrayRecord files into one."""
    print(f"\n  Merging {len(input_paths)} partial files...")

    writer = array_record_module.ArrayRecordWriter(output_path, 'group_size:1')
    total_rows = 0

    for i, input_path in enumerate(input_paths):
        print(f"    Merging {i+1}/{len(input_paths)}: {Path(input_path).name}", end='\r')
        reader = array_record_module.ArrayRecordReader(input_path)
        n_records = reader.num_records()

        for j in range(n_records):
            record_bytes = reader.read([j])[0]
            writer.write(record_bytes)
            total_rows += 1

        reader.close()

    writer.close()
    print(f"\n    Merged {total_rows:,} rows into {Path(output_path).name}")


def process_parquet_to_probe_rows_streaming(
    input_pattern: str,
    output_dir: Path,
    max_row_size_mb: float = 8.0,
    train_ratio: float = 0.9,
    num_workers: int = None,
    memory_limit_gb: float = None,
    assume_ordered: bool = True,
):
    """
    Memory-efficient parallel processing with streaming and post-processing sort.

    Strategy:
    1. Use DuckDB to write grouped data to disk (avoids OOM)
    2. Stream batches from disk to workers
    3. Sort measurements in-memory per-row (memory-efficient)
    4. Parallel processing of batches
    5. Merge results

    Args:
        input_pattern: Glob pattern for input parquet files
        output_dir: Output directory for ArrayRecord files
        max_row_size_mb: Maximum row size in MB
        train_ratio: Ratio of probes for training set
        num_workers: Number of worker processes (default: CPU count)
        memory_limit_gb: DuckDB memory limit in GB (default: available - 1GB)
        assume_ordered: Only affects outer ordering of result by src_addr.
                       Does NOT affect measurement ordering (always sorted in post-processing).
                       Set to False to add ORDER BY src_addr in DuckDB query.

    IMPORTANT: Measurements are ALWAYS sorted by timestamp in post-processing.
    This happens in parallel workers on a per-row basis, which is extremely
    memory-efficient (only one probe's measurements in memory at a time).
    This avoids DuckDB OOM errors from trying to sort 200M+ rows.

    The assume_ordered parameter now only controls whether the GROUP BY result
    is ordered by src_addr (cosmetic - doesn't affect correctness).
    """
    if num_workers is None:
        num_workers = cpu_count()

    if memory_limit_gb is None:
        available_gb = get_available_memory_gb()
        memory_limit_gb = max(1.0, available_gb - 1.0)  # Leave 1GB for system

    output_dir.mkdir(parents=True, exist_ok=True)
    max_size_bytes = int(max_row_size_mb * 1024 * 1024)

    print(f"=" * 80)
    print(f"MEMORY-EFFICIENT PARALLEL PROBE ROW PREPROCESSING")
    print(f"=" * 80)
    print(f"Workers: {num_workers}")
    print(f"Memory limit: {memory_limit_gb:.1f}GB")
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")
    print()

    start_time = time.time()

    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix="probe_rows_"))
    print(f"Temp directory: {temp_dir}")

    try:
        # Step 1: Stream grouping to disk
        print("\nStep 1: Streaming GROUP BY to disk (memory-safe)...")

        con = duckdb.connect(':memory:')
        con.execute(f"SET memory_limit='{memory_limit_gb}GB'")

        # Enable streaming optimizations
        con.execute("SET preserve_insertion_order=true")
        con.execute("SET temp_directory='{}'".format(temp_dir))

        # Try to use external algorithms for large aggregations
        try:
            con.execute("SET force_external=true")
        except:
            pass  # Older DuckDB versions may not support this

        # Get list of input files
        import glob
        parquet_files = sorted(glob.glob(input_pattern))
        print(f"  Found {len(parquet_files)} parquet files")

        if not parquet_files:
            raise FileNotFoundError(f"No files match pattern: {input_pattern}")

        # Stream all data to intermediate parquet with grouping
        intermediate_parquet = str(temp_dir / "grouped_probes.parquet")
        print(f"  Writing grouped data to: {intermediate_parquet}")

        group_start = time.time()

        # Build query - measurements are now sorted in post-processing
        # Only the outer ORDER BY depends on assume_ordered parameter
        if assume_ordered:
            print("  Grouping without ORDER BY (measurements sorted in post-processing)")
            query = f"""
                COPY (
                    SELECT
                        src_addr,
                        LIST(STRUCT_PACK(
                            event_time := event_time,
                            src_addr := src_addr,
                            dst_addr := dst_addr,
                            ip_version := ip_version,
                            rtt := rtt
                        )) as measurements
                    FROM read_parquet('{input_pattern}')
                    GROUP BY src_addr
                ) TO '{intermediate_parquet}' (FORMAT PARQUET)
            """
        else:
            print("  Grouping with outer ORDER BY src_addr (measurements sorted in post-processing)")
            query = f"""
                COPY (
                    SELECT
                        src_addr,
                        LIST(STRUCT_PACK(
                            event_time := event_time,
                            src_addr := src_addr,
                            dst_addr := dst_addr,
                            ip_version := ip_version,
                            rtt := rtt
                        )) as measurements
                    FROM read_parquet('{input_pattern}')
                    GROUP BY src_addr
                    ORDER BY src_addr
                ) TO '{intermediate_parquet}' (FORMAT PARQUET)
            """

        con.execute(query)

        group_time = time.time() - group_start

        # Count probes
        n_probes = con.execute(f"SELECT COUNT(*) FROM read_parquet('{intermediate_parquet}')").fetchone()[0]
        total_measurements = con.execute(f"SELECT SUM(LENGTH(measurements)) FROM read_parquet('{intermediate_parquet}')").fetchone()[0]

        print(f"  Grouped {n_probes:,} probes in {group_time:.1f}s")
        print(f"  Total measurements: {total_measurements:,}")

        con.close()

        # Step 2: Split train/test
        print("\nStep 2: Splitting train/test...")
        n_train = int(n_probes * train_ratio)
        print(f"  Train: {n_train:,} probes")
        print(f"  Test: {n_probes - n_train:,} probes")

        # Step 3: Batch processing in parallel
        print(f"\nStep 3: Parallel batch processing...")

        # Calculate batch sizes
        train_batch_size = max(100, n_train // (num_workers * 4))  # 4 batches per worker
        test_batch_size = max(100, (n_probes - n_train) // (num_workers * 4))

        # Create train batches
        train_batches = []
        for i in range(0, n_train, train_batch_size):
            end = min(i + train_batch_size, n_train)
            output_path = str(temp_dir / f"train_part_{len(train_batches)}.arrayrecord")
            train_batches.append((
                intermediate_parquet,
                i,
                end,
                output_path,
                max_size_bytes,
                len(train_batches)
            ))

        # Create test batches
        test_batches = []
        for i in range(n_train, n_probes, test_batch_size):
            end = min(i + test_batch_size, n_probes)
            output_path = str(temp_dir / f"test_part_{len(test_batches)}.arrayrecord")
            test_batches.append((
                intermediate_parquet,
                i,
                end,
                output_path,
                max_size_bytes,
                len(test_batches)
            ))

        print(f"  Train batches: {len(train_batches)}")
        print(f"  Test batches: {len(test_batches)}")

        # Process train batches
        print(f"\n  Processing TRAIN batches...")
        process_start = time.time()

        with Pool(num_workers) as pool:
            train_results = pool.map(process_probe_batch_from_parquet, train_batches)

        train_time = time.time() - process_start
        print(f"\n  Train processing: {train_time:.1f}s")

        # Process test batches
        if test_batches:
            print(f"\n  Processing TEST batches...")
            test_start = time.time()

            with Pool(num_workers) as pool:
                test_results = pool.map(process_probe_batch_from_parquet, test_batches)

            test_time = time.time() - test_start
            print(f"\n  Test processing: {test_time:.1f}s")
        else:
            test_results = []

        # Step 4: Merge
        print("\nStep 4: Merging results...")

        train_partial_files = [r[0] for r in train_results]
        train_output = str(output_dir / "train.arrayrecord")
        merge_arrayrecords(train_partial_files, train_output)

        if test_results:
            test_partial_files = [r[0] for r in test_results]
            test_output = str(output_dir / "test.arrayrecord")
            merge_arrayrecords(test_partial_files, test_output)

        # Calculate stats
        train_rows = sum(r[1] for r in train_results)
        train_measurements = sum(r[2] for r in train_results)
        test_rows = sum(r[1] for r in test_results) if test_results else 0
        test_measurements = sum(r[2] for r in test_results) if test_results else 0

        total_time = time.time() - start_time

        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)

        print(f"\nPerformance:")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"  Streaming GROUP BY: {group_time:.1f}s")
        print(f"  Train processing: {train_time:.1f}s")
        if test_results:
            print(f"  Test processing: {test_time:.1f}s")
        print(f"  Throughput: {total_measurements/total_time:,.0f} measurements/sec")

        print(f"\nTrain set:")
        print(f"  Rows: {train_rows:,}")
        print(f"  Measurements: {train_measurements:,}")
        print(f"  Avg measurements/row: {train_measurements/train_rows:.1f}")
        print(f"  Output: {train_output}")

        if test_results:
            print(f"\nTest set:")
            print(f"  Rows: {test_rows:,}")
            print(f"  Measurements: {test_measurements:,}")
            print(f"  Avg measurements/row: {test_measurements/test_rows:.1f}")
            print(f"  Output: {test_output}")

    finally:
        # Cleanup temp directory
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient parallelized Parquet to PLAN_3 ArrayRecord conversion"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern for input parquet files"
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
        help="Train/test split ratio (default: 0.9)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=None,
        help="DuckDB memory limit in GB (default: available - 1GB)"
    )
    parser.add_argument(
        "--no-assume-ordered",
        action="store_true",
        help="Do NOT assume data is pre-ordered (adds ORDER BY, slower but safer)"
    )

    args = parser.parse_args()

    process_parquet_to_probe_rows_streaming(
        input_pattern=args.input,
        output_dir=Path(args.output),
        max_row_size_mb=args.max_row_size_mb,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        memory_limit_gb=args.memory_limit_gb,
        assume_ordered=not args.no_assume_ordered,
    )


if __name__ == "__main__":
    main()
