#!/usr/bin/env python3
"""
Memory-efficient probe row preprocessing: per-file GROUP BY + hash partition + bucket merge.

Three-pass architecture:
  Pass 1: Per-file GROUP BY (parallel workers, each processes one parquet file)
  Pass 2: Hash-partition intermediates into N bucket parquet files (~2GB RAM)
  Pass 3: Per-bucket merge by src_addr + sort by event_time -> write to ArrayRecord (~5GB RAM)

Train/test split: hash_src_addr(src_addr) % 10 < 9 -> train (~90/10 deterministic split).

This replaces the OOM-prone DuckDB ORDER BY approach with a hash-partition strategy
that keeps peak memory under 10GB.
"""

import argparse
import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc
from pathlib import Path
from typing import List
import glob
import hashlib
from multiprocessing import Pool, cpu_count
import time
import tempfile
import shutil
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
    """Serialize list of measurement dicts to PyArrow IPC format.

    Used by the training pipeline datasource. Kept for compatibility.
    """
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


def serialize_arrow_to_ipc(struct_array: pa.StructArray) -> bytes:
    """Serialize an Arrow StructArray directly to IPC bytes.

    Avoids round-tripping through Python dicts. Used by the streaming path.
    """
    table = pa.Table.from_arrays(
        [struct_array.field(i) for i in range(struct_array.type.num_fields)],
        names=[struct_array.type.field(i).name for i in range(struct_array.type.num_fields)],
    )
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
    """Create single ArrayRecord entry.

    first_timestamp/last_timestamp can be either Python datetime objects
    or PyArrow scalars (will be converted via .as_py() if needed).
    """
    # Convert Arrow scalars to Python datetime if needed
    if hasattr(first_timestamp, 'as_py'):
        first_timestamp = first_timestamp.as_py()
    if hasattr(last_timestamp, 'as_py'):
        last_timestamp = last_timestamp.as_py()

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


def write_probe_arrow(
    writer,
    src_addr: str,
    sorted_struct_array: pa.StructArray,
    max_size_bytes: int,
) -> tuple:
    """Write probe measurements (as Arrow StructArray) to ArrayRecord, splitting if needed.

    Returns (rows_written, n_measurements).
    """
    n = len(sorted_struct_array)
    if n == 0:
        return 0, 0

    src_id = hash_src_addr(src_addr)

    # Get event_time field for timestamps
    event_times = sorted_struct_array.field('event_time')

    measurements_bytes = serialize_arrow_to_ipc(sorted_struct_array)

    if len(measurements_bytes) <= max_size_bytes:
        entry = create_arrayrecord_entry(
            src_id=src_id,
            measurements_bytes=measurements_bytes,
            n_measurements=n,
            first_timestamp=event_times[0],
            last_timestamp=event_times[n - 1],
        )
        writer.write(entry)
        return 1, n

    # Split into multiple rows via binary search
    left = 0
    rows_written = 0

    while left < n:
        lo, hi = 1, n - left
        best_size = 1

        while lo <= hi:
            mid = (lo + hi) // 2
            chunk = sorted_struct_array.slice(left, mid)
            chunk_bytes = serialize_arrow_to_ipc(chunk)

            if len(chunk_bytes) <= max_size_bytes:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1

        chunk = sorted_struct_array.slice(left, best_size)
        chunk_bytes = serialize_arrow_to_ipc(chunk)
        chunk_times = event_times.slice(left, best_size)
        entry = create_arrayrecord_entry(
            src_id=src_id,
            measurements_bytes=chunk_bytes,
            n_measurements=best_size,
            first_timestamp=chunk_times[0],
            last_timestamp=chunk_times[best_size - 1],
        )
        writer.write(entry)
        rows_written += 1
        left += best_size

    return rows_written, n


def process_parquet_file_worker(args):
    """Worker: GROUP BY src_addr for a single parquet file."""
    parquet_file, output_parquet, worker_id, worker_memory_gb, temp_dir = args

    print(f"  Worker {worker_id}: Processing {Path(parquet_file).name}")

    # Each worker gets its own temp directory to avoid DuckDB temp file collisions
    worker_temp = Path(temp_dir) / f"worker_{worker_id}"
    worker_temp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(':memory:')
    con.execute(f"SET memory_limit='{worker_memory_gb}GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{worker_temp}'")
    con.execute("SET debug_force_external=true")
    con.execute("SET enable_external_file_cache=false")

    con.execute(f"""
        COPY (
            SELECT
                src_addr,
                LIST(STRUCT_PACK(
                    event_time := event_time,
                    src_addr := src_addr,
                    dst_addr := dst_addr,
                    ip_version := ip_version,
                    rtt := rtt_avg
                ) ORDER BY event_time) as measurements
            FROM read_parquet('{parquet_file}')
            GROUP BY src_addr
        ) TO '{output_parquet}' (FORMAT PARQUET)
    """)

    con.close()
    return output_parquet, worker_id


def _hash_partition_intermediates(
    intermediate_dir: Path,
    bucket_dir: Path,
    n_buckets: int,
) -> None:
    """Pass 2: Hash-partition intermediate files into N bucket parquet files.

    Reads each of the ~720 intermediate grouped parquet files one at a time,
    assigns each row to a bucket via hash_src_addr(src_addr) % n_buckets,
    and appends to the corresponding bucket ParquetWriter.

    Peak memory: ~2GB (one intermediate file loaded at a time).
    """
    import pyarrow.parquet as pq

    print("\n  Pass 2: Hash-partitioning intermediates into buckets...")
    bucket_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip if all bucket files exist and are non-empty
    existing_buckets = [
        bucket_dir / f"bucket_{b:04d}.parquet"
        for b in range(n_buckets)
        if (bucket_dir / f"bucket_{b:04d}.parquet").exists()
        and (bucket_dir / f"bucket_{b:04d}.parquet").stat().st_size > 0
    ]
    if len(existing_buckets) == n_buckets:
        print(f"    RESUMING: Found {n_buckets} existing bucket files, skipping Pass 2")
        return

    intermediate_files = sorted(glob.glob(str(intermediate_dir / "grouped_*.parquet")))
    print(f"    Input: {len(intermediate_files)} intermediate files")
    print(f"    Output: {n_buckets} bucket files in {bucket_dir}")

    # Open N bucket writers
    # We'll determine the schema from the first file
    first_table = pq.read_table(intermediate_files[0])
    schema = first_table.schema
    del first_table

    bucket_paths = [str(bucket_dir / f"bucket_{b:04d}.parquet") for b in range(n_buckets)]
    writers = [pq.ParquetWriter(p, schema) for p in bucket_paths]

    partition_start = time.time()

    for file_idx, intermediate_file in enumerate(intermediate_files):
        table = pq.read_table(intermediate_file)

        # Drop rows with null src_addr
        null_mask = pc.is_null(table.column('src_addr'))
        n_nulls = pc.sum(null_mask).as_py()
        if n_nulls:
            table = table.filter(pc.invert(null_mask))

        src_addrs = table.column('src_addr')

        # Compute bucket assignment for each row
        bucket_ids = pa.array(
            [hash_src_addr(s.as_py()) % n_buckets for s in src_addrs],
            type=pa.int32(),
        )

        # Group rows by bucket and write
        for b in range(n_buckets):
            mask = pc.equal(bucket_ids, b)
            bucket_table = table.filter(mask)
            if len(bucket_table) > 0:
                writers[b].write_table(bucket_table)

        if (file_idx + 1) % 50 == 0 or file_idx == len(intermediate_files) - 1:
            elapsed = time.time() - partition_start
            print(f"    Partitioned {file_idx + 1}/{len(intermediate_files)} files "
                  f"({elapsed:.0f}s)")

    for w in writers:
        w.close()

    partition_time = time.time() - partition_start
    total_size_gb = sum(Path(p).stat().st_size for p in bucket_paths) / (1024 ** 3)
    print(f"    Partitioning complete: {partition_time:.0f}s, {total_size_gb:.1f}GB total")


def _process_bucket(
    bucket_path: str,
    train_writer,
    test_writer,
    max_size_bytes: int,
) -> dict:
    """Process a single bucket: group by src_addr, sort, write to ArrayRecord.

    Returns dict with per-bucket statistics.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(bucket_path)
    # Combine chunks into single arrays (read_table returns ChunkedArrays)
    src_addrs = table.column('src_addr').combine_chunks()
    meas_col = table.column('measurements').combine_chunks()

    # Group rows by src_addr using a Python dict
    # Each intermediate row has src_addr + measurements (list of structs)
    probe_chunks = {}  # src_addr -> list of StructArray chunks
    flat_values = meas_col.values
    list_offsets = meas_col.offsets

    for i in range(len(table)):
        src = src_addrs[i].as_py()
        if src is None:
            continue
        start = list_offsets[i].as_py()
        end = list_offsets[i + 1].as_py()
        chunk = flat_values.slice(start, end - start)

        if src in probe_chunks:
            probe_chunks[src].append(chunk)
        else:
            probe_chunks[src] = [chunk]

    del table, src_addrs, meas_col, flat_values, list_offsets

    stats = {
        'probes': 0,
        'train_rows': 0, 'train_measurements': 0,
        'test_rows': 0, 'test_measurements': 0,
    }

    for src_addr, chunks in probe_chunks.items():
        # Concatenate all measurement chunks
        if len(chunks) == 1:
            merged = chunks[0]
        else:
            merged = pa.concat_arrays(chunks)

        # Sort by event_time
        sort_indices = pc.sort_indices(merged, sort_keys=[('event_time', 'ascending')])
        sorted_measurements = merged.take(sort_indices)

        # Train/test split: deterministic hash-based
        is_train = hash_src_addr(src_addr) % 10 < 9
        writer = train_writer if is_train else test_writer

        rows, n_meas = write_probe_arrow(writer, src_addr, sorted_measurements, max_size_bytes)

        if is_train:
            stats['train_rows'] += rows
            stats['train_measurements'] += n_meas
        else:
            stats['test_rows'] += rows
            stats['test_measurements'] += n_meas

        stats['probes'] += 1

    return stats


def _process_all_buckets(
    bucket_dir: Path,
    output_dir: Path,
    n_buckets: int,
    max_size_bytes: int,
) -> dict:
    """Pass 3: Process all buckets sequentially, writing to shared train/test ArrayRecord.

    Returns dict with aggregate statistics.
    """
    print("\n  Pass 3: Processing buckets -> ArrayRecord...")

    train_output = str(output_dir / "train.arrayrecord")
    test_output = str(output_dir / "test.arrayrecord")
    train_writer = array_record_module.ArrayRecordWriter(train_output, 'group_size:1')
    test_writer = array_record_module.ArrayRecordWriter(test_output, 'group_size:1')

    totals = {
        'probes': 0,
        'train_rows': 0, 'train_measurements': 0,
        'test_rows': 0, 'test_measurements': 0,
    }

    process_start = time.time()

    for b in range(n_buckets):
        bucket_path = str(bucket_dir / f"bucket_{b:04d}.parquet")
        stats = _process_bucket(bucket_path, train_writer, test_writer, max_size_bytes)

        for k in totals:
            totals[k] += stats[k]

        if (b + 1) % 10 == 0 or b == n_buckets - 1:
            elapsed = time.time() - process_start
            rate = totals['probes'] / elapsed if elapsed > 0 else 0
            total_meas = totals['train_measurements'] + totals['test_measurements']
            print(f"    Bucket {b + 1}/{n_buckets}: {totals['probes']:,} probes, "
                  f"{total_meas:,} measurements ({rate:.0f} probes/s)")

    train_writer.close()
    test_writer.close()

    process_time = time.time() - process_start
    print(f"    Bucket processing complete: {totals['probes']:,} probes in {process_time:.0f}s")

    totals['process_time'] = process_time
    totals['train_output'] = train_output
    totals['test_output'] = test_output
    return totals


def process_parquet_to_probe_rows_streaming(
    input_pattern: str,
    output_dir: Path,
    max_row_size_mb: float = 8.0,
    num_workers: int = None,
    n_buckets: int = 100,
    temp_dir_path: str = None,
):
    """
    Three-pass probe row preprocessing:

    Pass 1: Per-file GROUP BY (parallel workers, memory-safe)
    Pass 2: Hash-partition intermediates into N bucket files (~2GB RAM)
    Pass 3: Per-bucket merge + sort -> write to train/test ArrayRecord (~5GB RAM)

    Train/test split: hash_src_addr(src_addr) % 10 < 9 -> train (~90/10).

    Args:
        input_pattern: Glob pattern for input parquet files
        output_dir: Output directory for ArrayRecord files
        max_row_size_mb: Maximum row size in MB
        num_workers: Number of worker processes for Pass 1 (default: CPU count)
        n_buckets: Number of hash buckets for partitioning (default: 100)
        temp_dir_path: Temp directory for intermediate files (enables resume on failure)
    """
    if num_workers is None:
        num_workers = cpu_count()

    output_dir.mkdir(parents=True, exist_ok=True)
    max_size_bytes = int(max_row_size_mb * 1024 * 1024)

    print("=" * 80)
    print("PROBE ROW PREPROCESSING (hash-partition)")
    print("=" * 80)
    print(f"Workers: {num_workers}")
    print(f"Buckets: {n_buckets}")
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")
    print()

    start_time = time.time()

    # Create or reuse temp directory for intermediate files
    if temp_dir_path:
        temp_dir = Path(temp_dir_path)
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="probe_rows_"))
    print(f"Temp directory: {temp_dir}")

    success = False
    try:
        # ============================================================
        # Pass 1: Per-file GROUP BY (parallel, memory-safe)
        # ============================================================
        print("\nPass 1: Per-file GROUP BY (parallel)...")

        parquet_files = sorted(glob.glob(input_pattern))
        print(f"  Found {len(parquet_files)} parquet files")

        if not parquet_files:
            raise FileNotFoundError(f"No files match pattern: {input_pattern}")

        worker_memory_gb = 8
        intermediate_dir = temp_dir / "per_file_grouped"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Check if Pass 1 already completed (resume support)
        existing_grouped = sorted(glob.glob(str(intermediate_dir / "grouped_*.parquet")))
        existing_nonzero = [f for f in existing_grouped if Path(f).stat().st_size > 0]

        if len(existing_nonzero) == len(parquet_files):
            print(f"  RESUMING: Found {len(existing_nonzero)} existing grouped files, skipping Pass 1")
        else:
            if existing_nonzero:
                print(f"  Found {len(existing_nonzero)}/{len(parquet_files)} existing files (incomplete), re-running Pass 1")

            worker_args = [
                (
                    str(pf),
                    str(intermediate_dir / f"grouped_{i:04d}.parquet"),
                    i,
                    worker_memory_gb,
                    str(temp_dir),
                )
                for i, pf in enumerate(parquet_files)
            ]

            print(f"  Workers: {num_workers}, memory per worker: {worker_memory_gb}GB")
            print(f"  Peak memory estimate: {num_workers * worker_memory_gb}GB")

            group_start = time.time()

            with Pool(num_workers) as pool:
                pool.map(process_parquet_file_worker, worker_args)

            per_file_time = time.time() - group_start
            print(f"  Per-file GROUP BY complete: {len(parquet_files)} files in {per_file_time:.1f}s")

        # ============================================================
        # Pass 2: Hash-partition into buckets
        # ============================================================
        bucket_dir = temp_dir / "buckets"
        _hash_partition_intermediates(intermediate_dir, bucket_dir, n_buckets)

        # ============================================================
        # Pass 3: Process buckets -> ArrayRecord
        # ============================================================
        stats = _process_all_buckets(bucket_dir, output_dir, n_buckets, max_size_bytes)

        total_time = time.time() - start_time
        total_measurements = stats['train_measurements'] + stats['test_measurements']

        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)

        print(f"\nPerformance:")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"  Bucket processing: {stats['process_time']:.1f}s")
        if total_time > 0:
            print(f"  Throughput: {total_measurements/total_time:,.0f} measurements/sec")

        print(f"\nTrain set:")
        print(f"  ArrayRecord rows: {stats['train_rows']:,}")
        print(f"  Measurements: {stats['train_measurements']:,}")
        if stats['train_rows'] > 0:
            print(f"  Avg measurements/row: {stats['train_measurements']/stats['train_rows']:.1f}")
        print(f"  Output: {stats['train_output']}")

        print(f"\nTest set:")
        print(f"  ArrayRecord rows: {stats['test_rows']:,}")
        print(f"  Measurements: {stats['test_measurements']:,}")
        if stats['test_rows'] > 0:
            print(f"  Avg measurements/row: {stats['test_measurements']/stats['test_rows']:.1f}")
        print(f"  Output: {stats['test_output']}")

        print(f"\nTotal probes: {stats['probes']:,}")

        success = True

    finally:
        if success:
            print(f"\nCleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\nKEEPING temp directory for resume: {temp_dir}")
            print(f"  Re-run with --temp-dir '{temp_dir}' to resume")


def main():
    parser = argparse.ArgumentParser(
        description="Probe row preprocessing: per-file GROUP BY + hash partition + bucket merge to ArrayRecord"
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
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for Pass 1 (default: CPU count)"
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=100,
        help="Number of hash buckets for partitioning (default: 100)"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temp directory for intermediate files (enables resume on failure)"
    )

    args = parser.parse_args()

    process_parquet_to_probe_rows_streaming(
        input_pattern=args.input,
        output_dir=Path(args.output),
        max_row_size_mb=args.max_row_size_mb,
        num_workers=args.workers,
        n_buckets=args.n_buckets,
        temp_dir_path=args.temp_dir,
    )


if __name__ == "__main__":
    main()
