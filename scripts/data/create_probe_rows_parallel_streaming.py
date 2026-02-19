#!/usr/bin/env python3
"""
Memory-efficient probe row preprocessing: per-file GROUP BY + direct streaming to ArrayRecord.

Two-pass architecture:
  Pass 1: Per-file GROUP BY (parallel workers, each processes one parquet file)
  Pass 2: DuckDB ORDER BY src_addr (external sort) -> stream Arrow batches ->
           accumulate per src_addr -> sort by event_time -> write to ArrayRecord

This eliminates the OOM-prone FLATTEN(LIST(measurements)) merge step and the
intermediate merged parquet file. Only one probe's measurements (~57MB max)
are held in memory at a time during Pass 2.
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
    try:
        con.execute("SET force_external=true")
    except:
        pass

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


def _stream_to_arrayrecord(
    intermediate_glob: str,
    output_dir: Path,
    train_ratio: float,
    max_size_bytes: int,
    memory_limit_gb: float,
    temp_dir: str,
) -> dict:
    """Stream sorted intermediate parquets directly to train/test ArrayRecord files.

    Replaces the old merge-to-parquet + batch-processing + merge-arrayrecord pipeline.

    1. Count distinct probes for train/test split boundary
    2. DuckDB ORDER BY src_addr (external merge sort)
    3. Stream Arrow batches, accumulate per src_addr
    4. Sort each probe's measurements by event_time
    5. Write directly to train or test ArrayRecord

    Returns dict with statistics.
    """
    print("\n  Step 2: Streaming directly to ArrayRecord...")

    # --- Count probes for train/test split ---
    con = duckdb.connect(':memory:')
    con.execute(f"SET memory_limit='{memory_limit_gb}GB'")
    con.execute(f"SET temp_directory='{temp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=8")
    try:
        con.execute("SET force_external=true")
    except:
        pass

    n_probes = con.execute(f"""
        SELECT COUNT(DISTINCT src_addr) FROM read_parquet('{intermediate_glob}')
    """).fetchone()[0]

    n_train = int(n_probes * train_ratio)
    n_test = n_probes - n_train
    print(f"    Probes: {n_probes:,} (train: {n_train:,}, test: {n_test:,})")

    # --- DuckDB ORDER BY (external sort) ---
    print(f"    Running DuckDB ORDER BY src_addr (external sort)...")
    sort_start = time.time()

    result = con.execute(f"""
        SELECT src_addr, measurements
        FROM read_parquet('{intermediate_glob}')
        ORDER BY src_addr
    """)

    sort_time = time.time() - sort_start
    print(f"    Sort query submitted in {sort_time:.1f}s")

    # --- Stream Arrow batches ---
    reader = result.fetch_arrow_reader(batch_size=4096)

    train_output = str(output_dir / "train.arrayrecord")
    test_output = str(output_dir / "test.arrayrecord")
    train_writer = array_record_module.ArrayRecordWriter(train_output, 'group_size:1')
    test_writer = array_record_module.ArrayRecordWriter(test_output, 'group_size:1')

    current_src = None
    current_chunks = []  # List of Arrow StructArray slices
    probes_processed = 0
    total_rows_written = 0
    total_measurements = 0
    train_rows = 0
    train_measurements = 0
    test_rows = 0
    test_measurements = 0

    stream_start = time.time()

    def emit_probe():
        nonlocal probes_processed, total_rows_written, total_measurements
        nonlocal train_rows, train_measurements, test_rows, test_measurements
        if current_src is None:
            return

        # Concatenate all measurement chunks into one StructArray
        if len(current_chunks) == 1:
            merged = current_chunks[0]
        else:
            merged = pa.concat_arrays(current_chunks)

        # Sort by event_time using Arrow compute
        sort_indices = pc.sort_indices(merged, sort_keys=[('event_time', 'ascending')])
        sorted_measurements = merged.take(sort_indices)

        # Choose writer based on probe index
        if probes_processed < n_train:
            active_writer = train_writer
        else:
            active_writer = test_writer

        rows, n_meas = write_probe_arrow(active_writer, current_src, sorted_measurements, max_size_bytes)

        if probes_processed < n_train:
            train_rows += rows
            train_measurements += n_meas
        else:
            test_rows += rows
            test_measurements += n_meas

        total_rows_written += rows
        total_measurements += n_meas
        probes_processed += 1

        if probes_processed % 1000 == 0:
            elapsed = time.time() - stream_start
            rate = probes_processed / elapsed if elapsed > 0 else 0
            print(f"    Processed {probes_processed:,}/{n_probes:,} probes "
                  f"({total_measurements:,.0f} measurements, {rate:.0f} probes/s)    ", end='\r')

    for batch in reader:
        src_col = batch.column('src_addr')
        meas_col = batch.column('measurements')
        flat_values = meas_col.values
        list_offsets = meas_col.offsets

        for i in range(len(batch)):
            src = src_col[i].as_py()
            start = list_offsets[i].as_py()
            end = list_offsets[i + 1].as_py()
            meas_slice = flat_values.slice(start, end - start)

            if src != current_src:
                emit_probe()
                current_src = src
                current_chunks = [meas_slice]
            else:
                current_chunks.append(meas_slice)

    emit_probe()  # Last probe

    train_writer.close()
    test_writer.close()
    con.close()

    stream_time = time.time() - stream_start
    print(f"\n    Streaming complete: {probes_processed:,} probes in {stream_time:.1f}s")

    return {
        'n_probes': probes_processed,
        'n_train': n_train,
        'n_test': n_test,
        'train_rows': train_rows,
        'train_measurements': train_measurements,
        'test_rows': test_rows,
        'test_measurements': test_measurements,
        'total_rows': total_rows_written,
        'total_measurements': total_measurements,
        'stream_time': stream_time,
        'train_output': train_output,
        'test_output': test_output,
    }


def process_parquet_to_probe_rows_streaming(
    input_pattern: str,
    output_dir: Path,
    max_row_size_mb: float = 8.0,
    train_ratio: float = 0.9,
    num_workers: int = None,
    memory_limit_gb: float = None,
    temp_dir_path: str = None,
):
    """
    Two-pass probe row preprocessing: per-file GROUP BY + direct streaming to ArrayRecord.

    Pass 1: Per-file GROUP BY (parallel workers, memory-safe)
    Pass 2: DuckDB ORDER BY -> stream -> accumulate -> sort -> write ArrayRecord

    Args:
        input_pattern: Glob pattern for input parquet files
        output_dir: Output directory for ArrayRecord files
        max_row_size_mb: Maximum row size in MB
        train_ratio: Ratio of probes for training set
        num_workers: Number of worker processes (default: CPU count)
        memory_limit_gb: DuckDB memory limit in GB (default: available - 1GB)
        temp_dir_path: Temp directory for intermediate files (enables resume on failure)
    """
    if num_workers is None:
        num_workers = cpu_count()

    if memory_limit_gb is None:
        available_gb = get_available_memory_gb()
        memory_limit_gb = max(1.0, available_gb - 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    max_size_bytes = int(max_row_size_mb * 1024 * 1024)

    print(f"=" * 80)
    print(f"PROBE ROW PREPROCESSING (direct streaming)")
    print(f"=" * 80)
    print(f"Workers: {num_workers}")
    print(f"Memory limit: {memory_limit_gb:.1f}GB")
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
            per_file_time = 0.0
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
        # Pass 2: Stream sorted data directly to ArrayRecord
        # ============================================================
        intermediate_glob = str(intermediate_dir / "grouped_*.parquet")

        stats = _stream_to_arrayrecord(
            intermediate_glob=intermediate_glob,
            output_dir=output_dir,
            train_ratio=train_ratio,
            max_size_bytes=max_size_bytes,
            memory_limit_gb=memory_limit_gb,
            temp_dir=str(temp_dir),
        )

        total_time = time.time() - start_time

        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)

        print(f"\nPerformance:")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"  Streaming: {stats['stream_time']:.1f}s")
        print(f"  Throughput: {stats['total_measurements']/total_time:,.0f} measurements/sec")

        print(f"\nTrain set:")
        print(f"  Probes: {stats['n_train']:,}")
        print(f"  ArrayRecord rows: {stats['train_rows']:,}")
        print(f"  Measurements: {stats['train_measurements']:,}")
        if stats['train_rows'] > 0:
            print(f"  Avg measurements/row: {stats['train_measurements']/stats['train_rows']:.1f}")
        print(f"  Output: {stats['train_output']}")

        print(f"\nTest set:")
        print(f"  Probes: {stats['n_test']:,}")
        print(f"  ArrayRecord rows: {stats['test_rows']:,}")
        print(f"  Measurements: {stats['test_measurements']:,}")
        if stats['test_rows'] > 0:
            print(f"  Avg measurements/row: {stats['test_measurements']/stats['test_rows']:.1f}")
        print(f"  Output: {stats['test_output']}")

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
        description="Probe row preprocessing: per-file GROUP BY + direct streaming to ArrayRecord"
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
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        memory_limit_gb=args.memory_limit_gb,
        temp_dir_path=args.temp_dir,
    )


if __name__ == "__main__":
    main()
