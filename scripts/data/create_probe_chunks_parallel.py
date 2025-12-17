#!/usr/bin/env python3
"""
Create probe-centric measurement chunks with parallel processing (DATA_LOADING_PLAN_1.md).

OPTIMIZED VERSION:
- Uses DuckDB for parallel partitioning (GROUP BY uses all cores)
- Uses multiprocessing for parallel tokenization
- 20-50x faster than sequential version

This script:
1. Uses DuckDB to partition by (src_addr, 5-minute bucket) in parallel
2. Tokenizes partitions in parallel using multiprocessing.Pool
3. Writes to ArrayRecord format for fast Grain access

Expected time: ~45-90 minutes for 200M rows on 48 cores (vs 3-4 hours sequential)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import struct
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenization import encode_measurement, VOCAB_SIZE


def tokenize_partition_chunk(
    partition_data: tuple, max_tokens_per_record: int = 100000
) -> list[dict]:
    """
    Tokenize a single partition (one probe's 5-minute bucket).

    This function is called in parallel by multiprocessing.Pool.
    Serialization happens HERE (in worker process) for better parallelism.

    Args:
        partition_data: Tuple of (src_id, bucket_start, measurements_list)
        max_tokens_per_record: Maximum tokens per chunk before splitting

    Returns:
        List of chunk dictionaries with serialized tokens/offsets
    """
    src_id, bucket_start, measurements = partition_data

    chunks = []
    current_tokens = []
    current_meas_offsets = []
    prev_timestamp = None
    part_id = 0

    for meas in measurements:
        # Track start offset of this measurement
        meas_start_offset = len(current_tokens)

        # For split chunks (part_id > 0), reset to absolute timestamp
        if part_id > 0 and prev_timestamp is not None:
            prev_timestamp = None

        # Encode measurement with delta timestamps
        meas_tokens = encode_measurement(
            meas,
            prev_timestamp=prev_timestamp,
            include_timestamp=True,  # Always include timestamps in chunks
        )

        # Check if adding this measurement would exceed limit
        if (
            len(current_tokens) + len(meas_tokens) > max_tokens_per_record
            and current_tokens
        ):
            # Flush current chunk with serialization (in worker process!)
            chunks.append(
                {
                    "src_id": src_id,
                    "bucket_start_time": bucket_start,
                    "bucket_duration_s": 300,
                    "part_id": part_id,
                    "tokens": serialize_tokens(current_tokens),  # Serialize in worker
                    "meas_offsets": serialize_offsets(
                        current_meas_offsets
                    ),  # Serialize in worker
                    "n_tokens": len(current_tokens),
                    "n_measurements": len(current_meas_offsets),
                }
            )

            # Start new chunk
            current_tokens = []
            current_meas_offsets = []
            part_id += 1
            prev_timestamp = None  # Reset for split boundary
            meas_start_offset = 0

        # Add measurement to current chunk
        current_meas_offsets.append(meas_start_offset)
        current_tokens.extend(meas_tokens)
        prev_timestamp = meas["event_time"]

    # Flush final chunk with serialization (in worker process!)
    if current_tokens:
        chunks.append(
            {
                "src_id": src_id,
                "bucket_start_time": bucket_start,
                "bucket_duration_s": 300,
                "part_id": part_id,
                "tokens": serialize_tokens(current_tokens),  # Serialize in worker
                "meas_offsets": serialize_offsets(
                    current_meas_offsets
                ),  # Serialize in worker
                "n_tokens": len(current_tokens),
                "n_measurements": len(current_meas_offsets),
            }
        )

    return chunks


def serialize_tokens(tokens: list[int]) -> bytes:
    """Serialize token list as uint16 array."""
    assert all(0 <= t < VOCAB_SIZE for t in tokens), f"Invalid token IDs found"
    return struct.pack(f"{len(tokens)}H", *tokens)


def serialize_offsets(offsets: list[int]) -> bytes:
    """Serialize offset list as int32 array."""
    return struct.pack(f"{len(offsets)}i", *offsets)


def _create_arrayrecord_writer(output_file: Path):
    """Create an ArrayRecord writer, raising a clear error if the dependency is missing."""
    try:
        import array_record.python.array_record_module as array_record_module
    except ImportError:
        print(
            "ERROR: array_record not installed. Install with: pip install array_record"
        )
        sys.exit(1)

    return array_record_module.ArrayRecordWriter(str(output_file), "group_size:1")


def _write_chunk_record(writer, chunk: dict, schema: pa.Schema):
    """Serialize and write a single chunk to an ArrayRecord writer."""
    # Convert to a single-row RecordBatch
    batch = pa.Table.from_pylist([chunk], schema=schema).to_batches()[0]

    # Serialize to IPC bytes
    sink = pa.BufferOutputStream()
    writer_ipc = pa.ipc.new_stream(sink, batch.schema)
    writer_ipc.write_batch(batch)
    writer_ipc.close()
    record_bytes = sink.getvalue().to_pybytes()

    writer.write(record_bytes)


def create_probe_chunks_parallel(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.9,
    max_tokens_per_record: int = 100000,
    num_workers: int = None,
):
    """
    Create probe-centric chunks using DuckDB for parallel partitioning.

    Args:
        input_file: Path to input parquet file
        output_dir: Output directory for ArrayRecord files
        train_ratio: Fraction of probes for training (default: 0.9)
        max_tokens_per_record: Maximum tokens per chunk before splitting
        num_workers: Number of parallel workers (default: all CPU cores)
    """
    import duckdb

    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores for system

    # Report CPU info to make it obvious how many cores are available/used.
    affinity_count = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    print("Creating probe-centric chunks with parallel processing")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(
        f"Detected CPUs: cpu_count={cpu_count()}",
        end="",
    )
    if affinity_count is not None:
        print(f", affinity={affinity_count}")
    else:
        print()
    print(f"Workers: {num_workers} parallel workers")
    print(f"Max tokens per chunk: {max_tokens_per_record:,}")
    print()

    # Step 1: Use DuckDB for parallel partitioning and grouping
    print("Step 1: Partitioning by (src_addr, 5-minute bucket) with DuckDB...")
    print("(DuckDB will use all CPU cores automatically)")
    sys.stdout.flush()

    con = duckdb.connect()

    # Enable parallelism and progress bar
    con.execute(f"SET threads TO {num_workers}")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET enable_progress_bar_print = true")

    print(f"DuckDB configured with {num_workers} threads")
    print("Starting query execution (streaming mode to avoid OOM)...")
    sys.stdout.flush()

    # First, get probe IDs mapping
    print("Step 1a: Creating probe ID mapping...")
    sys.stdout.flush()

    probe_query = """
    SELECT src_addr, ROW_NUMBER() OVER (ORDER BY src_addr) - 1 as src_id
    FROM (SELECT DISTINCT src_addr FROM read_parquet(?) WHERE src_addr IS NOT NULL)
    ORDER BY src_addr
    """
    probe_result = con.execute(probe_query, [input_file]).fetchall()
    src_addr_to_id = {row[0]: row[1] for row in probe_result}
    n_probes = len(src_addr_to_id)

    print(f"✓ Found {n_probes:,} unique probes")
    sys.stdout.flush()

    # Determine train/test split
    train_probe_count = int(n_probes * train_ratio)
    train_probe_ids = set(range(train_probe_count))

    print(
        f"✓ Train: {len(train_probe_ids):,} probes, Test: {n_probes - len(train_probe_ids):,} probes"
    )
    print("\nStep 1b: Getting total row count...")
    sys.stdout.flush()

    # Get total row count for progress tracking
    count_query = "SELECT COUNT(*) FROM read_parquet(?) WHERE src_addr IS NOT NULL"
    total_rows = con.execute(count_query, [input_file]).fetchone()[0]
    print(f"✓ Total rows to process: {total_rows:,}")

    print("\nStep 1c: Streaming partitions from DuckDB and tokenizing in parallel...")
    sys.stdout.flush()

    # SIMPLIFIED: Just write individual rows and group in Python
    # This avoids expensive STRUCT_PACK and is faster
    partition_query = """
    SELECT
        src_id,
        bucket_start_time,
        src_addr,
        dst_addr,
        ip_version,
        rtt,
        event_time
    FROM (
        SELECT
            t.src_addr,
            t.dst_addr,
            t.ip_version,
            t.rtt,
            t.event_time,
            (CAST(epoch(t.event_time) AS BIGINT) / 300) * 300 as bucket_start_time
        FROM read_parquet(?) t
        WHERE t.src_addr IS NOT NULL
        ORDER BY t.src_addr, bucket_start_time, t.event_time
    ) bucketed
    JOIN (SELECT unnest(?) as src_addr, unnest(?) as src_id) probe_map USING (src_addr)
    ORDER BY src_id, bucket_start_time, event_time
    """

    # Prepare probe mapping for query
    src_addrs = list(src_addr_to_id.keys())
    src_ids = [src_addr_to_id[addr] for addr in src_addrs]

    # Execute query and get cursor for streaming
    cursor = con.execute(partition_query, [input_file, src_addrs, src_ids])

    print("✓ Query executing, streaming and tokenizing partitions...")
    sys.stdout.flush()

    # Step 2: Process partitions in streaming batches to avoid OOM
    # Fetch and process in batches of 10000 partitions at a time
    total_partitions_processed = 0
    total_rows_processed = 0
    row_batch_size = 1000000  # Fetch 1M rows at a time

    tokenize_func = partial(
        tokenize_partition_chunk, max_tokens_per_record=max_tokens_per_record
    )

    print(
        f"Processing rows in batches of {row_batch_size:,} with {num_workers} workers..."
    )
    sys.stdout.flush()

    # Group rows into partitions as we stream
    import time

    # Prepare output writers and schema up front so we can stream writes
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("src_id", pa.int64()),
            ("bucket_start_time", pa.int64()),
            ("bucket_duration_s", pa.int32()),
            ("part_id", pa.int32()),
            ("tokens", pa.binary()),
            ("meas_offsets", pa.binary()),
            ("n_tokens", pa.int32()),
            ("n_measurements", pa.int32()),
        ]
    )

    train_writer = None
    test_writer = None

    start_time = time.time()

    train_chunk_count = 0
    test_chunk_count = 0
    train_tokens = 0
    test_tokens = 0

    try:
        train_writer = _create_arrayrecord_writer(output_path / "train.arrayrecord")
        test_writer = _create_arrayrecord_writer(output_path / "test.arrayrecord")

        with Pool(num_workers) as pool:
            pending_partitions = []  # Accumulate complete partitions here
            current_partition_key = None
            current_measurements = []

            while True:
                # Fetch next batch of rows
                row_batch = cursor.fetchmany(row_batch_size)
                if not row_batch:
                    # Flush final partition if any
                    if current_measurements:
                        pending_partitions.append(
                            (
                                current_partition_key[0],
                                current_partition_key[1],
                                current_measurements,
                            )
                        )
                    break

                # Group rows into partitions
                for row in row_batch:
                    (
                        src_id,
                        bucket_start_time,
                        src_addr,
                        dst_addr,
                        ip_version,
                        rtt,
                        event_time,
                    ) = row
                    partition_key = (src_id, bucket_start_time)

                    measurement = {
                        "src_addr": src_addr,
                        "dst_addr": dst_addr,
                        "ip_version": ip_version,
                        "rtt": rtt,
                        "event_time": event_time,
                    }

                    if partition_key != current_partition_key:
                        # New partition - flush previous one
                        if current_measurements:
                            pending_partitions.append(
                                (
                                    current_partition_key[0],
                                    current_partition_key[1],
                                    current_measurements,
                                )
                            )
                        current_partition_key = partition_key
                        current_measurements = [measurement]
                    else:
                        # Same partition - accumulate
                        current_measurements.append(measurement)

                total_rows_processed += len(row_batch)

                # Process accumulated partitions in parallel (send large batches to workers)
                if len(pending_partitions) >= 5000:  # Process 5k partitions at a time
                    for partition_chunks in pool.imap_unordered(
                        tokenize_func, pending_partitions, chunksize=100
                    ):
                        for chunk in partition_chunks:
                            if chunk["src_id"] in train_probe_ids:
                                _write_chunk_record(train_writer, chunk, schema)
                                train_chunk_count += 1
                                train_tokens += chunk["n_tokens"]
                            else:
                                _write_chunk_record(test_writer, chunk, schema)
                                test_chunk_count += 1
                                test_tokens += chunk["n_tokens"]

                    total_partitions_processed += len(pending_partitions)

                    # Calculate progress and ETA
                    elapsed = time.time() - start_time
                    progress_pct = (
                        (total_rows_processed / total_rows) * 100 if total_rows > 0 else 0
                    )
                    rows_per_sec = total_rows_processed / elapsed if elapsed > 0 else 0
                    remaining_rows = total_rows - total_rows_processed
                    eta_seconds = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0
                    eta_mins = eta_seconds / 60

                    print(
                        f"  Progress: {total_rows_processed:,}/{total_rows:,} rows ({progress_pct:.1f}%), "
                        f"{total_partitions_processed:,} partitions, {train_chunk_count + test_chunk_count:,} chunks | "
                        f"{rows_per_sec:,.0f} rows/s | ETA: {eta_mins:.1f} min",
                        flush=True,
                    )
                    pending_partitions = []

            # Process any remaining partitions
            if pending_partitions:
                for partition_chunks in pool.imap_unordered(
                    tokenize_func, pending_partitions, chunksize=100
                ):
                    for chunk in partition_chunks:
                        if chunk["src_id"] in train_probe_ids:
                            _write_chunk_record(train_writer, chunk, schema)
                            train_chunk_count += 1
                            train_tokens += chunk["n_tokens"]
                        else:
                            _write_chunk_record(test_writer, chunk, schema)
                            test_chunk_count += 1
                            test_tokens += chunk["n_tokens"]
                total_partitions_processed += len(pending_partitions)
    finally:
        if train_writer is not None:
            train_writer.close()
        if test_writer is not None:
            test_writer.close()

    print(f"\n✓ Tokenization complete!")
    print(f"  Total partitions processed: {total_partitions_processed:,}")
    print(f"  Train chunks: {train_chunk_count:,}")
    print(f"  Test chunks: {test_chunk_count:,}")

    # Calculate statistics
    print(f"\nStatistics:")
    print(
        f"  Train: {train_tokens:,} tokens, avg {train_tokens // train_chunk_count if train_chunk_count else 0:,} tokens/chunk"
    )
    print(
        f"  Test: {test_tokens:,} tokens, avg {test_tokens // test_chunk_count if test_chunk_count else 0:,} tokens/chunk"
    )
    sys.stdout.flush()
    print("\n✅ Probe chunks created and written successfully!")
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print(f"  - {output_path / 'train.arrayrecord'}")
    print(f"  - {output_path / 'test.arrayrecord'}")


def write_chunks_to_arrayrecord(
    chunks: list[dict], output_file: Path, schema: pa.Schema
):
    """Write chunks to ArrayRecord file."""
    try:
        import array_record.python.array_record_module as array_record_module
    except ImportError:
        print(
            "ERROR: array_record not installed. Install with: pip install array_record"
        )
        sys.exit(1)

    # Convert to PyArrow table
    pa_table = pa.Table.from_pylist(chunks, schema=schema)

    # Write to ArrayRecord
    writer = array_record_module.ArrayRecordWriter(str(output_file), "group_size:1")

    for i in tqdm(range(len(pa_table)), desc=f"Writing {output_file.name}"):
        # Serialize record as PyArrow RecordBatch (single row)
        batch = pa_table.slice(i, 1)

        # Serialize to bytes (IPC format)
        sink = pa.BufferOutputStream()
        writer_ipc = pa.ipc.new_stream(sink, batch.schema)
        writer_ipc.write_batch(batch.to_batches()[0])
        writer_ipc.close()
        record_bytes = sink.getvalue().to_pybytes()

        writer.write(record_bytes)

    writer.close()
    print(f"✓ Wrote {len(pa_table):,} records to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create probe-centric chunks with parallel processing (DATA_LOADING_PLAN_1)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/training_data.parquet",
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/probe_chunks",
        help="Output directory for ArrayRecord files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Training split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100000,
        help="Maximum tokens per chunk before splitting (default: 100000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto = CPU count - 2)",
    )

    args = parser.parse_args()

    create_probe_chunks_parallel(
        input_file=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        max_tokens_per_record=args.max_tokens,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
