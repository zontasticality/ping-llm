"""
Simple Modal runner for streaming probe row preprocessing.

This is the easiest way to run the preprocessor on Modal.
"""

import modal

app = modal.App("probe-rows-streaming")

# Image with all dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pyarrow",
    "duckdb",
    "array_record",
    "psutil",
)

# Mount the volume
volume = modal.Volume.from_name("ping-llm", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=3600 * 6,  # 6 hours (generous)
    cpu=8.0,
    memory=64 * 1024,  # 32GB
)
def preprocess(
    input_pattern: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/probe_rows",
    workers: int = 8,
    memory_limit_gb: float = 50.0,
):
    """Run streaming preprocessor."""
    import sys
    import subprocess
    from pathlib import Path

    print("=" * 80)
    print("MODAL: Streaming Probe Row Preprocessing")
    print("=" * 80)
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers}")
    print(f"Memory limit: {memory_limit_gb}GB")
    print()

    # Write the streaming script inline
    script = '''
import argparse
import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from typing import List
import hashlib
from multiprocessing import Pool, cpu_count
import time
import tempfile
import shutil
import psutil

try:
    import array_record.python.array_record_module as array_record_module
except ImportError:
    raise ImportError("array_record not installed")


def get_available_memory_gb():
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except:
        return 4.0


def hash_src_addr(src_addr: str) -> int:
    hash_hex = hashlib.md5(src_addr.encode()).hexdigest()[:15]
    return int(hash_hex, 16)


def serialize_measurements_to_ipc(measurements: List[dict]) -> bytes:
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


def create_arrayrecord_entry(src_id, measurements_bytes, n_measurements, first_timestamp, last_timestamp):
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


def write_probe_to_arrayrecord(writer, src_addr, measurements, max_size_bytes):
    if not measurements:
        return 0
    src_id = hash_src_addr(src_addr)
    measurements_bytes = serialize_measurements_to_ipc(measurements)
    if len(measurements_bytes) <= max_size_bytes:
        entry = create_arrayrecord_entry(src_id, measurements_bytes, len(measurements), measurements[0]['event_time'], measurements[-1]['event_time'])
        writer.write(entry)
        return 1
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
        entry = create_arrayrecord_entry(src_id, chunk_bytes, len(chunk), chunk[0]['event_time'], chunk[-1]['event_time'])
        writer.write(entry)
        rows_written += 1
        left += best_size
    return rows_written


def process_probe_batch_from_parquet(args):
    parquet_file, row_start, row_end, output_path, max_size_bytes, worker_id = args
    con = duckdb.connect(':memory:')
    probe_batch = con.execute(f"""
        SELECT src_addr, measurements
        FROM read_parquet('{parquet_file}')
        LIMIT {row_end - row_start} OFFSET {row_start}
    """).fetchall()
    con.close()
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
        rows = write_probe_to_arrayrecord(writer, src_addr, measurements, max_size_bytes)
        rows_written += rows
        measurements_total += len(measurements)
    writer.close()
    print(f"  Worker {worker_id}: Completed batch [{row_start}:{row_end}] -> {rows_written} rows")
    return output_path, rows_written, measurements_total


def merge_arrayrecords(input_paths, output_path):
    print(f"  Merging {len(input_paths)} partial files...")
    writer = array_record_module.ArrayRecordWriter(output_path, 'group_size:1')
    total_rows = 0
    for i, input_path in enumerate(input_paths):
        print(f"    Merging {i+1}/{len(input_paths)}", end='\\r')
        reader = array_record_module.ArrayRecordReader(input_path)
        n_records = reader.num_records()
        for j in range(n_records):
            record_bytes = reader.read([j])[0]
            writer.write(record_bytes)
            total_rows += 1
        reader.close()
    writer.close()
    print(f"\\n    Merged {total_rows:,} rows")


def main(input_pattern, output_dir, max_row_size_mb, train_ratio, num_workers, memory_limit_gb):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_size_bytes = int(max_row_size_mb * 1024 * 1024)

    print("=" * 80)
    print("STREAMING PREPROCESSING")
    print("=" * 80)
    print(f"Workers: {num_workers}")
    print(f"Memory limit: {memory_limit_gb}GB")
    print()

    start_time = time.time()
    temp_dir = Path(tempfile.mkdtemp(prefix="probe_rows_"))

    try:
        print("Step 1: Streaming GROUP BY to disk...")
        con = duckdb.connect(':memory:')
        con.execute(f"SET memory_limit='{memory_limit_gb}GB'")

        intermediate_parquet = str(temp_dir / "grouped_probes.parquet")
        group_start = time.time()

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
                FROM read_parquet('{input_pattern}')
                GROUP BY src_addr
                ORDER BY src_addr
            ) TO '{intermediate_parquet}' (FORMAT PARQUET)
        """)

        group_time = time.time() - group_start
        n_probes = con.execute(f"SELECT COUNT(*) FROM read_parquet('{intermediate_parquet}')").fetchone()[0]
        total_measurements = con.execute(f"SELECT SUM(LENGTH(measurements)) FROM read_parquet('{intermediate_parquet}')").fetchone()[0]

        print(f"  Grouped {n_probes:,} probes in {group_time:.1f}s")
        print(f"  Total measurements: {total_measurements:,}")
        con.close()

        print("\\nStep 2: Splitting train/test...")
        n_train = int(n_probes * train_ratio)
        print(f"  Train: {n_train:,} probes")
        print(f"  Test: {n_probes - n_train:,} probes")

        print(f"\\nStep 3: Parallel processing ({num_workers} workers)...")
        train_batch_size = max(100, n_train // (num_workers * 4))
        test_batch_size = max(100, (n_probes - n_train) // (num_workers * 4))

        train_batches = []
        for i in range(0, n_train, train_batch_size):
            end = min(i + train_batch_size, n_train)
            output_path = str(temp_dir / f"train_part_{len(train_batches)}.arrayrecord")
            train_batches.append((intermediate_parquet, i, end, output_path, max_size_bytes, len(train_batches)))

        test_batches = []
        for i in range(n_train, n_probes, test_batch_size):
            end = min(i + test_batch_size, n_probes)
            output_path = str(temp_dir / f"test_part_{len(test_batches)}.arrayrecord")
            test_batches.append((intermediate_parquet, i, end, output_path, max_size_bytes, len(test_batches)))

        print(f"  Processing {len(train_batches)} train batches...")
        process_start = time.time()
        with Pool(num_workers) as pool:
            train_results = pool.map(process_probe_batch_from_parquet, train_batches)
        train_time = time.time() - process_start
        print(f"  Train: {train_time:.1f}s")

        if test_batches:
            print(f"  Processing {len(test_batches)} test batches...")
            test_start = time.time()
            with Pool(num_workers) as pool:
                test_results = pool.map(process_probe_batch_from_parquet, test_batches)
            test_time = time.time() - test_start
            print(f"  Test: {test_time:.1f}s")
        else:
            test_results = []

        print("\\nStep 4: Merging...")
        train_output = str(output_dir / "train.arrayrecord")
        merge_arrayrecords([r[0] for r in train_results], train_output)

        if test_results:
            test_output = str(output_dir / "test.arrayrecord")
            merge_arrayrecords([r[0] for r in test_results], test_output)

        train_rows = sum(r[1] for r in train_results)
        train_meas = sum(r[2] for r in train_results)
        test_rows = sum(r[1] for r in test_results) if test_results else 0
        test_meas = sum(r[2] for r in test_results) if test_results else 0
        total_time = time.time() - start_time

        print("\\n" + "=" * 80)
        print("COMPLETE!")
        print("=" * 80)
        print(f"Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Throughput: {total_measurements/total_time:,.0f} meas/sec")
        print(f"\\nTrain: {train_rows:,} rows, {train_meas:,} measurements")
        print(f"Test: {test_rows:,} rows, {test_meas:,} measurements")
        print(f"\\nOutput: {output_dir}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
'''

    # Write script to file
    script_path = Path("/tmp/preprocess_streaming.py")
    script_path.write_text(script)

    # Run it
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            input_pattern,
            output_dir,
            "8.0",  # max_row_size_mb
            "0.9",  # train_ratio
            str(workers),
            str(memory_limit_gb),
        ],
        check=True,
        capture_output=False,
        text=True,
    )

    # Commit volume
    print("\nCommitting volume...")
    volume.commit()

    print("\n✓ Preprocessing complete and volume committed!")
    return output_dir


@app.local_entrypoint()
def main():
    """Run preprocessing from command line."""
    print("Starting preprocessing on Modal...")
    output = preprocess.remote()
    print(f"\n✓ Complete! Output: {output}")
