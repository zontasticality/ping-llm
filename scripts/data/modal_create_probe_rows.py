"""
Modal deployment for PLAN_3 probe row preprocessing.

This script runs create_probe_rows.py on Modal to process the full dataset.
"""

import modal

# Create Modal app
app = modal.App("probe-rows-preprocessing")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow",
        "duckdb",
    )
    .apt_install("curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'source $HOME/.local/bin/env' >> ~/.bashrc",
    )
)

# Mount the ping-llm volume
volume = modal.Volume.from_name("ping-llm-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=3600 * 4,  # 4 hours
    cpu=8.0,  # 8 cores for better parallelism
    memory=32768,  # 32GB RAM (allows 31GB for DuckDB)
)
def create_probe_rows(
    input_pattern: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/probe_rows",
    max_row_size_mb: float = 8.0,
    train_ratio: float = 0.9,
):
    """
    Run probe row preprocessing on Modal.

    Args:
        input_pattern: Input parquet file pattern
        output_dir: Output directory for ArrayRecord files
        max_row_size_mb: Maximum row size in MB
        train_ratio: Train/test split ratio
    """
    import subprocess
    import sys
    from pathlib import Path

    print(f"Starting probe row preprocessing...")
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")
    print(f"Max row size: {max_row_size_mb}MB")
    print(f"Train ratio: {train_ratio}")

    # Install array_record
    print("\nInstalling array_record...")
    subprocess.run(
        ["pip", "install", "array_record"],
        check=True,
        capture_output=True,
    )

    # Copy preprocessing script to Modal
    script_content = '''
"""Probe row preprocessing script (embedded in Modal)."""
import argparse
import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from typing import List
import hashlib

try:
    import array_record.python.array_record_module as array_record_module
except ImportError:
    raise ImportError("array_record not installed")


def hash_src_addr(src_addr: str) -> int:
    """Convert src_addr to stable integer ID (within int64 range)."""
    hash_hex = hashlib.md5(src_addr.encode()).hexdigest()[:15]
    return int(hash_hex, 16)


def serialize_measurements_to_ipc(measurements: List[dict]) -> bytes:
    """Serialize measurements to PyArrow IPC format."""
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


def create_arrayrecord_entry(src_id, measurements_bytes, n_measurements, first_timestamp, last_timestamp) -> bytes:
    """Create ArrayRecord entry."""
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


def write_probe_to_arrayrecord(writer, src_addr, measurements, max_size_bytes) -> int:
    """Write probe to ArrayRecord, splitting if needed."""
    if not measurements:
        return 0
    src_id = hash_src_addr(src_addr)
    rows_written = 0
    measurements_bytes = serialize_measurements_to_ipc(measurements)
    if len(measurements_bytes) <= max_size_bytes:
        entry = create_arrayrecord_entry(src_id, measurements_bytes, len(measurements), measurements[0]['event_time'], measurements[-1]['event_time'])
        writer.write(entry)
        return 1
    left = 0
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


def process(input_pattern, output_dir, max_row_size_mb, train_ratio):
    """Main processing function."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_size_bytes = int(max_row_size_mb * 1024 * 1024)
    print(f"Reading parquet: {input_pattern}")
    con = duckdb.connect(':memory:')
    con.execute(f"CREATE VIEW all_measurements AS SELECT * FROM read_parquet('{input_pattern}')")
    total_count = con.execute("SELECT COUNT(*) FROM all_measurements").fetchone()[0]
    print(f"Total measurements: {total_count:,}")
    src_addrs = [row[0] for row in con.execute("SELECT DISTINCT src_addr FROM all_measurements ORDER BY src_addr").fetchall()]
    print(f"Unique probes: {len(src_addrs):,}")
    n_train = int(len(src_addrs) * train_ratio)
    train_addrs = set(src_addrs[:n_train])
    test_addrs = set(src_addrs[n_train:])
    print(f"Train probes: {len(train_addrs):,}")
    print(f"Test probes: {len(test_addrs):,}")
    train_path = str(output_dir / "train.arrayrecord")
    test_path = str(output_dir / "test.arrayrecord")
    train_writer = array_record_module.ArrayRecordWriter(train_path, 'group_size:1')
    test_writer = array_record_module.ArrayRecordWriter(test_path, 'group_size:1')
    train_rows = 0
    test_rows = 0
    train_measurements_total = 0
    test_measurements_total = 0
    for idx, src_addr in enumerate(src_addrs):
        if (idx + 1) % 100 == 0 or idx == len(src_addrs) - 1:
            print(f"Processing probe {idx + 1}/{len(src_addrs)}...", end='\\r')
        result = con.execute("SELECT event_time, src_addr, dst_addr, ip_version, rtt FROM all_measurements WHERE src_addr = ? ORDER BY event_time", [src_addr]).fetchall()
        measurements = [{'event_time': row[0], 'src_addr': row[1], 'dst_addr': row[2], 'ip_version': row[3], 'rtt': row[4]} for row in result]
        if src_addr in train_addrs:
            rows = write_probe_to_arrayrecord(train_writer, src_addr, measurements, max_size_bytes)
            train_rows += rows
            train_measurements_total += len(measurements)
        else:
            rows = write_probe_to_arrayrecord(test_writer, src_addr, measurements, max_size_bytes)
            test_rows += rows
            test_measurements_total += len(measurements)
    print()
    train_writer.close()
    test_writer.close()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Train: {train_rows:,} rows, {train_measurements_total:,} measurements")
    print(f"Test: {test_rows:,} rows, {test_measurements_total:,} measurements")
    print(f"Train output: {train_path}")
    print(f"Test output: {test_path}")


if __name__ == "__main__":
    import sys
    process(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
'''

    # Write script to temp file
    script_path = Path("/tmp/preprocess.py")
    script_path.write_text(script_content)

    # Run preprocessing
    print("\nRunning preprocessing...")
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            input_pattern,
            output_dir,
            str(max_row_size_mb),
            str(train_ratio),
        ],
        check=True,
    )

    # Commit volume
    volume.commit()

    print("\n✓ Preprocessing complete!")
    print(f"Output written to: {output_dir}")
    print("Volume committed successfully")


@app.local_entrypoint()
def main(
    input_pattern: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/probe_rows",
):
    """Local entrypoint to trigger preprocessing."""
    print("Starting probe row preprocessing on Modal...")
    create_probe_rows.remote(
        input_pattern=input_pattern,
        output_dir=output_dir,
    )
    print("✓ Done!")
