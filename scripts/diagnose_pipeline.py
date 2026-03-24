"""Diagnose data pipeline bottleneck on Modal.

Profiles each stage independently:
1. ArrayRecord read (raw bytes from volume)
2. PyArrow IPC deserialization (bytes → Python dicts)
3. Tokenization (dicts → token arrays)
4. Full grain pipeline (end-to-end batches)

Usage:
  modal run scripts/diagnose_pipeline.py
"""

import os
import time
import modal

APP_NAME = "ping-llm-diagnose"
DATA_VOLUME = "ping-llm-data"
DATA_MOUNT = "/mnt/data"
WORKDIR = "/workspace"

IGNORE_PATTERNS = [
    ".git", ".venv", ".train_venv", ".slurm_venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", "outputs", "logs", "data",
    "local_datasets", "archive", "*.parquet", "*.arrayrecord",
    ".DS_Store", ".claude", "docs",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("git", "build-essential", "tzdata")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system "
        "torch --index-url https://download.pytorch.org/whl/cu124 && "
        "uv pip install --system "
        "pyarrow numpy grain array_record"
    )
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME)


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    volumes={DATA_MOUNT: data_vol},
    timeout=600,
)
def diagnose():
    import sys
    sys.path.insert(0, f"{WORKDIR}/src")

    import numpy as np
    import pyarrow.ipc as ipc
    import grain.python as grain
    from ping_llm.data.datasource import DeserializeProbeRow, ProbeRowSampler
    from ping_llm.data.tokenization import encode_measurement
    from ping_llm.data.pipeline import build_probe_chunk_dataset
    from datetime import datetime

    shard_paths = [
        f"{DATA_MOUNT}/train_shards/train.arrayrecord-0000{i}-of-00004"
        for i in range(4)
    ]

    print("=" * 60)
    print("STAGE 1: ArrayRecord raw read speed")
    print("=" * 60)
    source = grain.ArrayRecordDataSource(shard_paths)
    n_total = len(source)
    print(f"Total records: {n_total}")

    # Read 20 raw records and time it
    N_READ = 20
    sizes = []
    t0 = time.perf_counter()
    raw_records = []
    for i in range(N_READ):
        raw = source[i]
        raw_records.append(raw)
        sizes.append(len(raw))
    elapsed = time.perf_counter() - t0
    print(f"{N_READ} records read in {elapsed:.3f}s = {elapsed/N_READ*1000:.1f}ms/record")
    print(f"Record sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f} bytes")
    print(f"Read throughput: {sum(sizes)/elapsed/1e6:.1f} MB/s")
    print()

    print("=" * 60)
    print("STAGE 2: PyArrow IPC deserialization")
    print("=" * 60)
    deserializer = DeserializeProbeRow()
    deserialized_rows = []
    t0 = time.perf_counter()
    for raw in raw_records:
        row = deserializer.map(raw)
        deserialized_rows.append(row)
    elapsed = time.perf_counter() - t0
    n_meas = [len(r['measurements']) for r in deserialized_rows]
    print(f"{N_READ} records deserialized in {elapsed:.3f}s = {elapsed/N_READ*1000:.1f}ms/record")
    print(f"Measurements per row: min={min(n_meas)}, max={max(n_meas)}, avg={sum(n_meas)/len(n_meas):.0f}")
    print()

    print("=" * 60)
    print("STAGE 3: ProbeRowSampler (FlatMap)")
    print("=" * 60)
    sampler = ProbeRowSampler(crop_size=1024, seed=42)
    total_contexts = 0
    t0 = time.perf_counter()
    for row in deserialized_rows:
        contexts = sampler.flat_map(row)
        total_contexts += len(contexts)
    elapsed = time.perf_counter() - t0
    print(f"{N_READ} rows → {total_contexts} contexts in {elapsed:.3f}s")
    print(f"  {elapsed/N_READ*1000:.1f}ms per row, {elapsed/total_contexts*1000:.2f}ms per context")
    print(f"  Avg contexts/row: {total_contexts/N_READ:.1f}")
    print()

    print("=" * 60)
    print("STAGE 4: End-to-end grain pipeline (no mp_prefetch)")
    print("=" * 60)
    dataset_no_mp = build_probe_chunk_dataset(
        arrayrecord_path=shard_paths,
        batch_size=32,
        crop_size=1024,
        shuffle=True,
        shuffle_seed=42,
        num_workers=0,
        prefetch_buffer_size=2,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    for i, batch in enumerate(dataset_no_mp):
        if i >= 5:
            break
    elapsed = time.perf_counter() - t0
    print(f"5 batches (no mp) in {elapsed:.3f}s = {elapsed/5:.1f}s/batch")
    tokens_per_batch = 32 * 1024
    print(f"Throughput: {5 * tokens_per_batch / elapsed:.0f} tok/s")
    print()

    print("=" * 60)
    print("STAGE 5: End-to-end grain pipeline (with mp_prefetch, 6 workers)")
    print("=" * 60)
    dataset_mp = build_probe_chunk_dataset(
        arrayrecord_path=shard_paths,
        batch_size=32,
        crop_size=1024,
        shuffle=True,
        shuffle_seed=42,
        num_workers=6,
        prefetch_buffer_size=4,
        use_multiprocessing=True,
    )
    # Warmup: first batch includes worker startup
    t_warmup = time.perf_counter()
    batch_iter = iter(dataset_mp)
    _ = next(batch_iter)
    warmup_time = time.perf_counter() - t_warmup
    print(f"First batch (warmup): {warmup_time:.1f}s")

    # Steady state
    t0 = time.perf_counter()
    for i in range(10):
        _ = next(batch_iter)
    elapsed = time.perf_counter() - t0
    print(f"Next 10 batches: {elapsed:.3f}s = {elapsed/10:.1f}s/batch")
    print(f"Steady-state throughput: {10 * tokens_per_batch / elapsed:.0f} tok/s")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)


@app.local_entrypoint()
def run():
    diagnose.remote()
