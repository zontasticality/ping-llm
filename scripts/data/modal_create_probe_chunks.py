"""Create probe-centric chunks on Modal using temporary sharding + parallel shard workers.

Usage:
  modal run scripts/data/modal_create_probe_chunks.py::create_chunks

Assumes:
  - Modal Volume (default name: ping-llm, override with MODAL_VOLUME)
    is mounted at /mnt and contains /mnt/data/training_data.parquet
  - Outputs are written back to the Volume under /mnt/data/probe_chunks

What this does:
  - Stage A: DuckDB writes a temporary sharded Parquet dataset to local disk (/tmp)
  - Stage B: multiprocessing across shards; each shard writes its own train/test ArrayRecords

Outputs:
  /mnt/data/probe_chunks/
    manifest.json
    preprocess_stats.jsonl
    probe_map.parquet
    train/train_shard_*.arrayrecord
    test/test_shard_*.arrayrecord

Notes:
  - This approach avoids Python-side row streaming/pickling bottlenecks.
  - Temp shards live on the container's local disk for speed.
"""

from __future__ import annotations

import os
import subprocess
import sys

import modal

APP_NAME = "ping-llm-create-probe-chunks"
WORKDIR = "/workspace"
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")

# Ignore heavy paths when forwarding code into the container.
IGNORE_PATTERNS = [
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "outputs",
    "logs",
    "data",
    "local_datasets",
    "*.parquet",
    "*.arrayrecord",
    ".DS_Store",
]

# Prefer uv_pip_install for faster, more reproducible dependency installs.
# Pin versions if you want fully reproducible builds.
image = (
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "pyarrow",
        "numpy",
        "tqdm",
        "duckdb",
        "array_record",
        "pytz",
    )
    # By default, files are forwarded at container startup (copy=False), which speeds iteration.
    .add_local_dir(".", remote_path=WORKDIR, ignore=IGNORE_PATTERNS)
)

app = modal.App(APP_NAME)
shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": shared_vol},
    cpu=32,
    memory=64 * 1024,  # MiB
    timeout=60 * 60 * 4,
    ephemeral_disk=800 * 1024,  # 800 GiB (MiB units)
)
def create_chunks(
    input_path: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/probe_chunks",
    train_ratio: float = 0.9,
    max_tokens: int = 100000,
    num_shards: int = 256,
):
    """Create probe-centric chunks in ArrayRecord format."""

    # Modal often constrains CPUs via cpuset; prefer affinity if present.
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except Exception:
            affinity = None

    cpu_count = os.cpu_count() or 1
    effective_cpus = affinity or cpu_count

    print(f"Input:        {input_path}")
    print(f"Output:       {output_dir}")
    print(f"Train ratio:  {train_ratio}")
    print(f"Max tokens:   {max_tokens:,}")
    print(f"Temp shards:  /tmp/probe_shards")
    print(f"Shards:       {num_shards}")
    print(
        f"CPUs:         os.cpu_count={cpu_count}, affinity={affinity}, effective={effective_cpus}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Use local disk for temp shards (faster and avoids creating many files on the Volume).
    temp_shards_dir = "/tmp/probe_shards"

    # Use most CPUs for shard-processing workers; leave 1 core for OS/DuckDB overhead.
    workers = max(1, effective_cpus - 1)

    # IMPORTANT: place probe_chunk_preprocess.py at scripts/data/probe_chunk_preprocess.py in your repo.
    cmd = [
        "python",
        "-u",
        "scripts/data/probe_chunk_preprocess.py",
        "--input",
        input_path,
        "--output",
        output_dir,
        "--temp-shards",
        temp_shards_dir,
        "--num-shards",
        str(num_shards),
        "--train-ratio",
        str(train_ratio),
        "--max-tokens",
        str(max_tokens),
        "--workers",
        str(workers),
        "--duckdb-threads-sharding",
        str(effective_cpus),
        "--duckdb-threads-worker",
        "1",
        "--overwrite-temp",
    ]

    print("\nRunning:", " ".join(cmd))
    print("=" * 80)

    subprocess.run(
        cmd,
        check=True,
        cwd=WORKDIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    print("\nCommitting volume changes...")
    shared_vol.commit()

    print("\n✅ Probe chunks created successfully!")
    print(f"Output location: {output_dir}")
    print("Key outputs:")
    print(f"  - {output_dir}/manifest.json")
    print(f"  - {output_dir}/train/train_shard_*.arrayrecord")
    print(f"  - {output_dir}/test/test_shard_*.arrayrecord")


@app.local_entrypoint()
def main():
    print("Starting probe chunk creation on Modal...")
    create_chunks.remote()
    print("\n✅ Completed!")
