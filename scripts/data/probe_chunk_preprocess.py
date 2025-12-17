#!/usr/bin/env python3
"""probe_chunk_preprocess.py

Probe-centric chunk builder for large Parquet measurement logs.

Design goals (matches DATA_LOADING_PLAN_1):
- Partition by (src_id, 5-minute bucket)
- Sort each partition by event_time
- Tokenize sequentially with delta timestamps
- Enforce max_tokens_per_record; when split, reset timestamp chain
- Write chunk records to ArrayRecord for fast Grain access

Key performance fixes vs the original script:
1) Temporary sharding:
   - Stage A writes a temporary sharded Parquet dataset by hash(src_addr) % N.
   - This removes the global ORDER BY bottleneck and enables embarrassingly-parallel processing.

2) Parallel batch transforms:
   - Stage B uses multiprocessing across shards.
   - Each worker reads its own shard(s) and tokenizes locally (no giant Python objects pickled to workers).

3) Parallel streaming writes:
   - Each shard is written to its own train/test ArrayRecord files.
   - No cross-process contention on a single output file.

This script is friendly to Modal-style execution (single node, many cores, ephemeral disk).

Example:
  python probe_chunk_preprocess.py \
    --input data/training_data.parquet \
    --output data/probe_chunks_out \
    --temp-shards /tmp/probe_shards \
    --num-shards 256 \
    --workers 48 \
    --max-tokens 100000 \
    --train-ratio 0.9

Notes for Modal:
- Modal containers often restrict CPU via cpuset/cgroups.
  This script reports both cpu_count() and sched_getaffinity().
- Prefer writing temp shards to the local filesystem inside the container (fast),
  then copy final outputs to persistent storage.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from functools import partial
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pyarrow as pa

# Add project root to path (same convention as your existing script).
# Adjust if you place this file elsewhere.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tokenization import encode_measurement, VOCAB_SIZE  # noqa: E402


# -----------------------------
# Small utilities
# -----------------------------


def effective_cpu_count() -> int:
    """Return the number of CPUs this process is *allowed* to use."""
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except Exception:
            affinity = None

    if affinity is not None and affinity > 0:
        return affinity

    try:
        import multiprocessing

        return multiprocessing.cpu_count()
    except Exception:
        return 1


def require_arrayrecord() -> "array_record_module":
    """Import ArrayRecord with a clear error message."""
    try:
        import array_record.python.array_record_module as array_record_module

        return array_record_module
    except ImportError as e:
        raise RuntimeError(
            "array_record is not installed. Install with: pip install array_record"
        ) from e


def serialize_tokens(tokens: list[int]) -> bytes:
    """Serialize token list as uint16 array."""
    # Safety: token ids must fit uint16.
    if not tokens:
        return b""
    if not all(0 <= t < VOCAB_SIZE for t in tokens):
        bad = [t for t in tokens if not (0 <= t < VOCAB_SIZE)][:10]
        raise ValueError(f"Invalid token IDs found (showing up to 10): {bad}")
    return struct.pack(f"{len(tokens)}H", *tokens)


def serialize_offsets(offsets: list[int]) -> bytes:
    """Serialize offset list as int32 array."""
    if not offsets:
        return b""
    return struct.pack(f"{len(offsets)}i", *offsets)


def chunk_schema() -> pa.Schema:
    return pa.schema(
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


def chunk_to_ipc_bytes(chunk: dict, schema: pa.Schema) -> bytes:
    """Encode one chunk dict as Arrow IPC stream bytes (single-row RecordBatch)."""
    batch = pa.Table.from_pylist([chunk], schema=schema).to_batches()[0]
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()


# -----------------------------
# Tokenization
# -----------------------------


def tokenize_partition(
    src_id: int,
    bucket_start_time: int,
    measurements: list[dict],
    *,
    max_tokens_per_record: int,
) -> Iterator[dict]:
    """Tokenize one (src_id, bucket_start_time) partition into 1+ chunk records."""

    current_tokens: list[int] = []
    current_offsets: list[int] = []
    prev_timestamp = None
    part_id = 0

    for meas in measurements:
        meas_start = len(current_tokens)

        # At split boundaries, reset timestamp chain so first measurement is absolute.
        if part_id > 0 and prev_timestamp is not None:
            prev_timestamp = None

        meas_tokens = encode_measurement(
            meas,
            prev_timestamp=prev_timestamp,
            include_timestamp=True,
        )

        if current_tokens and (
            len(current_tokens) + len(meas_tokens) > max_tokens_per_record
        ):
            yield {
                "src_id": int(src_id),
                "bucket_start_time": int(bucket_start_time),
                "bucket_duration_s": 300,
                "part_id": int(part_id),
                "tokens": serialize_tokens(current_tokens),
                "meas_offsets": serialize_offsets(current_offsets),
                "n_tokens": int(len(current_tokens)),
                "n_measurements": int(len(current_offsets)),
            }
            current_tokens = []
            current_offsets = []
            part_id += 1
            prev_timestamp = None
            meas_start = 0

        current_offsets.append(meas_start)
        current_tokens.extend(meas_tokens)
        prev_timestamp = meas["event_time"]

    if current_tokens:
        yield {
            "src_id": int(src_id),
            "bucket_start_time": int(bucket_start_time),
            "bucket_duration_s": 300,
            "part_id": int(part_id),
            "tokens": serialize_tokens(current_tokens),
            "meas_offsets": serialize_offsets(current_offsets),
            "n_tokens": int(len(current_tokens)),
            "n_measurements": int(len(current_offsets)),
        }


# -----------------------------
# DuckDB stages
# -----------------------------


def duckdb_connect(threads: int, enable_progress: bool = False):
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {int(threads)}")
    con.execute("SET preserve_insertion_order = false")

    # Optional progress printing (can be noisy in distributed logs).
    if enable_progress:
        con.execute("SET enable_progress_bar = true")
        con.execute("SET enable_progress_bar_print = true")
    else:
        con.execute("SET enable_progress_bar = false")

    return con


@dataclass
class ProbeMapInfo:
    n_probes: int
    train_cutoff_src_id: int


def build_probe_map(
    con, input_glob: str, train_ratio: float, out_dir: Path
) -> ProbeMapInfo:
    """Create a stable src_addr -> src_id mapping in DuckDB and persist it."""

    out_dir.mkdir(parents=True, exist_ok=True)
    probe_map_parquet = out_dir / "probe_map.parquet"

    # Materialize probe_map as a TEMP table for joins.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE probe_map AS
        SELECT
          src_addr,
          ROW_NUMBER() OVER (ORDER BY src_addr) - 1 AS src_id
        FROM (
          SELECT DISTINCT src_addr
          FROM read_parquet(?)
          WHERE src_addr IS NOT NULL
        )
        ORDER BY src_addr;
        """,
        [input_glob],
    )

    n_probes = con.execute("SELECT COUNT(*) FROM probe_map").fetchone()[0]
    train_cutoff = int(n_probes * train_ratio)

    # Persist for debugging / reproducibility.
    con.execute(
        "COPY probe_map TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(probe_map_parquet)],
    )

    # Also save small JSON metadata.
    (out_dir / "probe_map_meta.json").write_text(
        json.dumps(
            {
                "n_probes": int(n_probes),
                "train_ratio": float(train_ratio),
                "train_cutoff_src_id": int(train_cutoff),
            },
            indent=2,
        )
    )

    return ProbeMapInfo(n_probes=int(n_probes), train_cutoff_src_id=int(train_cutoff))


def make_temp_shards(
    con,
    input_glob: str,
    temp_shards_dir: Path,
    num_shards: int,
    *,
    overwrite: bool,
    enable_progress: bool,
) -> None:
    """Stage A: write a temporary sharded Parquet dataset.

    Output layout:
      temp_shards_dir/
        shard_id=0/part-*.parquet
        shard_id=1/part-*.parquet
        ...

    Each row includes computed:
      - src_id (joined from probe_map)
      - bucket_start_time
      - shard_id

    This is intentionally a *wide* intermediate to make Stage B simple and fast.
    """

    if overwrite and temp_shards_dir.exists():
        # Be very careful with deletes; this is intended for a temp path.
        for p in temp_shards_dir.glob("**/*"):
            if p.is_file():
                p.unlink()
        for p in sorted(temp_shards_dir.glob("**/*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass

    temp_shards_dir.mkdir(parents=True, exist_ok=True)

    # A stable shard_id based on hash(src_addr).
    # Use a mod that keeps shard_id in [0, num_shards).
    shard_expr = f"((hash(src_addr) % {int(num_shards)}) + {int(num_shards)}) % {int(num_shards)}"

    con.execute("SET VARIABLE input_glob = ?", [input_glob])
    con.execute("SET VARIABLE out_dir = ?", [str(temp_shards_dir)])

    # Note: we do NOT do a global ORDER BY here.
    # Stage B will order per-shard (much cheaper than global sort).
    #
    # DuckDB will write multiple files per shard partition. That is OK.
    query = f"""
    COPY (
      SELECT
        pm.src_id AS src_id,
        t.src_addr,
        t.dst_addr,
        t.ip_version,
        t.rtt,
        t.event_time,
        (CAST(epoch(t.event_time) AS BIGINT) / 300) * 300 AS bucket_start_time,
        {shard_expr} AS shard_id
      FROM read_parquet(getvariable('input_glob')) t
      JOIN probe_map pm USING (src_addr)
      WHERE t.src_addr IS NOT NULL
    )
    TO (getvariable('out_dir'))
    (FORMAT PARQUET, PARTITION_BY (shard_id), COMPRESSION ZSTD);
    """

    if enable_progress:
        con.execute("SET enable_progress_bar = true")
        con.execute("SET enable_progress_bar_print = true")

    con.execute(query)


# -----------------------------
# Worker stage (shard -> arrayrecords)
# -----------------------------


@dataclass
class ShardStats:
    shard_id: int
    rows_seen: int
    partitions_seen: int
    train_chunks: int
    test_chunks: int
    train_tokens: int
    test_tokens: int
    seconds: float


def process_one_shard(
    shard_id: int,
    *,
    temp_shards_dir: str,
    output_dir: str,
    train_cutoff_src_id: int,
    max_tokens_per_record: int,
    duckdb_threads: int,
    row_batch_size: int,
    enable_duckdb_progress: bool,
) -> ShardStats:
    """Stage B: tokenizes one shard and writes train/test ArrayRecord files.

    Each shard produces:
      output_dir/train/train_shard_{shard_id:05d}.arrayrecord
      output_dir/test/test_shard_{shard_id:05d}.arrayrecord

    This makes writes embarrassingly-parallel.
    """

    start = time.time()

    temp_dir = Path(temp_shards_dir)
    out_dir = Path(output_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    # Discover this shard's parquet files.
    shard_path = temp_dir / f"shard_id={int(shard_id)}"
    if not shard_path.exists():
        return ShardStats(
            shard_id=int(shard_id),
            rows_seen=0,
            partitions_seen=0,
            train_chunks=0,
            test_chunks=0,
            train_tokens=0,
            test_tokens=0,
            seconds=0.0,
        )

    parquet_glob = str(shard_path / "*.parquet")

    # Use a local DuckDB connection in the worker.
    # Set threads low to avoid oversubscription across workers.
    con = duckdb_connect(threads=duckdb_threads, enable_progress=enable_duckdb_progress)

    # Stream ordered rows for this shard.
    # Ordering per shard avoids the global sort bottleneck.
    cursor = con.execute(
        """
        SELECT
          src_id,
          bucket_start_time,
          src_addr,
          dst_addr,
          ip_version,
          rtt,
          event_time
        FROM read_parquet(?)
        ORDER BY src_id, bucket_start_time, event_time;
        """,
        [parquet_glob],
    )

    schema = chunk_schema()
    array_record_module = require_arrayrecord()

    train_file = out_dir / "train" / f"train_shard_{int(shard_id):05d}.arrayrecord"
    test_file = out_dir / "test" / f"test_shard_{int(shard_id):05d}.arrayrecord"

    train_writer = array_record_module.ArrayRecordWriter(
        str(train_file), "group_size:1"
    )
    test_writer = array_record_module.ArrayRecordWriter(str(test_file), "group_size:1")

    rows_seen = 0
    partitions_seen = 0
    train_chunks = 0
    test_chunks = 0
    train_tokens = 0
    test_tokens = 0

    current_key: Optional[tuple[int, int]] = None
    current_measurements: list[dict] = []

    def flush_partition(key: tuple[int, int], measurements: list[dict]) -> None:
        nonlocal partitions_seen, train_chunks, test_chunks, train_tokens, test_tokens
        if not measurements:
            return
        partitions_seen += 1
        src_id, bucket_start_time = key
        for chunk in tokenize_partition(
            src_id,
            bucket_start_time,
            measurements,
            max_tokens_per_record=max_tokens_per_record,
        ):
            record_bytes = chunk_to_ipc_bytes(chunk, schema)
            if src_id < train_cutoff_src_id:
                train_writer.write(record_bytes)
                train_chunks += 1
                train_tokens += int(chunk["n_tokens"])
            else:
                test_writer.write(record_bytes)
                test_chunks += 1
                test_tokens += int(chunk["n_tokens"])

    try:
        while True:
            batch = cursor.fetchmany(row_batch_size)
            if not batch:
                break

            rows_seen += len(batch)

            # Convert rows -> streaming grouping. Avoid building huge structures.
            for (
                src_id,
                bucket_start_time,
                src_addr,
                dst_addr,
                ip_version,
                rtt,
                event_time,
            ) in batch:
                key = (int(src_id), int(bucket_start_time))

                if current_key is None:
                    current_key = key

                if key != current_key:
                    flush_partition(current_key, current_measurements)
                    current_key = key
                    current_measurements = []

                current_measurements.append(
                    {
                        "src_addr": src_addr,
                        "dst_addr": dst_addr,
                        "ip_version": ip_version,
                        "rtt": rtt,
                        "event_time": event_time,
                    }
                )

        if current_key is not None:
            flush_partition(current_key, current_measurements)

    finally:
        try:
            train_writer.close()
        except Exception:
            pass
        try:
            test_writer.close()
        except Exception:
            pass
        try:
            con.close()
        except Exception:
            pass

    seconds = time.time() - start
    return ShardStats(
        shard_id=int(shard_id),
        rows_seen=int(rows_seen),
        partitions_seen=int(partitions_seen),
        train_chunks=int(train_chunks),
        test_chunks=int(test_chunks),
        train_tokens=int(train_tokens),
        test_tokens=int(test_tokens),
        seconds=float(seconds),
    )


# -----------------------------
# Driver
# -----------------------------


def discover_shard_ids(temp_shards_dir: Path) -> list[int]:
    shard_ids: list[int] = []
    for p in temp_shards_dir.glob("shard_id=*"):
        if not p.is_dir():
            continue
        name = p.name
        try:
            shard_id = int(name.split("=")[1])
            shard_ids.append(shard_id)
        except Exception:
            continue
    return sorted(shard_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe-centric chunk builder with temporary sharding + parallel ArrayRecord writing"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Parquet path or glob (DuckDB read_parquet() supports globs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory (train/ and test/ subdirs will be created)",
    )

    parser.add_argument(
        "--temp-shards",
        type=str,
        default=None,
        help="Temporary sharded Parquet directory. Defaults to <output>/_temp_shards",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=256,
        help="Number of temp shards to create (default: 256)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of multiprocessing workers for shard processing (default: effective CPU count)",
    )
    parser.add_argument(
        "--duckdb-threads-sharding",
        type=int,
        default=None,
        help="DuckDB threads to use during sharding stage (default: effective CPU count)",
    )
    parser.add_argument(
        "--duckdb-threads-worker",
        type=int,
        default=1,
        help="DuckDB threads per worker during shard processing (default: 1)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio by src_id ordering (default: 0.9)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100000,
        help="Max tokens per chunk before splitting (default: 100000)",
    )

    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=250_000,
        help="Rows to fetch per batch in workers (default: 250k)",
    )

    parser.add_argument(
        "--skip-sharding",
        action="store_true",
        help="Skip Stage A and assume --temp-shards already exists",
    )
    parser.add_argument(
        "--overwrite-temp",
        action="store_true",
        help="Overwrite temp shards directory if it exists",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary shards after completion (default: delete)",
    )

    parser.add_argument(
        "--duckdb-progress",
        action="store_true",
        help="Enable DuckDB progress bars (can be noisy in Modal logs)",
    )

    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_shards_dir = (
        Path(args.temp_shards) if args.temp_shards else (out_dir / "_temp_shards")
    )

    eff_cpus = effective_cpu_count()
    workers = args.workers or eff_cpus

    duckdb_threads_sharding = args.duckdb_threads_sharding or eff_cpus

    # Report environment.
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except Exception:
            affinity = None

    print("=== Probe chunk preprocessing ===")
    print(f"Input:        {args.input}")
    print(f"Output:       {out_dir}")
    print(f"Temp shards:  {temp_shards_dir}")
    print(
        f"CPUs:         effective={eff_cpus}, cpu_count={os.cpu_count()}, affinity={affinity}"
    )
    print(f"Workers:      {workers}")
    print(f"Num shards:   {args.num_shards}")
    print(f"Max tokens:   {args.max_tokens:,}")
    print(f"Train ratio:  {args.train_ratio}")
    print("")

    # Stage 0: probe map.
    print("[Stage 0] Building src_addr -> src_id mapping...")
    con = duckdb_connect(
        threads=duckdb_threads_sharding, enable_progress=args.duckdb_progress
    )
    probe_info = build_probe_map(con, args.input, args.train_ratio, out_dir)
    print(
        f"  ✓ n_probes={probe_info.n_probes:,} train_cutoff_src_id={probe_info.train_cutoff_src_id:,}"
    )

    # Stage A: temporary sharding.
    if not args.skip_sharding:
        print("[Stage A] Writing temporary sharded Parquet dataset...")
        t0 = time.time()
        make_temp_shards(
            con,
            args.input,
            temp_shards_dir,
            args.num_shards,
            overwrite=args.overwrite_temp,
            enable_progress=args.duckdb_progress,
        )
        print(f"  ✓ temp shards written in {time.time() - t0:.1f}s")

    try:
        con.close()
    except Exception:
        pass

    # Discover shard IDs.
    shard_ids = discover_shard_ids(temp_shards_dir)
    if not shard_ids:
        raise RuntimeError(
            f"No shards found under {temp_shards_dir}. "
            "If you used --skip-sharding, verify the directory exists and has shard_id=* subdirs."
        )

    print(f"[Stage B] Processing {len(shard_ids)} shards with {workers} workers...")

    # Multiprocessing: one task per shard.
    from multiprocessing import Pool

    worker_args = dict(
        temp_shards_dir=str(temp_shards_dir),
        output_dir=str(out_dir),
        train_cutoff_src_id=probe_info.train_cutoff_src_id,
        max_tokens_per_record=int(args.max_tokens),
        duckdb_threads=int(args.duckdb_threads_worker),
        row_batch_size=int(args.row_batch_size),
        enable_duckdb_progress=bool(args.duckdb_progress),
    )

    t_stage_b = time.time()

    # Store stats incrementally.
    stats_path = out_dir / "preprocess_stats.jsonl"
    if stats_path.exists():
        stats_path.unlink()

    total_rows = 0
    total_partitions = 0
    total_train_chunks = 0
    total_test_chunks = 0
    total_train_tokens = 0
    total_test_tokens = 0

    with Pool(processes=workers) as pool:
        fn = partial(process_one_shard, **worker_args)
        for stat in pool.imap_unordered(fn, shard_ids, chunksize=1):
            # Aggregate.
            total_rows += stat.rows_seen
            total_partitions += stat.partitions_seen
            total_train_chunks += stat.train_chunks
            total_test_chunks += stat.test_chunks
            total_train_tokens += stat.train_tokens
            total_test_tokens += stat.test_tokens

            # Persist per-shard stats (useful in Modal logs).
            with stats_path.open("a") as f:
                f.write(json.dumps(asdict(stat)) + "\n")

            print(
                f"  shard {stat.shard_id:5d}: rows={stat.rows_seen:,} partitions={stat.partitions_seen:,} "
                f"train_chunks={stat.train_chunks:,} test_chunks={stat.test_chunks:,} "
                f"sec={stat.seconds:.1f}"
            )

    elapsed_b = time.time() - t_stage_b

    # Write a manifest for Grain or other loaders.
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": args.input,
        "output": str(out_dir),
        "temp_shards": str(temp_shards_dir),
        "num_shards_present": len(shard_ids),
        "train_cutoff_src_id": probe_info.train_cutoff_src_id,
        "train_ratio": args.train_ratio,
        "max_tokens_per_record": args.max_tokens,
        "totals": {
            "rows_seen": total_rows,
            "partitions_seen": total_partitions,
            "train_chunks": total_train_chunks,
            "test_chunks": total_test_chunks,
            "train_tokens": total_train_tokens,
            "test_tokens": total_test_tokens,
            "stage_b_seconds": elapsed_b,
        },
        "files": {
            "train": sorted(str(p) for p in (out_dir / "train").glob("*.arrayrecord")),
            "test": sorted(str(p) for p in (out_dir / "test").glob("*.arrayrecord")),
        },
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== Done ===")
    print(f"Stage B time: {elapsed_b/60:.1f} min")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Total partitions:     {total_partitions:,}")
    print(
        f"Train chunks:         {total_train_chunks:,} (tokens={total_train_tokens:,})"
    )
    print(f"Test chunks:          {total_test_chunks:,} (tokens={total_test_tokens:,})")
    print(f"Manifest:             {out_dir / 'manifest.json'}")

    # Clean up temporary shards unless requested.
    if not args.keep_temp:
        print(f"Cleaning up temp shards at {temp_shards_dir} ...")
        for p in temp_shards_dir.glob("**/*"):
            if p.is_file():
                p.unlink()
        for p in sorted(temp_shards_dir.glob("**/*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            temp_shards_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
