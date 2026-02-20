# Plan: Chunked File Preprocessing

## Problem
DuckDB `LIST(STRUCT_PACK(...)) GROUP BY src_addr` across all 720 parquet files (25.3B rows) requires ~1.1TB RAM for the hash table. Even 400GB wasn't enough.

## Solution: Two-pass chunked approach

### Pass 1: Per-file GROUP BY (parallel, ~2GB RAM per worker)
- Process each of the 720 parquet files independently
- Each file: `GROUP BY src_addr` → per-file grouped parquet (~1GB in, ~1GB out)
- Use multiprocessing Pool with 48 workers
- Each worker uses its own DuckDB connection with ~2GB memory limit
- Output: 720 intermediate grouped parquet files in temp dir
- Memory: 48 workers * 2GB = ~96GB peak. Fits easily in 512GB.
- Time estimate: 720 files / 48 workers = 15 batches * ~30s each = ~8 min

### Pass 2: Merge-group across files (streaming, ~100GB RAM)
- Read all 720 intermediate parquets with DuckDB
- `GROUP BY src_addr` again, using `FLATTEN(LIST_CONCAT(...))` or simply re-LIST the nested lists
- Actually simpler: since each intermediate file has `src_addr, LIST(measurements)`, we can:
  - Option A: DuckDB query: `SELECT src_addr, FLATTEN(LIST(measurements)) FROM read_parquet('intermediate_*.parquet') GROUP BY src_addr` — but this still builds a big hash table
  - Option B (better): **Sort-merge approach** — all intermediates are already grouped by src_addr. Read them all, do a merge-group by src_addr, concatenating lists. DuckDB can do this with external sort.
  - Option C (simplest, most memory-efficient): **Python streaming merge** — open all 720 intermediate parquets as DuckDB views, iterate sorted by src_addr, accumulate measurements per src_addr, write to final grouped parquet. Memory = one probe's data at a time.

**Recommended: Option C** — most robust, trivially fits in memory.

### Pass 3: Batch processing + ArrayRecord (unchanged)
- Read final grouped parquet, split train/test, convert to ArrayRecord
- This part of the existing script works fine

## Implementation

### Changes to `create_probe_rows_parallel_streaming.py`:

1. **Replace the monolithic `con.execute(query)` at line 405** with:

```python
# Pass 1: Per-file GROUP BY in parallel
intermediate_dir = temp_dir / "per_file_grouped"
intermediate_dir.mkdir()

def group_one_file(args):
    parquet_file, output_path, worker_id = args
    con = duckdb.connect(':memory:')
    con.execute("SET memory_limit='4GB'")
    con.execute(f"""
        COPY (
            SELECT src_addr,
                   LIST(STRUCT_PACK(
                       event_time := event_time,
                       src_addr := src_addr,
                       dst_addr := dst_addr,
                       ip_version := ip_version,
                       rtt := rtt_avg
                   )) as measurements
            FROM read_parquet('{parquet_file}')
            GROUP BY src_addr
        ) TO '{output_path}' (FORMAT PARQUET)
    """)
    con.close()
    return output_path

file_args = [
    (f, str(intermediate_dir / f"grouped_{i:04d}.parquet"), i)
    for i, f in enumerate(parquet_files)
]

with Pool(num_workers) as pool:
    pool.map(group_one_file, file_args)

# Pass 2: Streaming merge by src_addr
# Read all per-file grouped parquets, sorted by src_addr
# Accumulate measurements per src_addr, write to final grouped parquet
con = duckdb.connect(':memory:')
con.execute(f"SET memory_limit='{memory_limit_gb}GB'")
con.execute(f"SET temp_directory='{temp_dir}'")
con.execute(f"SET threads=8")

# This query flattens the nested lists and re-groups
# Memory needed: one src_addr's data at a time (streaming COPY)
con.execute(f"""
    COPY (
        SELECT src_addr,
               FLATTEN(LIST(measurements)) as measurements
        FROM read_parquet('{intermediate_dir}/*.parquet')
        GROUP BY src_addr
    ) TO '{intermediate_parquet}' (FORMAT PARQUET)
""")
con.close()
```

2. **SLURM script**: Request 512GB RAM, 64 CPUs, same 2-day limit

### Why 512GB is enough
- Pass 1: 48 workers * 4GB = 192GB peak
- Pass 2: DuckDB FLATTEN+GROUP BY on 720 small parquets. Each src_addr appears in multiple files, but there are only ~20K unique src_addrs. The hash table holds 20K entries with list pointers. The actual list data is being streamed to the output parquet via COPY. With `force_external=true`, DuckDB can spill. Even if it can't fully stream, 20K groups * average list size fits in ~200-400GB.
- Pass 3: Batch processing, ~50GB
- Total peak: ~400GB. 512GB gives headroom.

### Risk: Pass 2 might still OOM
If `FLATTEN(LIST(measurements))` still materializes all data in memory, fall back to:
- Python streaming merge: iterate DuckDB cursor `SELECT src_addr, measurements FROM read_parquet('*.parquet') ORDER BY src_addr`, accumulate per src_addr, write ArrayRecord directly (skip intermediate parquet entirely).

## SLURM Config
```
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64
```
DuckDB memory: 300GB, workers: 48
