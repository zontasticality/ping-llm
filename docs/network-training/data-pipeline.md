# Data Pipeline Guide

## Overview

The training pipeline converts raw parquet measurement files into ArrayRecord format for efficient training:

```
data/parquet_ping/*.parquet  (raw hourly snapshots)
    -> DuckDB GROUP BY src_addr
    -> ArrayRecord probe rows (one row per source IP)
    -> Runtime tokenization during training
```

## Preprocessing: Parquet to ArrayRecord

Use `create_probe_rows_parallel_streaming.py` to convert raw parquet files into probe-centric ArrayRecord rows.

```bash
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/parquet_ping/*.parquet" \
  --output data/probe_rows \
  --workers 8
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Glob pattern for input parquet files |
| `--output` | (required) | Output directory for ArrayRecord files |
| `--max-row-size-mb` | 8.0 | Maximum row size in MB (large probes split across rows) |
| `--train-ratio` | 0.9 | Train/test split ratio |
| `--workers` | CPU count | Number of worker processes |
| `--memory-limit-gb` | auto | DuckDB memory limit (default: available RAM - 1GB) |
| `--no-assume-ordered` | false | Add ORDER BY src_addr (slower but deterministic ordering) |

### Output

```
data/probe_rows/
  train.arrayrecord    # 90% of probes
  test.arrayrecord     # 10% of probes
```

### How It Works

1. **Stream grouping**: DuckDB groups all measurements by `src_addr` and writes to an intermediate parquet file on disk (memory-safe).
2. **Batch processing**: Workers read batches from the intermediate file, sort measurements by timestamp per probe, serialize to PyArrow IPC, and write ArrayRecord entries.
3. **Row splitting**: Probes larger than `--max-row-size-mb` are split across multiple ArrayRecord rows (same `src_id`).
4. **Merge**: Partial ArrayRecord files from each worker are merged into final train/test files.

### Performance

For ~200M measurements on a 32GB system with 8 workers:
- Time: ~30 minutes
- Memory: configurable (auto-detects available RAM)
- Disk: ~10GB temp space

### Modal Deployment

```bash
modal run scripts/data/modal_create_probe_rows_parallel_streaming.py
```

---

## Memory Limit Guidelines

The streaming script auto-detects available memory by default. To set explicitly:

```
16GB system  ->  --memory-limit-gb 14  (leave 2GB)
32GB system  ->  --memory-limit-gb 28  (leave 4GB)
64GB system  ->  --memory-limit-gb 60  (leave 4GB)
```

---

## Inspection Tools

Several scripts are available for inspecting preprocessed data:

| Script | Purpose |
|--------|---------|
| `scripts/data/quick_inspect_probe_rows.py` | Quick stats, violations, ordering checks |
| `scripts/data/inspect_probe_rows.py` | Detailed row sizes, measurement distributions |
| `scripts/data/verify_tokenization.py` | Validate tokenization correctness |
| `scripts/data/analyze_padding.py` | Analyze padding waste for different crop sizes |

---

## Troubleshooting

### OOM Error

```
_duckdb.OutOfMemoryException: failed to allocate data
```

Solutions:
1. Set `--memory-limit-gb` lower
2. Reduce `--workers` count

### Disk Space Error

```
No space left on device
```

Solutions:
1. Streaming version uses temp directory (~10GB)
2. Set `TMPDIR=/path/to/large/disk`
