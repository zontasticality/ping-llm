"""
Modal deployment for streaming probe row preprocessing.

This uses the actual implementation from create_probe_rows_parallel_streaming.py
instead of embedding duplicated code inline.
"""

import modal
from pathlib import Path

app = modal.App("probe-rows-streaming")

# Image with all dependencies and local scripts mounted
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow",
        "duckdb",
        "array_record",
        "psutil",
        "pytz",
    )
    .add_local_python_source("scripts")
)

# Mount the volume
volume = modal.Volume.from_name("ping-llm", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=3600 * 6,  # 6 hours (generous)
    cpu=8.0,
    memory=128 * 1024,  # 64GB
)
def preprocess(
    input_pattern: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/probe_rows",
    workers: int = 8,
    memory_limit_gb: float = 120.0,
    assume_ordered: bool = False,
):
    """
    Run streaming preprocessor using the actual implementation.

    Args:
        input_pattern: Input parquet file pattern
        output_dir: Output directory
        workers: Number of parallel workers
        memory_limit_gb: DuckDB memory limit
        assume_ordered: Assume data is pre-ordered (recommended for 200M+ rows).
                       This removes ORDER BY clauses that can cause OOM errors.
                       Set to False if your data is NOT pre-sorted by (src_addr, event_time).
    """
    from scripts.data.create_probe_rows_parallel_streaming import (
        process_parquet_to_probe_rows_streaming,
    )

    print("=" * 80)
    print("MODAL: Streaming Probe Row Preprocessing")
    print("=" * 80)
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers}")
    print(f"Memory limit: {memory_limit_gb}GB")
    print(f"Assume ordered: {assume_ordered}")
    print()

    # Call the actual implementation
    process_parquet_to_probe_rows_streaming(
        input_pattern=input_pattern,
        output_dir=Path(output_dir),
        max_row_size_mb=8.0,
        train_ratio=0.9,
        num_workers=workers,
        memory_limit_gb=memory_limit_gb,
        assume_ordered=assume_ordered,
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
