"""
Modal deployment for PLAN_3 probe row preprocessing (streaming version).

This script runs the memory-efficient streaming preprocessor on Modal.
"""

import modal

# Create Modal app
app = modal.App("probe-rows-preprocessing-v2")

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow",
        "duckdb",
        "array_record",
        "psutil",
    )
)

# Mount the ping-llm volume
volume = modal.Volume.from_name("ping-llm-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=3600 * 4,  # 4 hours
    cpu=8.0,  # 8 cores
    memory=32768,  # 32GB RAM
)
def create_probe_rows(
    input_pattern: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/probe_rows",
    max_row_size_mb: float = 8.0,
    train_ratio: float = 0.9,
    workers: int = 8,
    memory_limit_gb: float = 28.0,  # Leave 4GB for system
):
    """
    Run memory-efficient probe row preprocessing on Modal.

    Args:
        input_pattern: Input parquet file pattern
        output_dir: Output directory for ArrayRecord files
        max_row_size_mb: Maximum row size in MB
        train_ratio: Train/test split ratio
        workers: Number of worker processes
        memory_limit_gb: DuckDB memory limit in GB
    """
    import subprocess
    from pathlib import Path

    print("=" * 80)
    print("MODAL: PROBE ROW PREPROCESSING (STREAMING)")
    print("=" * 80)
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers}")
    print(f"Memory limit: {memory_limit_gb}GB")
    print()

    # Copy the streaming script to Modal
    script_path = Path("/tmp/create_probe_rows_streaming.py")

    # Download the script from the repo
    # In practice, you'd mount the repo or include it in the image
    # For now, we'll use the local version

    # Run the streaming preprocessor
    cmd = [
        "python",
        "/root/create_probe_rows_parallel_streaming.py",  # Mounted from local
        "--input", input_pattern,
        "--output", output_dir,
        "--max-row-size-mb", str(max_row_size_mb),
        "--train-ratio", str(train_ratio),
        "--workers", str(workers),
        "--memory-limit-gb", str(memory_limit_gb),
    ]

    print("Running command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, check=True)

    # Commit volume
    volume.commit()

    print("\n✓ Preprocessing complete!")
    print(f"Output: {output_dir}")
    print("Volume committed successfully")


@app.local_entrypoint()
def main(
    input_pattern: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/probe_rows",
):
    """Local entrypoint to trigger preprocessing."""
    print("Starting probe row preprocessing on Modal (streaming version)...")
    create_probe_rows.remote(
        input_pattern=input_pattern,
        output_dir=output_dir,
    )
    print("✓ Done!")
