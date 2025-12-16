"""Shard a large Parquet on Modal CPU (>=64GB RAM).

Usage:
  modal run scripts/train/modal_shard.py::shard

Assumes:
  - Modal volume `ping-llm` (or set MODAL_VOLUME) has /data/training_data.parquet
  - Writes shards to /mnt/data/sharded/{train,test}
"""

import os
import subprocess

from modal import App, Image, Volume

APP_NAME = "ping-llm-shard"
WORKDIR = "/workspace"
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")

# Ignore heavy paths when copying code
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
    ".DS_Store",
]

image = (
    Image.debian_slim(python_version="3.12")
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
    .pip_install("pyarrow", "pandas")
)

app = App(APP_NAME)
shared_vol = Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": shared_vol},
    cpu=32,
    memory=64 * 1024,  # 64GB
    timeout=60 * 60 * 4,  # 4 hours
)
def shard(
    input_path: str = "/mnt/data/training_data.parquet",
    output_dir: str = "/mnt/data/sharded",
    train_shards: int = 180,
    test_shards: int = 20,
):
    """Shard the large Parquet into train/test splits on CPU."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python",
        "scripts/data/shard_parquet.py",
        "--input",
        input_path,
        "--output",
        output_dir,
        "--train-shards",
        str(train_shards),
        "--test-shards",
        str(test_shards),
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR)
