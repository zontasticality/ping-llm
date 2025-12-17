"""
Sample random probe chunks on Modal using the Grain pipeline.

Purpose:
- Quick sanity check of ArrayRecord outputs before full training.
- Uses the same ProbeChunkDataSource + ProbeChunkCropper infrastructure as training.
- Prints readable token samples from train/test splits.

Usage:
  modal run scripts/data/modal_sample_probe_chunks.py::sample_chunks

Assumes:
- Modal volume (default: ping-llm) has probe_chunks/{train,test}_shard_*.arrayrecord
- Or single-file probe_chunks/{train,test}.arrayrecord
"""

import os
import random
import sys
from pathlib import Path

from modal import App, Image, Volume

APP_NAME = "ping-llm-sample-probe-chunks"
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
    "*.arrayrecord",
    ".DS_Store",
]

# Build image with the dependencies used by the Grain pipeline
image = (
    Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "ninja-build",
    )
    .pip_install("uv")
    # Stage 1: Copy ONLY files needed for dependency resolution
    .add_local_file("pyproject.toml", f"{WORKDIR}/pyproject.toml", copy=True)
    .add_local_file("README.md", f"{WORKDIR}/README.md", copy=True)
    .add_local_file("build_hooks.py", f"{WORKDIR}/build_hooks.py", copy=True)
    .add_local_dir("dependencies", f"{WORKDIR}/dependencies", copy=True)
    .add_local_file(
        "src/MaxText/__init__.py", f"{WORKDIR}/src/MaxText/__init__.py", copy=True
    )  # Only __init__.py for version
    .add_local_dir(
        "src/install_maxtext_extra_deps",
        f"{WORKDIR}/src/install_maxtext_extra_deps",
        copy=True,
    )
    # Stage 2: Install dependencies (this layer is cached unless above files change)
    .run_commands(
        f"cd {WORKDIR} && CC=gcc CXX=g++ uv pip install --system -e '.[cuda12]' --resolution=lowest",
        f"cd {WORKDIR} && install_maxtext_github_deps",
    )
    .pip_install("wandb")
    # Stage 3: Copy the rest of the code (fast layer that rebuilds on code changes)
    # CACHE BUST: 2025-12-16-12 - Changed to SIGINT in train_with_wandb_sync.py
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
)

app = App(APP_NAME)
shared_vol = Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _find_arrayrecord_files(base_dir: Path, split: str) -> list[Path]:
    """Find ArrayRecord files for a split, supporting single-file or sharded layouts."""
    # Prefer nested sharded layout (e.g., base/train/train_shard_*.arrayrecord)
    nested = sorted((base_dir / split).glob(f"{split}_shard_*.arrayrecord"))
    if nested:
        return nested

    # Fallback to single-file layout (e.g., base/train.arrayrecord)
    top_level = sorted(base_dir.glob(f"{split}*.arrayrecord"))
    if top_level:
        return top_level

    raise FileNotFoundError(
        f"No ArrayRecord files found for split '{split}' in {base_dir}"
    )


def _decode_uint16(data: bytes) -> list[int]:
    import numpy as np

    return np.frombuffer(data, dtype=np.uint16).tolist()


@app.function(
    image=image,
    volumes={"/mnt": shared_vol},
    cpu=4,
    memory=8 * 1024,
    timeout=60 * 15,
)
def sample_chunks(
    data_dir: str = "/mnt/data/probe_chunks",
    num_samples_per_split: int = 3,
    crop_size: int = 128,
    seed: int = 42,
):
    """
    Sample random cropped token windows from train/test ArrayRecord files and print them.

    Args:
        data_dir: Directory containing train/test ArrayRecord files (sharded or single).
        num_samples_per_split: Number of random samples to print per split.
        crop_size: Crop size passed to ProbeChunkCropper (default 128).
        seed: RNG seed for reproducibility.
    """
    sys.path.insert(0, str(Path(WORKDIR)))  # ensure repo imports work

    from src.MaxText.input_pipeline._probe_chunk_datasource import (
        ProbeChunkDataSource,
        ProbeChunkCropper,
    )

    base = Path(data_dir)
    rng = random.Random(seed)

    for split in ["train", "test"]:
        files = _find_arrayrecord_files(base, split)
        print(f"\n== Split: {split} ==")
        print(f"Found {len(files)} ArrayRecord file(s)")
        for f in files[:3]:
            print(f"  - {f}")

        # Sample from one shard (pick random shard for variety)
        shard_path = rng.choice(files)
        print(f"Sampling from shard: {shard_path}")
        source = ProbeChunkDataSource(str(shard_path), build_probe_index=False)
        cropper = ProbeChunkCropper(crop_size=crop_size, seed=seed)

        n = len(source)
        if n == 0:
            print(f"  (no chunks in {files[0]})")
            continue

        indices = [rng.randrange(n) for _ in range(min(num_samples_per_split, n))]
        for i, idx in enumerate(indices, 1):
            chunk = source[idx]
            cropped = cropper.random_map(chunk, rng)

            # ProbeChunkCropper outputs MaxText-format fields; use inputs as the cropped token window
            tokens = cropped["inputs"].tolist()
            seg = cropped.get("inputs_segmentation")
            from tokenization import decode_token_stream_pretty, decode_tokens_to_measurements
            pretty_tokens = decode_token_stream_pretty(tokens)
            measurements = decode_tokens_to_measurements(tokens, segmentation=seg.tolist() if hasattr(seg, "tolist") else seg)
            meta = chunk["metadata"]
            print(f"\n  Sample {i} (idx {idx}):")
            print(
                f"    src_id={chunk['src_id']} part={meta['part_id']} bucket_start={meta['bucket_start_time']}"
            )
            print(
                f"    chunk_tokens={chunk['n_tokens']}, meas={chunk['n_measurements']}, crop_size={len(tokens)}"
            )
            print(f"    token stream (first 64): {pretty_tokens[:64]}")
            if len(tokens) > 64:
                print(f"    token stream (last 16): {pretty_tokens[-16:]}")
            if measurements:
                print("    decoded measurements (first 2):")
                for m_idx, meas in enumerate(measurements[:2], 1):
                    print(f"      m{m_idx}:")
                    for b in meas["blocks"]:
                        print(f"        - {b}")


@app.local_entrypoint()
def main():
    sample_chunks.remote()
