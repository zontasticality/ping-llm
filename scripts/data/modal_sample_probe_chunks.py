"""
Sample random probe rows on Modal using the Grain pipeline (PLAN_3).

Purpose:
- Quick sanity check of ArrayRecord outputs before full training
- Uses the same ProbeRowDataSource + ProbeRowSampler infrastructure as training
- Prints readable token samples from train/test splits
- Measures tokens/second throughput

Usage:
  modal run scripts/data/modal_sample_probe_chunks.py::sample_rows

Assumes:
- Modal volume (default: ping-llm) has probe_rows/train.arrayrecord and probe_rows/test.arrayrecord
"""

import os
import sys
import time
from pathlib import Path

from modal import App, Image, Volume

APP_NAME = "ping-llm-sample-probe-rows"
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
    # CACHE BUST: 2025-12-18-01 - PLAN_3 refactor
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=False)
)

app = App(APP_NAME)
shared_vol = Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={"/mnt": shared_vol},
    cpu=4,
    memory=8 * 1024,
    timeout=60 * 15,
)
def sample_rows(
    data_dir: str = "/mnt/data/probe_rows",
    num_samples_per_split: int = 3,
    crop_size: int = 128,
    seed: int = 42,
    measure_throughput: bool = True,
    throughput_batches: int = 100,
):
    """
    Sample random cropped token windows from train/test ArrayRecord files and print them.
    Also measure tokens/second throughput.

    Args:
        data_dir: Directory containing train.arrayrecord and test.arrayrecord
        num_samples_per_split: Number of random samples to print per split
        crop_size: Crop size passed to ProbeRowSampler (default 128)
        seed: RNG seed for reproducibility
        measure_throughput: Whether to measure tokens/second throughput
        throughput_batches: Number of batches to use for throughput measurement
    """
    sys.path.insert(0, str(Path(WORKDIR)))  # ensure repo imports work

    from src.MaxText.input_pipeline._probe_chunk_datasource import (
        ProbeRowDataSource,
        ProbeRowSampler,
    )
    from tokenization import (
        decode_token_stream_pretty,
        decode_tokens_to_measurements,
    )

    base = Path(data_dir)

    for split in ["train", "test"]:
        arrayrecord_file = base / f"{split}.arrayrecord"

        if not arrayrecord_file.exists():
            print(f"\n== Split: {split} ==")
            print(f"ArrayRecord file not found: {arrayrecord_file}")
            continue

        print(f"\n== Split: {split} ==")
        print(f"ArrayRecord file: {arrayrecord_file}")

        # Create data source
        source = ProbeRowDataSource(str(arrayrecord_file))
        print(f"Total rows: {len(source):,}")

        # Create sampler
        sampler = ProbeRowSampler(crop_size=crop_size, seed=seed)

        # Sample random rows
        import random
        rng = random.Random(seed)
        n = len(source)

        if n == 0:
            print(f"  (no rows in {arrayrecord_file})")
            continue

        indices = [rng.randrange(n) for _ in range(min(num_samples_per_split, n))]

        for i, idx in enumerate(indices, 1):
            row = source[idx]

            # Generate one context using the sampler
            context = sampler.random_map(row, rng)

            # Extract token stream and segmentation
            tokens = context["inputs"].tolist()
            seg = context["inputs_segmentation"]

            # Decode for display
            pretty_tokens = decode_token_stream_pretty(tokens)
            measurements = decode_tokens_to_measurements(
                tokens, segmentation=seg.tolist() if hasattr(seg, "tolist") else seg
            )

            meta = row["metadata"]
            print(f"\n  Sample {i} (row {idx}):")
            print(f"    src_id={row['src_id']}")
            print(f"    n_measurements={row['n_measurements']:,}")
            print(f"    time_span={meta['time_span_seconds']:.1f}s")
            print(f"    first_timestamp={meta['first_timestamp']}")
            print(f"    last_timestamp={meta['last_timestamp']}")
            print(f"    crop_size={crop_size}, actual_tokens={seg.sum()}")
            print(f"    token stream (first 64): {pretty_tokens[:64]}")
            if len(tokens) > 64:
                print(f"    token stream (last 16): {pretty_tokens[-16:]}")
            if measurements:
                print("    decoded measurements (first 2):")
                for m_idx, meas in enumerate(measurements[:2], 1):
                    print(f"      m{m_idx}:")
                    for b in meas["blocks"]:
                        print(f"        - {b}")

        # Measure throughput if requested
        if measure_throughput and n > 0:
            print(f"\n  === Throughput Test ({throughput_batches} batches, crop_size={crop_size}) ===")

            start_time = time.time()
            total_tokens = 0

            for batch_idx in range(throughput_batches):
                # Sample random row
                row_idx = rng.randrange(n)
                row = source[row_idx]

                # Generate context
                context = sampler.random_map(row, rng)

                # Count real tokens (not padding)
                seg = context["inputs_segmentation"]
                total_tokens += seg.sum()

            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed

            print(f"    Total tokens generated: {total_tokens:,}")
            print(f"    Time elapsed: {elapsed:.2f}s")
            print(f"    Throughput: {tokens_per_sec:,.0f} tokens/sec")
            print(f"    Throughput: {tokens_per_sec / 1000:.1f}K tokens/sec")


@app.local_entrypoint()
def main():
    sample_rows.remote()
