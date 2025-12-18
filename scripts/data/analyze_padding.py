#!/usr/bin/env python3
"""
Analyze padding waste for ArrayRecord chunks.

Given a crop size, compute how much padding would be added per chunk (max(0, crop_size - n_tokens))
and print summary statistics + histogram buckets.

Usage:
  python scripts/data/analyze_padding.py \
    --data-dir data/probe_chunks \
    --crop-size 1024

Supports sharded layout (data_dir/{train,test}/{train,test}_shard_*.arrayrecord) or single files
(data_dir/{train,test}.arrayrecord).
"""

import argparse
import sys
from pathlib import Path
import random
import numpy as np

# Ensure repo root is on sys.path for MaxText imports when running locally or in Modal
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MaxText.input_pipeline.probe_chunk_pipeline import build_probe_chunk_dataset


def find_arrayrecord_files(base_dir: Path, split: str) -> list[Path]:
    nested = sorted((base_dir / split).glob(f"{split}_shard_*.arrayrecord"))
    if nested:
        return nested
    top_level = sorted(base_dir.glob(f"{split}*.arrayrecord"))
    if top_level:
        return top_level
    raise FileNotFoundError(f"No ArrayRecord files for split '{split}' in {base_dir}")


def analyze_split(
    files: list[Path],
    crop_size: int,
    max_records: int,
    sample_rate: float,
    seed: int,
    num_workers: int,
    prefetch_buffer_size: int,
):
    """
    Sample random crops using the same Grain pipeline as training.

    We pick random shards, build the dataset with shuffle enabled, and draw
    examples until we hit max_records or exhaust the sample budget.
    """
    rng = random.Random(seed)
    pad_samples = []
    len_samples = []
    pad_frac_samples = []
    total_chunks = 0
    sampled = 0

    # Build a shuffled list of shards to draw from
    shard_paths = files[:]
    rng.shuffle(shard_paths)

    for path in shard_paths:
        # Build dataset for this shard (batch_size=1 to inspect individual crops)
        dataset = build_probe_chunk_dataset(
            arrayrecord_path=str(path),
            batch_size=1,
            crop_size=crop_size,
            shuffle=True,
            shuffle_seed=seed,
            num_workers=num_workers,
            prefetch_buffer_size=prefetch_buffer_size,
        )

        # Count chunks in shard
        try:
            from MaxText.input_pipeline._probe_chunk_datasource import (
                ProbeChunkDataSource,
            )

            total_chunks += len(ProbeChunkDataSource(str(path)))
        except Exception:
            pass

        # Determine how many samples to draw from this shard.
        # If sampling (<1.0), base it on shard size; never exceed remaining budget.
        remaining = max_records - sampled
        if remaining <= 0:
            break
        if sample_rate >= 1.0:
            target = min(remaining, len(dataset))
        else:
            target = min(remaining, max(1, int(len(dataset) * sample_rate)))

        it = iter(dataset)
        for i in range(target):
            try:
                batch = next(it)
            except StopIteration:
                break
            # batch is dict of arrays with leading batch dim = 1
            inputs = batch["inputs"][0]
            seg = batch["inputs_segmentation"][0]
            valid_len = int(seg.sum())
            pad = crop_size - valid_len
            pad_samples.append(pad)
            len_samples.append(valid_len)
            pad_frac_samples.append(pad / crop_size)
            sampled += 1
            if sampled >= max_records:
                break
            if sampled % 5000 == 0:
                print(f"    sampled {sampled} crops so far (current shard: {path})")

        print(
            f"  sampled {min(target, sampled)} records from {path} (total sampled: {sampled})"
        )
        if sampled >= max_records:
            break

    pads = np.array(pad_samples, dtype=np.int64)
    lengths = np.array(len_samples, dtype=np.int64)
    pad_fracs = np.array(pad_frac_samples, dtype=np.float64)

    hist_pad, edges_pad = np.histogram(
        pads, bins=[0, 1, 8, 32, 128, 512, 1024, 2048, 1e9]
    )
    hist_len, edges_len = np.histogram(
        lengths, bins=[0, 64, 128, 256, 512, 768, 1024, 2048, 1e9]
    )
    hist_padfrac, edges_padfrac = np.histogram(
        pad_fracs, bins=[0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 10.0]
    )
    return {
        "count": total_chunks,
        "sampled": len(pads),
        "mean_pad": pads.mean() if len(pads) else 0,
        "p50": np.percentile(pads, 50) if len(pads) else 0,
        "p90": np.percentile(pads, 90) if len(pads) else 0,
        "p99": np.percentile(pads, 99) if len(pads) else 0,
        "max": pads.max() if len(pads) else 0,
        "mean_len": lengths.mean() if len(lengths) else 0,
        "p50_len": np.percentile(lengths, 50) if len(lengths) else 0,
        "p90_len": np.percentile(lengths, 90) if len(lengths) else 0,
        "p99_len": np.percentile(lengths, 99) if len(lengths) else 0,
        "max_len": lengths.max() if len(lengths) else 0,
        "mean_padfrac": pad_fracs.mean() if len(pad_fracs) else 0,
        "p50_padfrac": np.percentile(pad_fracs, 50) if len(pad_fracs) else 0,
        "p90_padfrac": np.percentile(pad_fracs, 90) if len(pad_fracs) else 0,
        "p99_padfrac": np.percentile(pad_fracs, 99) if len(pad_fracs) else 0,
        "hist_pad": hist_pad.tolist(),
        "edges_pad": edges_pad.tolist(),
        "hist_len": hist_len.tolist(),
        "edges_len": edges_len.tolist(),
        "hist_padfrac": hist_padfrac.tolist(),
        "edges_padfrac": edges_padfrac.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze padding for ArrayRecord chunks."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/probe_chunks",
        help="Directory with ArrayRecord files",
    )
    parser.add_argument(
        "--crop-size", type=int, default=1024, help="Crop size used for training"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=50000,
        help="Maximum records to scan per split",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Bernoulli sampling rate (0<r<=1). If <1.0, uses sampling; if 1.0, scans up to max-records.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Grain read threads (0 = no threading)",
    )
    parser.add_argument(
        "--prefetch-buffer-size",
        type=int,
        default=2,
        help="Prefetch buffer per Grain worker",
    )
    args = parser.parse_args()

    base = Path(args.data_dir)
    for split in ["train", "test"]:
        files = find_arrayrecord_files(base, split)
        stats = analyze_split(
            files,
            args.crop_size,
            args.max_records,
            args.sample_rate,
            args.seed,
            args.num_workers,
            args.prefetch_buffer_size,
        )
        print(f"\n== {split} ==")
        print(f"chunks: {stats['count']:,}")
        print(
            f"sampled: {stats['sampled']:,} (rate={args.sample_rate}, cap={args.max_records})"
        )
        print(
            f"padding tokens (if crop_size={args.crop_size}): "
            f"mean={stats['mean_pad']:.1f}, p50={stats['p50']:.0f}, p90={stats['p90']:.0f}, "
            f"p99={stats['p99']:.0f}, max={stats['max']}"
        )
        print("hist buckets (pad counts):")
        edges = stats["edges_pad"]
        hist = stats["hist_pad"]
        for i, count in enumerate(hist):
            lo = edges[i]
            hi = edges[i + 1]
            if i == len(hist) - 1:
                print(f"  pad [{lo}, inf): {count}")
            else:
                print(f"  pad [{lo}, {hi}): {count}")

        print(
            f"sequence length: mean={stats['mean_len']:.1f}, p50={stats['p50_len']:.0f}, "
            f"p90={stats['p90_len']:.0f}, p99={stats['p99_len']:.0f}, max={stats['max_len']}"
        )
        edges = stats["edges_len"]
        hist = stats["hist_len"]
        for i, count in enumerate(hist):
            lo = edges[i]
            hi = edges[i + 1]
            if i == len(hist) - 1:
                print(f"  len [{lo}, inf): {count}")
            else:
                print(f"  len [{lo}, {hi}): {count}")

        print(
            f"padding fraction (pad/crop_size): "
            f"mean={stats['mean_padfrac']:.3f}, p50={stats['p50_padfrac']:.3f}, "
            f"p90={stats['p90_padfrac']:.3f}, p99={stats['p99_padfrac']:.3f}"
        )
        edges = stats["edges_padfrac"]
        hist = stats["hist_padfrac"]
        for i, count in enumerate(hist):
            lo = edges[i]
            hi = edges[i + 1]
            if i == len(hist) - 1:
                print(f"  padfrac [{lo:.2f}, inf): {count}")
            else:
                print(f"  padfrac [{lo:.2f}, {hi:.2f}): {count}")


# Optional Modal entrypoint for remote execution
try:
    from modal import App, Image, Volume

    app = App("ping-llm-analyze-padding")
    WORKDIR = "/workspace"
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
        .uv_pip_install(
            "pyarrow",
            "numpy",
            "array_record",
            "grain",
            "modal",  # ensure modal is available inside the container
        )
        # Stage 3: Copy the rest of the code (fast layer that rebuilds on code changes)
        # CACHE BUST: 2025-12-16-12 - Changed to SIGINT in train_with_wandb_sync.py
        .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=False)
    )
    shared_vol = Volume.from_name("ping-llm", create_if_missing=True)

    @app.function(
        image=image,
        volumes={"/mnt": shared_vol},
        timeout=60 * 10,
        memory=4 * 1024,
        cpu=4,
    )
    def run_modal(
        data_dir: str = "/mnt/data/probe_chunks",
        crop_size: int = 1024,
        max_records: int = 50000,
        sample_rate: float = 0.05,
        seed: int = 42,
        num_workers: int = 0,
        prefetch_buffer_size: int = 2,
    ):
        base = Path(data_dir)
        for split in ["train", "test"]:
            files = find_arrayrecord_files(base, split)
            stats = analyze_split(
                files,
                crop_size,
                max_records,
                sample_rate,
                seed,
                num_workers,
                prefetch_buffer_size,
            )
            print(f"\n== {split} ==")
            print(f"chunks: {stats['count']:,}")
            print(
                f"sampled: {stats['sampled']:,} (rate={sample_rate}, cap={max_records})"
            )
            print(
                f"padding tokens (if crop_size={crop_size}): "
                f"mean={stats['mean_pad']:.1f}, p50={stats['p50']:.0f}, p90={stats['p90']:.0f}, "
                f"p99={stats['p99']:.0f}, max={stats['max']}"
            )
            print("hist buckets (pad counts):")
            edges = stats["edges_pad"]
            hist = stats["hist_pad"]
            for i, count in enumerate(hist):
                lo = edges[i]
                hi = edges[i + 1]
                if i == len(hist) - 1:
                    print(f"  pad [{lo}, inf): {count}")
                else:
                    print(f"  pad [{lo}, {hi}): {count}")

            print(
                f"sequence length: mean={stats['mean_len']:.1f}, p50={stats['p50_len']:.0f}, "
                f"p90={stats['p90_len']:.0f}, p99={stats['p99_len']:.0f}, max={stats['max_len']}"
            )
            edges = stats["edges_len"]
            hist = stats["hist_len"]
            for i, count in enumerate(hist):
                lo = edges[i]
                hi = edges[i + 1]
                if i == len(hist) - 1:
                    print(f"  len [{lo}, inf): {count}")
                else:
                    print(f"  len [{lo}, {hi}): {count}")

            print(
                f"padding fraction (pad/crop_size): "
                f"mean={stats['mean_padfrac']:.3f}, p50={stats['p50_padfrac']:.3f}, "
                f"p90={stats['p90_padfrac']:.3f}, p99={stats['p99_padfrac']:.3f}"
            )
            edges = stats["edges_padfrac"]
            hist = stats["hist_padfrac"]
            for i, count in enumerate(hist):
                lo = edges[i]
                hi = edges[i + 1]
                if i == len(hist) - 1:
                    print(f"  padfrac [{lo:.2f}, inf): {count}")
                else:
                    print(f"  padfrac [{lo:.2f}, {hi:.2f}): {count}")

    @app.local_entrypoint()
    def _local_entrypoint():
        main()

except ImportError:
    if __name__ == "__main__":
        main()
