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
from pathlib import Path
import numpy as np

try:
    import array_record.python.array_record_module as arm
except ImportError:
    raise ImportError("array_record not installed. Install with: pip install array_record")

import pyarrow as pa
import pyarrow.ipc as ipc


def find_arrayrecord_files(base_dir: Path, split: str) -> list[Path]:
    nested = sorted((base_dir / split).glob(f"{split}_shard_*.arrayrecord"))
    if nested:
        return nested
    top_level = sorted(base_dir.glob(f"{split}*.arrayrecord"))
    if top_level:
        return top_level
    raise FileNotFoundError(f"No ArrayRecord files for split '{split}' in {base_dir}")


def n_tokens_from_record(reader, idx: int) -> int:
    record_bytes = reader.read([idx])[0]
    batch = ipc.open_stream(record_bytes).read_next_batch()
    return batch.column("n_tokens")[0].as_py()


def analyze_split(files: list[Path], crop_size: int):
    pad_amounts = []
    total_chunks = 0

    for path in files:
        reader = arm.ArrayRecordReader(str(path))
        length = reader.num_records()
        total_chunks += length
        for i in range(length):
            ntoks = n_tokens_from_record(reader, i)
            pad = max(0, crop_size - ntoks)
            pad_amounts.append(pad)
        reader.close()

    pads = np.array(pad_amounts, dtype=np.int64)
    return {
        "count": total_chunks,
        "mean_pad": pads.mean() if len(pads) else 0,
        "p50": np.percentile(pads, 50) if len(pads) else 0,
        "p90": np.percentile(pads, 90) if len(pads) else 0,
        "p99": np.percentile(pads, 99) if len(pads) else 0,
        "max": pads.max() if len(pads) else 0,
        "hist": np.histogram(pads, bins=[0, 1, 8, 32, 128, 512, 1024, 2048, 1e9])[0].tolist(),
        "hist_bins": [0, 1, 8, 32, 128, 512, 1024, 2048, "inf"],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze padding for ArrayRecord chunks.")
    parser.add_argument("--data-dir", type=str, default="data/probe_chunks", help="Directory with ArrayRecord files")
    parser.add_argument("--crop-size", type=int, default=1024, help="Crop size used for training")
    args = parser.parse_args()

    base = Path(args.data_dir)
    for split in ["train", "test"]:
        files = find_arrayrecord_files(base, split)
        stats = analyze_split(files, args.crop_size)
        print(f"\n== {split} ==")
        print(f"chunks: {stats['count']:,}")
        print(
            f"padding tokens (if crop_size={args.crop_size}): "
            f"mean={stats['mean_pad']:.1f}, p50={stats['p50']:.0f}, p90={stats['p90']:.0f}, "
            f"p99={stats['p99']:.0f}, max={stats['max']}"
        )
        print("hist buckets (pad counts):")
        for bucket, count in zip(stats["hist_bins"], stats["hist"]):
            print(f"  <= {bucket}: {count}")


# Optional Modal entrypoint for remote execution
try:
    from modal import App, Image, Volume

    app = App("ping-llm-analyze-padding")
    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install("pyarrow", "numpy", "array_record")
        .add_local_dir(".", "/workspace", ignore=[".git", "__pycache__", ".venv", "*.parquet", "*.arrayrecord"])
    )
    shared_vol = Volume.from_name("ping-llm", create_if_missing=True)

    @app.function(image=image, volumes={"/mnt": shared_vol}, timeout=60 * 10, memory=4 * 1024)
    def run_modal(data_dir: str = "/mnt/data/probe_chunks", crop_size: int = 1024):
        main_args = argparse.Namespace(data_dir=data_dir, crop_size=crop_size)
        # Reuse the analysis logic
        base = Path(main_args.data_dir)
        for split in ["train", "test"]:
            files = find_arrayrecord_files(base, split)
            stats = analyze_split(files, main_args.crop_size)
            print(f"\n== {split} ==")
            print(f"chunks: {stats['count']:,}")
            print(
                f"padding tokens (if crop_size={main_args.crop_size}): "
                f"mean={stats['mean_pad']:.1f}, p50={stats['p50']:.0f}, p90={stats['p90']:.0f}, "
                f"p99={stats['p99']:.0f}, max={stats['max']}"
            )
            print("hist buckets (pad counts):")
            for bucket, count in zip(stats["hist_bins"], stats["hist"]):
                print(f"  <= {bucket}: {count}")

    @app.local_entrypoint()
    def _local_entrypoint():
        main()

except ImportError:
    if __name__ == "__main__":
        main()
