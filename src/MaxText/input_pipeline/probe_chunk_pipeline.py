"""
Shared helpers for building the probe-chunk Grain pipeline.

This centralizes dataset construction so both training and analysis scripts can
re-use the exact same pipeline:
  - ProbeChunkDataSource (ArrayRecord-backed)
  - Shuffle
  - ProbeChunkCropper (measurement-aligned cropping + timestamp masking)
  - Batching + optional worker threads

Inputs:
  - arrayrecord_path: path to a single ArrayRecord file (train or test shard/file)
  - batch_size: number of cropped examples per batch
  - crop_size: tokens per example (padding applied inside the cropper)
  - shuffle / shuffle_seed: controls shuffling
  - num_workers / prefetch_buffer_size: Grain read options
  - build_probe_index: optional src_id -> indices map (slow; for client-first sampling)

Outputs:
  - grain.IterDataset yielding MaxText-compatible dicts:
      inputs, inputs_segmentation, inputs_position,
      targets, targets_segmentation, targets_position
"""

from pathlib import Path
import grain.python as grain
from MaxText.input_pipeline._probe_chunk_datasource import (
    ProbeChunkDataSource,
    ProbeChunkCropper,
)


def build_probe_chunk_dataset(
    arrayrecord_path: str,
    batch_size: int = 32,
    crop_size: int = 1024,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    prefetch_buffer_size: int = 2,
    build_probe_index: bool = False,
) -> grain.IterDataset:
    """
    Construct the probe-chunk Grain pipeline (DATA_LOADING_PLAN_1).

    Args:
        arrayrecord_path: Path to ArrayRecord file (single shard/file)
        batch_size: Number of cropped examples per batch
        crop_size: Tokens per example (padding applied as needed)
        shuffle: Whether to shuffle chunk order
        shuffle_seed: Seed for shuffle + cropper RNG
        num_workers: Grain read threads (0 disables threading)
        prefetch_buffer_size: Prefetch buffer per worker
        build_probe_index: Build src_id -> indices map (optional; slower init)

    Returns:
        grain.IterDataset ready for consumption (training or analysis)
    """
    path = Path(arrayrecord_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ArrayRecord file not found: {path}")

    source = ProbeChunkDataSource(
        arrayrecord_path=str(path),
        build_probe_index=build_probe_index,
    )

    dataset = grain.MapDataset.source(source)
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)

    cropper = ProbeChunkCropper(
        crop_size=crop_size,
        seed=shuffle_seed,
    )
    dataset = dataset.random_map(cropper, seed=shuffle_seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if num_workers > 0:
        dataset = dataset.to_iter_dataset(
            read_options=grain.ReadOptions(
                num_threads=num_workers,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
    else:
        dataset = dataset.to_iter_dataset()

    return dataset
