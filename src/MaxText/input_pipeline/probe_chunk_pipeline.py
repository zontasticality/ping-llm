"""
Shared helpers for building the probe-row Grain pipeline (DATA_LOADING_PLAN_3).

This centralizes dataset construction so both training and analysis scripts can
re-use the exact same pipeline:
  - ProbeRowDataSource (ArrayRecord-backed)
  - Shuffle
  - ProbeRowSampler (multi-scale temporal sampling + timestamp modes)
  - K contexts per row generation
  - Batching + optional worker threads

Inputs:
  - arrayrecord_path: path to a single ArrayRecord file (train or test shard/file)
  - batch_size: number of contexts per batch
  - crop_size: tokens per example (padding applied inside the sampler)
  - shuffle / shuffle_seed: controls shuffling
  - num_workers / prefetch_buffer_size: Grain read options

Outputs:
  - grain.IterDataset yielding MaxText-compatible dicts:
      inputs, inputs_segmentation, inputs_position,
      targets, targets_segmentation, targets_position
"""

from pathlib import Path
import grain.python as grain
from MaxText.input_pipeline._probe_chunk_datasource import (
    ProbeRowDataSource,
    ProbeRowSampler,
)


def build_probe_chunk_dataset(
    arrayrecord_path: str,
    batch_size: int = 32,
    crop_size: int = 1024,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    prefetch_buffer_size: int = 2,
    use_multiprocessing: bool = True,
    ram_budget_mb: int = 8192,
) -> grain.IterDataset:
    """
    Construct the probe-row Grain pipeline (DATA_LOADING_PLAN_3).

    Args:
        arrayrecord_path: Path to ArrayRecord file (single shard/file)
        batch_size: Number of contexts per batch
        crop_size: Tokens per example (padding applied as needed)
        shuffle: Whether to shuffle row order
        shuffle_seed: Seed for shuffle + sampler RNG
        num_workers: Grain read threads (0 disables threading)
        prefetch_buffer_size: Prefetch buffer per worker
        use_multiprocessing: Whether to use mp_prefetch for parallel processing
        ram_budget_mb: RAM budget for auto-tuning multiprocessing workers

    Returns:
        grain.IterDataset ready for consumption (training or analysis)
    """
    from grain.experimental import pick_performance_config

    path = Path(arrayrecord_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ArrayRecord file not found: {path}")

    source = ProbeRowDataSource(arrayrecord_path=str(path))

    dataset = grain.MapDataset.source(source)
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)

    # Repeat infinitely - with data augmentation, same rows generate different contexts
    dataset = dataset.repeat(None)

    sampler = ProbeRowSampler(
        crop_size=crop_size,
        seed=shuffle_seed,
    )
    # Apply FlatMapTransform to generate K contexts per row
    # K is calculated based on row size (larger rows = more contexts)
    # With repeat(), each row generates fresh random samples on each epoch
    dataset = dataset.apply(sampler)

    # IMPORTANT FIX: Convert to IterDataset BEFORE batching
    # This allows parallel processing of individual elements
    if num_workers > 0:
        dataset = dataset.to_iter_dataset(
            read_options=grain.ReadOptions(
                num_threads=num_workers,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
    else:
        dataset = dataset.to_iter_dataset()

    # Batch AFTER converting to IterDataset (correct ordering)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Add multiprocessing for parallel tokenization (CRITICAL for performance)
    if use_multiprocessing:
        multiprocessing_options = pick_performance_config(
            ds=dataset,
            ram_budget_mb=ram_budget_mb,
            max_workers=None,  # Auto-tune
            max_buffer_size=None,  # Auto-tune
        ).multiprocessing_options

        dataset = dataset.mp_prefetch(multiprocessing_options)

    return dataset
