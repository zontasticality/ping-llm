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
  - grain.IterDataset yielding dicts:
      inputs, inputs_segmentation, inputs_position,
      targets, targets_segmentation, targets_position
"""

import glob as globmod
from pathlib import Path
import grain.python as grain
from ping_llm.data.datasource import (
    DeserializeProbeRow,
    ProbeRowDataSource,
    ProbeRowSampler,
)


def build_probe_chunk_dataset(
    arrayrecord_path: str | list[str],
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
    # Resolve paths: support single path, glob pattern, or list of paths
    paths: list[str] | None = None
    if isinstance(arrayrecord_path, list):
        paths = arrayrecord_path
    elif '*' in arrayrecord_path:
        paths = sorted(globmod.glob(arrayrecord_path))
        if not paths:
            raise FileNotFoundError(f"No files matched glob: {arrayrecord_path}")

    if paths is not None:
        # Sharded mode: use grain's built-in ArrayRecordDataSource + DeserializeProbeRow
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"ArrayRecord shard not found: {p}")
        source = grain.ArrayRecordDataSource(paths)
        dataset = grain.MapDataset.source(source)
        dataset = dataset.map(DeserializeProbeRow())
    else:
        # Single-file mode: use ProbeRowDataSource (handles deserialization internally)
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

    # Convert to IterDataset BEFORE batching.
    # When using mp_prefetch, each worker process runs the full pipeline
    # independently. Threads inside each worker are redundant for CPU-bound
    # tokenization (GIL-limited), so use minimal threads with mp_prefetch.
    if use_multiprocessing:
        # mp_prefetch provides process-level parallelism — keep threads minimal
        dataset = dataset.to_iter_dataset(
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=2)
        )
    elif num_workers > 0:
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

    # Multiprocessing for parallel data loading + tokenization.
    # Note: pick_performance_config auto-tuning deadlocks in spawn mode on Modal,
    # so we use fixed MultiprocessingOptions.
    if use_multiprocessing:
        # Cap workers: more workers than CPUs causes contention, not speedup.
        # Leave 2 CPUs for main process + GPU work.
        mp_workers = min(num_workers if num_workers > 0 else 4, 6)
        multiprocessing_options = grain.MultiprocessingOptions(
            num_workers=mp_workers,
            per_worker_buffer_size=min(prefetch_buffer_size, 4),
        )
        dataset = dataset.mp_prefetch(multiprocessing_options)

    return dataset
