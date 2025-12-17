"""
Integration layer for network measurement data with MaxText Grain pipeline.

This module provides two data loading approaches:

1. PLAN_2 (Legacy): Window-based sampling from Parquet shards
   - Use create_network_measurement_dataset()
   - 64 measurements per context, sampled at training time
   - Good for testing and development

2. DATA_LOADING_PLAN_1 (Recommended): Pre-chunked probe-centric data from ArrayRecord
   - Use create_probe_chunk_dataset()
   - Probe-centric 5-minute buckets, pre-tokenized
   - 50-100x faster I/O, perfect for federated/decentralized training
   - Requires running scripts/data/create_probe_chunks.py first
"""

import sys
import os
from pathlib import Path

# Add project root to path to import our custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import grain.python as grain
from network_grain_datasource import create_grain_pipeline
from MaxText.input_pipeline._probe_chunk_datasource import create_probe_chunk_pipeline
from MaxText import max_logging


def create_network_measurement_dataset(
    data_file_pattern: str,
    batch_size: int,
    max_tokens: int = 1024,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_epoch: int = 1,
    dataloading_host_index: int = 0,
    dataloading_host_count: int = 1,
    grain_worker_count: int = 0,
    grain_per_worker_buffer_size: int = 2,
    window_size: int = 64,
    window_stride: int = None,
    cache_size: int = None,
) -> grain.IterDataset:
    """
    Create full PLAN_2 Grain dataset for network measurements.

    This implements the complete PLAN_2 specification:
    - Window-based sampling (64 measurements per context)
    - 3 training modes (40/30/30 split): full_timestamp, no_timestamp, mixed
    - Delta timestamp encoding (2 tokens vs 9 for absolute)
    - MaxText-compatible output format
    - Memory-efficient LRU caching

    Args:
        data_file_pattern: Glob pattern for parquet files (e.g., "data/sharded/train/*.parquet")
        batch_size: Batch size for training
        max_tokens: Maximum sequence length (default: 1024)
        shuffle: Whether to shuffle data
        shuffle_seed: Random seed for shuffling
        num_epoch: Number of epochs to repeat (currently only 1 supported)
        dataloading_host_index: Index of this host (for distributed loading)
        dataloading_host_count: Total number of hosts
        grain_worker_count: Number of worker threads
        grain_per_worker_buffer_size: Buffer size per worker
        window_size: Number of measurements per context window (default: 64, per PLAN_2)
        window_stride: Step between windows (default: None = window_size for non-overlapping)
        cache_size: Number of parquet files to cache in memory (default: None = all files)

    Returns:
        Grain IterDataset ready for MaxText training
    """
    # Find parquet files
    import glob
    from pathlib import Path

    if data_file_pattern.startswith("gs://"):
        raise NotImplementedError("GCS paths not yet supported for network data")
    else:
        data_files = sorted(glob.glob(str(Path(data_file_pattern).expanduser().resolve())))

    if not data_files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {data_file_pattern}")

    max_logging.log(f"[PLAN_2] Found {len(data_files)} parquet files for network measurements")
    max_logging.log(f"[PLAN_2] Window size: {window_size} measurements per context")
    max_logging.log(f"[PLAN_2] Training modes: 40% full_timestamp, 30% no_timestamp, 30% mixed")

    # Multi-epoch and distributed loading support
    if num_epoch > 1:
        max_logging.log(f"[PLAN_2] Repeating dataset for {num_epoch} epochs")
        # Repeat the file list for multiple epochs
        data_files = data_files * num_epoch

    # Shard files across hosts (for distributed training)
    if dataloading_host_count > 1:
        max_logging.log(f"[PLAN_2] Sharding across {dataloading_host_count} hosts (index {dataloading_host_index})")
        data_files = data_files[dataloading_host_index::dataloading_host_count]
        max_logging.log(f"[PLAN_2] This host will process {len(data_files)} files")

    # Set cache_size to load all files if not specified (for maximum performance)
    # With ~200 files @ 9MB each = ~1.8GB, easily fits in memory
    if cache_size is None:
        cache_size = len(data_files)
        max_logging.log(f"[PLAN_2] Cache size set to {cache_size} (all files) for optimal performance")
    else:
        max_logging.log(f"[PLAN_2] Cache size set to {cache_size} (limited caching)")

    # Create full PLAN_2 pipeline
    dataset = create_grain_pipeline(
        parquet_files=data_files,
        batch_size=batch_size,
        window_size=window_size,
        max_tokens=max_tokens,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_workers=grain_worker_count,  # Use 0 to disable multithreading (fixed bug: was max(1, grain_worker_count))
        window_stride=window_stride,
        cache_size=cache_size,  # LRU cache size for parquet files (default: all files)
        eager_load=True,  # Preload all files at initialization for maximum performance
        prefetch_buffer_size=grain_per_worker_buffer_size,  # Buffer size per worker
    )

    max_logging.log(f"[PLAN_2] Network measurement dataset created successfully")
    max_logging.log(f"[PLAN_2] Features: window-based sampling, 40/30/30 training modes, delta timestamps")

    return dataset


def create_probe_chunk_dataset(
    data_file_pattern: str,
    batch_size: int,
    crop_size: int = 1024,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_epoch: int = 1,
    dataloading_host_index: int = 0,
    dataloading_host_count: int = 1,
    grain_worker_count: int = 0,
    grain_per_worker_buffer_size: int = 2,
    build_probe_index: bool = False,
) -> grain.IterDataset:
    """
    Create probe-centric chunk dataset (DATA_LOADING_PLAN_1).

    This is the RECOMMENDED approach for production training. It provides:
    - 50-100x faster I/O (one chunk read vs 64 measurements)
    - Pre-tokenized data (removes tokenization from training hot path)
    - Probe-centric design (perfect for federated/decentralized deployment)
    - ArrayRecord format (optimized for random access)

    Prerequisites:
        Run scripts/data/create_probe_chunks.py to create chunked data first.

    Args:
        data_file_pattern: Path to ArrayRecord file (e.g., "data/probe_chunks/train.arrayrecord")
        batch_size: Batch size for training
        crop_size: Tokens per training example (default: 1024)
        shuffle: Whether to shuffle chunks
        shuffle_seed: Random seed for shuffling
        num_epoch: Number of epochs to repeat (default: 1)
        dataloading_host_index: Index of this host (for distributed loading)
        dataloading_host_count: Total number of hosts
        grain_worker_count: Number of worker threads
        grain_per_worker_buffer_size: Buffer size per worker
        build_probe_index: Build probe index for client-first sampling (slower initialization)

    Returns:
        Grain IterDataset ready for MaxText training
    """
    # Resolve path
    arrayrecord_path = Path(data_file_pattern).expanduser().resolve()

    if not arrayrecord_path.exists():
        raise FileNotFoundError(
            f"ArrayRecord file not found: {arrayrecord_path}\n"
            f"Run: python scripts/data/create_probe_chunks.py"
        )

    max_logging.log(f"[DATA_LOADING_PLAN_1] Loading probe-centric chunks from {arrayrecord_path}")
    max_logging.log(f"[DATA_LOADING_PLAN_1] Crop size: {crop_size} tokens per training example")
    max_logging.log(f"[DATA_LOADING_PLAN_1] Training modes: 40% full_timestamp, 30% no_timestamp, 30% mixed")

    # Multi-epoch support (repeat dataset)
    if num_epoch > 1:
        max_logging.log(f"[DATA_LOADING_PLAN_1] WARNING: Multi-epoch repeat not yet implemented for ArrayRecord")
        max_logging.log(f"[DATA_LOADING_PLAN_1] Will use single epoch. Implement repeat in training loop instead.")

    # Distributed loading support
    if dataloading_host_count > 1:
        max_logging.log(f"[DATA_LOADING_PLAN_1] WARNING: Distributed loading across {dataloading_host_count} hosts")
        max_logging.log(f"[DATA_LOADING_PLAN_1] ArrayRecord sharding not yet implemented - all hosts will read same data")
        max_logging.log(f"[DATA_LOADING_PLAN_1] For distributed training, pre-shard ArrayRecord files by host")

    # Create probe chunk pipeline
    dataset = create_probe_chunk_pipeline(
        arrayrecord_path=str(arrayrecord_path),
        batch_size=batch_size,
        crop_size=crop_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_workers=grain_worker_count,
        prefetch_buffer_size=grain_per_worker_buffer_size,
        build_probe_index=build_probe_index,
    )

    max_logging.log(f"[DATA_LOADING_PLAN_1] Probe chunk dataset created successfully")
    max_logging.log(f"[DATA_LOADING_PLAN_1] Benefits: 50-100x faster I/O, pre-tokenized, probe-centric design")

    return dataset
