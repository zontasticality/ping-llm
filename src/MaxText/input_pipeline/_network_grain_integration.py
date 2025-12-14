"""
Integration layer for network measurement data with MaxText Grain pipeline.

This module bridges the full PLAN_2 network_grain_datasource.py with MaxText's
training loop.

Implements:
- Window-based sampling (64 measurements per context)
- 3 training modes (40/30/30 split)
- Delta timestamp encoding
- MaxText-compatible output format
"""

import sys
import os

# Add project root to path to import our custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import grain.python as grain
from network_grain_datasource import create_grain_pipeline
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

    # Create full PLAN_2 pipeline
    dataset = create_grain_pipeline(
        parquet_files=data_files,
        batch_size=batch_size,
        window_size=window_size,
        max_tokens=max_tokens,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_workers=max(1, grain_worker_count),
        window_stride=window_stride,
        cache_size=4,  # LRU cache size for parquet files
    )

    max_logging.log(f"[PLAN_2] Network measurement dataset created successfully")
    max_logging.log(f"[PLAN_2] Features: window-based sampling, 40/30/30 training modes, delta timestamps")

    return dataset
