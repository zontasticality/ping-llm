"""
Integration layer for network measurement data with MaxText Grain pipeline.

This module provides DATA_LOADING_PLAN_3:

Probe-centric big-row data loading from ArrayRecord:
   - Use create_probe_chunk_dataset()
   - Minimal padding (<5% vs 50-90% in previous plans)
   - Multi-scale temporal learning (log-uniform window sampling)
   - Runtime tokenization with data augmentation (3 timestamp modes)
   - Probe-centric design (perfect for federated/decentralized training)
   - Requires running scripts/data/create_probe_rows_parallel_streaming.py first
"""

import sys
import os
from pathlib import Path

# Add project root to path to import our custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import grain.python as grain
from MaxText.input_pipeline.probe_chunk_pipeline import build_probe_chunk_dataset
from MaxText import max_logging


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
) -> grain.IterDataset:
    """
    Create probe-centric row dataset (PLAN_3).

    This is the RECOMMENDED approach for production training. It provides:
    - Minimal padding (<5% vs 50-90%)
    - Multi-scale temporal learning (log-uniform window sampling)
    - Runtime tokenization with data augmentation
    - Probe-centric design (perfect for federated/decentralized deployment)
    - ArrayRecord format (optimized for random access)

    Prerequisites:
        Run scripts/data/modal_create_probe_rows_parallel_streaming.py to create probe row data first.

    Args:
        data_file_pattern: Path to ArrayRecord file (e.g., "data/probe_rows/train.arrayrecord")
        batch_size: Batch size for training
        crop_size: Tokens per training example (default: 1024)
        shuffle: Whether to shuffle rows
        shuffle_seed: Random seed for shuffling
        num_epoch: Number of epochs to repeat (default: 1)
        dataloading_host_index: Index of this host (for distributed loading)
        dataloading_host_count: Total number of hosts
        grain_worker_count: Number of worker threads
        grain_per_worker_buffer_size: Buffer size per worker

    Returns:
        Grain IterDataset ready for MaxText training
    """
    # Resolve path
    arrayrecord_path = Path(data_file_pattern).expanduser().resolve()

    if not arrayrecord_path.exists():
        raise FileNotFoundError(
            f"ArrayRecord file not found: {arrayrecord_path}\n"
            f"Run: python scripts/data/create_probe_rows.py"
        )

    max_logging.log(f"[PLAN_3] Loading probe-centric rows from {arrayrecord_path}")
    max_logging.log(f"[PLAN_3] Crop size: {crop_size} tokens per training example")
    max_logging.log(f"[PLAN_3] Training modes: 40% full_timestamp, 30% partial_timestamp, 30% no_timestamp")

    # Multi-epoch support (repeat dataset)
    if num_epoch > 1:
        max_logging.log(f"[PLAN_3] WARNING: Multi-epoch repeat not yet implemented for ArrayRecord")
        max_logging.log(f"[PLAN_3] Will use single epoch. Implement repeat in training loop instead.")

    # Distributed loading support
    if dataloading_host_count > 1:
        max_logging.log(f"[PLAN_3] WARNING: Distributed loading across {dataloading_host_count} hosts")
        max_logging.log(f"[PLAN_3] ArrayRecord sharding not yet implemented - all hosts will read same data")
        max_logging.log(f"[PLAN_3] For distributed training, pre-shard ArrayRecord files by host")

    # Create probe row pipeline
    dataset = build_probe_chunk_dataset(
        arrayrecord_path=str(arrayrecord_path),
        batch_size=batch_size,
        crop_size=crop_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        num_workers=grain_worker_count,
        prefetch_buffer_size=grain_per_worker_buffer_size,
    )

    max_logging.log(f"[PLAN_3] Probe row dataset created successfully")
    max_logging.log(f"[PLAN_3] Benefits: minimal padding, multi-scale temporal learning, runtime tokenization")

    return dataset
