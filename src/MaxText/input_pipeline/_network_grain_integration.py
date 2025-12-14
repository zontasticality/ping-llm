"""
Integration layer for network measurement data with MaxText Grain pipeline.

This module bridges the custom network_grain_datasource_simple.py with MaxText's
training loop.
"""

import sys
import os

# Add project root to path to import our custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from network_grain_datasource_simple import ParquetMeasurementDataSource, SimpleMeasurementTokenizer
import grain.python as grain
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
) -> grain.IterDataset:
    """
    Create Grain dataset for network measurements using custom data source.

    This function integrates the custom ParquetMeasurementDataSource with MaxText's
    training pipeline.

    Args:
        data_file_pattern: Glob pattern for parquet files (e.g., "data/sharded/train/*.parquet")
        batch_size: Batch size for training
        max_tokens: Maximum sequence length
        shuffle: Whether to shuffle data
        shuffle_seed: Random seed for shuffling
        num_epoch: Number of epochs to repeat
        dataloading_host_index: Index of this host (for distributed loading)
        dataloading_host_count: Total number of hosts
        grain_worker_count: Number of worker threads
        grain_per_worker_buffer_size: Buffer size per worker

    Returns:
        Grain IterDataset ready for MaxText training
    """
    # Find parquet files
    import glob
    from pathlib import Path

    if data_file_pattern.startswith("gs://"):
        raise NotImplementedError("GCS paths not yet supported for network data")
    else:
        data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))

    if not data_files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {data_file_pattern}")

    max_logging.log(f"Found {len(data_files)} parquet files for network measurements")

    # Create data source
    source = ParquetMeasurementDataSource(data_files)
    max_logging.log(f"Created ParquetMeasurementDataSource with {len(source)} total rows")

    # Wrap in MapDataset
    dataset = grain.MapDataset.source(source)

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)

    # Repeat for epochs
    if num_epoch > 1:
        dataset = dataset.repeat(num_epoch)

    # Shard across hosts (for distributed training)
    if dataloading_host_count > 1:
        dataset = dataset[dataloading_host_index::dataloading_host_count]

    # Tokenize measurements
    tokenizer = SimpleMeasurementTokenizer(
        include_timestamp=True,  # TODO: Make this configurable for training modes
        max_tokens=max_tokens,
        seed=shuffle_seed,
    )
    dataset = dataset.map(tokenizer)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Convert to IterDataset with multiprocessing
    read_options = grain.ReadOptions(
        num_threads=max(1, grain_worker_count),
        prefetch_buffer_size=max(1, grain_per_worker_buffer_size),
    )

    dataset = dataset.to_iter_dataset(read_options=read_options)

    max_logging.log(f"Network measurement dataset created successfully")

    return dataset
