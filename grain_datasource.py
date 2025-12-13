"""
Grain DataSource for network measurement tokenization.

This module provides a custom Grain DataSource that reads Parquet files
and applies tokenization on-the-fly for MaxText training.

Usage with MaxText:
    # In your MaxText data loading config
    from grain_datasource import NetworkMeasurementDataSource

    datasource = NetworkMeasurementDataSource(
        file_pattern="data/sharded/train/*.parquet",
        shuffle=True,
        seed=42,
    )
"""

import glob
from typing import Dict, Any, List, Iterator
import pyarrow.parquet as pq
import numpy as np

try:
    import grain.python as grain
    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False
    print("Warning: grain.python not available. Install with: pip install grain-python")

from tokenization import encode_measurement, VOCAB_SIZE


class NetworkMeasurementDataSource:
    """
    Custom Grain-compatible DataSource for network measurements.

    This DataSource:
    1. Reads Parquet files matching a glob pattern
    2. Applies tokenization on-the-fly using encode_measurement()
    3. Returns dictionaries with 'tokens' and 'length' keys

    Example:
        datasource = NetworkMeasurementDataSource(
            file_pattern="data/sharded/train/*.parquet"
        )

        for example in datasource:
            tokens = example['tokens']
            length = example['length']
            # tokens is a numpy array of shape (length,) with dtype int32
    """

    def __init__(
        self,
        file_pattern: str,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the data source.

        Args:
            file_pattern: Glob pattern for Parquet files (e.g., "data/sharded/train/*.parquet")
            shuffle: Whether to shuffle files (default: False)
            seed: Random seed for shuffling (default: 42)
        """
        self.file_pattern = file_pattern
        self.shuffle = shuffle
        self.seed = seed

        # Find all matching files
        self.files = sorted(glob.glob(file_pattern))
        if not self.files:
            raise ValueError(f"No files found matching pattern: {file_pattern}")

        if self.shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(self.files)

        print(f"NetworkMeasurementDataSource initialized:")
        print(f"  Pattern: {file_pattern}")
        print(f"  Files found: {len(self.files)}")
        print(f"  Shuffle: {shuffle}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all measurements in all files."""
        for file_path in self.files:
            # Read entire file (for small shards this is efficient)
            table = pq.read_table(file_path)
            df = table.to_pandas()

            # Iterate over rows
            for idx, row in df.iterrows():
                try:
                    row_dict = row.to_dict()
                    tokens = encode_measurement(row_dict)

                    yield {
                        'tokens': np.array(tokens, dtype=np.int32),
                        'length': len(tokens),
                    }
                except Exception as e:
                    # Log and skip problematic rows
                    print(f"Warning: Failed to tokenize row {idx} in {file_path}: {e}")
                    continue

    def __len__(self) -> int:
        """
        Return approximate number of examples.

        Note: This reads all files to count rows, which may be slow.
        For training, Grain typically doesn't require an exact length.
        """
        total_rows = 0
        for file_path in self.files:
            parquet_file = pq.ParquetFile(file_path)
            total_rows += parquet_file.metadata.num_rows
        return total_rows


def create_grain_dataset(
    file_pattern: str,
    batch_size: int = 32,
    max_length: int = 1024,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Create a Grain dataset with batching and packing.

    This is a helper function that creates a complete Grain pipeline:
    1. NetworkMeasurementDataSource
    2. Shuffling (optional)
    3. Packing multiple measurements into sequences
    4. Batching

    Args:
        file_pattern: Glob pattern for Parquet files
        batch_size: Number of sequences per batch
        max_length: Maximum sequence length (in tokens)
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        Grain dataset iterator

    Note: This requires grain.python to be installed.
    """
    if not GRAIN_AVAILABLE:
        raise ImportError("grain.python is required. Install with: pip install grain-python")

    # Create data source
    datasource = NetworkMeasurementDataSource(
        file_pattern=file_pattern,
        shuffle=shuffle,
        seed=seed,
    )

    # TODO: Add packing transform to combine multiple measurements into max_length sequences
    # TODO: Add batching transform
    # TODO: Add prefetching for performance

    # For now, return a simple iterator
    # Full Grain integration requires implementing:
    # - grain.MapTransform for packing
    # - grain.Batch for batching
    # - grain.experimental.DataLoader for multi-worker loading

    print("Note: Full Grain pipeline not yet implemented.")
    print("Using simple iterator. See grain_datasource.py for TODOs.")

    return iter(datasource)


# Example usage
if __name__ == "__main__":
    import sys

    # Test the datasource
    print("=" * 80)
    print("Grain DataSource Test")
    print("=" * 80)

    # Use single file for testing
    test_pattern = "data/training_data.parquet"

    print(f"\nCreating datasource with pattern: {test_pattern}")
    datasource = NetworkMeasurementDataSource(
        file_pattern=test_pattern,
        shuffle=False,
    )

    print(f"\nFetching first 5 examples...")
    for i, example in enumerate(datasource):
        if i >= 5:
            break

        tokens = example['tokens']
        length = example['length']

        print(f"\nExample {i + 1}:")
        print(f"  Length: {length}")
        print(f"  Token dtype: {tokens.dtype}")
        print(f"  Token shape: {tokens.shape}")
        print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
        print(f"  First 20 tokens: {tokens[:20].tolist()}")

    print("\n" + "=" * 80)
    print("✅ Grain DataSource test completed")
    print("=" * 80)
