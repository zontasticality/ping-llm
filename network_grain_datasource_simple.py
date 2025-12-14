"""
Simplified Grain DataSource for network measurement data (PLAN_2).

This version encodes measurements individually for compatibility with Grain's
MapTransform. Window-based sampling will be implemented later.
"""

import random
from typing import List, Dict, Any, Optional
import grain.python as grain
import pyarrow.parquet as pq

# Import our tokenization
from tokenization import encode_measurement


class ParquetMeasurementDataSource(grain.RandomAccessDataSource):
    """
    Grain DataSource for network measurement Parquet files.

    This reads Parquet files and yields individual measurement rows.
    """

    def __init__(self, parquet_files: List[str]):
        """
        Args:
            parquet_files: List of paths to Parquet shard files
        """
        self.parquet_files = parquet_files
        self._tables = []
        self._row_offsets = []  # Cumulative row counts for each file
        self._total_rows = 0

        # Load all Parquet files and compute offsets
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            self._tables.append(table)
            self._total_rows += len(table)
            self._row_offsets.append(self._total_rows)

    def __len__(self):
        return self._total_rows

    def __getitem__(self, index):
        """Get a single measurement row by global index."""
        if index < 0 or index >= self._total_rows:
            raise IndexError(f"Index {index} out of range [0, {self._total_rows})")

        # Binary search to find which file contains this index
        file_idx = 0
        for i, offset in enumerate(self._row_offsets):
            if index < offset:
                file_idx = i
                break

        # Compute local index within the file
        local_idx = index if file_idx == 0 else (index - self._row_offsets[file_idx - 1])

        # Extract row from PyArrow table
        table = self._tables[file_idx]
        row = {
            'src_addr': table['src_addr'][local_idx].as_py(),
            'dst_addr': table['dst_addr'][local_idx].as_py(),
            'ip_version': table['ip_version'][local_idx].as_py(),
            'rtt': table['rtt'][local_idx].as_py(),
            'event_time': table['event_time'][local_idx].as_py(),
        }

        return row


class SimpleMeasurementTokenizer(grain.MapTransform):
    """
    Simple tokenizer that encodes measurements individually.

    For Phase 2 testing. Window-based sampling will be added in Phase 3.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        timestamp_mode: str = 'full',
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ):
        """
        Args:
            include_timestamp: Whether to include timestamps
            timestamp_mode: 'full' (absolute) or 'delta' (not implemented yet)
            max_tokens: Maximum tokens per sequence
            seed: Random seed
        """
        self.include_timestamp = include_timestamp
        self.timestamp_mode = timestamp_mode
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)
        self.prev_timestamp = None

    def map(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a single measurement.

        Args:
            measurement: Measurement dictionary

        Returns:
            Dictionary with 'tokens', 'length'
        """
        # Decide whether to include timestamp (for training modes)
        include_ts = self.include_timestamp

        # Encode measurement
        tokens = encode_measurement(
            measurement,
            prev_timestamp=None,  # For now, always use absolute
            include_timestamp=include_ts
        )

        # Store original length before padding
        original_length = len(tokens)

        # Truncate if needed
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            original_length = self.max_tokens

        # Pad to max_tokens for batching
        # Use token ID 0 (MEASUREMENT_START) as padding (won't be used in loss)
        while len(tokens) < self.max_tokens:
            tokens.append(0)

        return {
            "tokens": tokens,
            "length": original_length,
        }


def create_simple_grain_pipeline(
    parquet_files: List[str],
    batch_size: int = 32,
    max_tokens: int = 1024,
    include_timestamp: bool = True,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_workers: int = 4,
) -> grain.IterDataset:
    """
    Create simple Grain data pipeline for network measurements.

    This version tokenizes measurements individually (not window-based).
    Suitable for initial testing and integration.

    Args:
        parquet_files: List of Parquet shard file paths
        batch_size: Batch size for training
        max_tokens: Maximum tokens per sequence
        include_timestamp: Whether to include timestamps
        shuffle: Whether to shuffle the dataset
        shuffle_seed: Random seed for shuffling
        num_workers: Number of worker threads for data loading

    Returns:
        Grain IterDataset ready for training
    """
    # Create data source
    source = ParquetMeasurementDataSource(parquet_files)

    # Wrap in MapDataset
    dataset = grain.MapDataset.source(source)

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)

    # Tokenize measurements
    tokenizer = SimpleMeasurementTokenizer(
        include_timestamp=include_timestamp,
        max_tokens=max_tokens,
        seed=shuffle_seed,
    )
    dataset = dataset.map(tokenizer)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Convert to IterDataset
    dataset = dataset.to_iter_dataset(
        read_options=grain.ReadOptions(
            num_threads=num_workers,
            prefetch_buffer_size=2,
        )
    )

    return dataset
