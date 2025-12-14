"""
Grain DataSource for network measurement data (PLAN_2).

This module implements:
1. Custom Parquet DataSource for network measurements
2. ContextWindowSampler with 3 training modes (40/30/30 split):
   - Mode 1 (40%): Full timestamp, temporal order
   - Mode 2 (30%): No timestamp, random shuffle
   - Mode 3 (30%): Mixed timestamp with interleaving
3. Tokenization using PLAN_2 schema
"""

import random
from typing import List, Dict, Any, Optional, Iterator
import grain.python as grain
import pyarrow.parquet as pq
import pyarrow.compute as pc
from datetime import datetime

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


class ContextWindowSampler(grain.MapTransform):
    """
    Sample context windows with PLAN_2 training modes.

    This implements the 3 training modes:
    - Mode 1 (40%): Full timestamp, temporal order
    - Mode 2 (30%): No timestamp, random shuffle
    - Mode 3 (30%): Mixed timestamp with interleaving
    """

    def __init__(
        self,
        window_size: int = 64,
        max_tokens: int = 1024,
        mode_weights: tuple = (0.40, 0.30, 0.30),
        seed: Optional[int] = None,
    ):
        """
        Args:
            window_size: Target number of measurements per context window
            max_tokens: Maximum tokens per sequence (hard limit)
            mode_weights: Tuple of (full_ts, no_ts, mixed_ts) probabilities
            seed: Random seed for reproducibility
        """
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.mode_weights = mode_weights
        self.rng = random.Random(seed)

    def map(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single measurement and accumulate into windows.

        Args:
            measurement: Single measurement dictionary (from DataSource)

        Returns:
            Dictionary with 'tokens', 'length', 'mode'

        Note: This is called per-measurement by Grain. We'll treat each
        measurement as a minimal window for now. For proper windowing,
        use grain.experimental.WindowDataset or custom batching.
        """
        # For now, encode single measurement as a minimal window
        # TODO: Implement proper sliding window with grain.WindowDataset
        window = [measurement]

        # Decide training mode (40/30/30 split)
        mode_choice = self.rng.choices(
            ['full_timestamp', 'no_timestamp', 'mixed'],
            weights=self.mode_weights
        )[0]

        if mode_choice == 'full_timestamp':
            # Mode 1: All have timestamps, keep temporal order
            window_sorted = sorted(window, key=lambda m: m['event_time'])
            has_timestamp = [True] * len(window_sorted)
            window_final = window_sorted

        elif mode_choice == 'no_timestamp':
            # Mode 2: None have timestamps, random order
            window_shuffled = window.copy()
            self.rng.shuffle(window_shuffled)
            has_timestamp = [False] * len(window_shuffled)
            window_final = window_shuffled

        else:  # mode_choice == 'mixed'
            # Mode 3: Randomly assign timestamps (40-60% coverage)
            has_timestamp = [self.rng.random() < 0.5 for _ in window]

            # Separate timestamped and non-timestamped
            timestamped = [m for m, has_ts in zip(window, has_timestamp) if has_ts]
            non_timestamped = [m for m, has_ts in zip(window, has_timestamp) if not has_ts]

            # Keep timestamped ordered, shuffle non-timestamped
            timestamped_sorted = sorted(timestamped, key=lambda m: m['event_time'])
            self.rng.shuffle(non_timestamped)

            # Interleave randomly (preserving timestamped order)
            window_final = self._interleave_preserving_order(timestamped_sorted, non_timestamped)

            # Update has_timestamp to match new order
            has_timestamp = [m in timestamped_sorted for m in window_final]

        # Encode window with optional timestamps
        tokens = self._encode_window(window_final, has_timestamp)

        # Truncate to max_tokens if needed
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]

        return {
            "tokens": tokens,
            "length": len(tokens),
            "mode": mode_choice,
        }

    def _interleave_preserving_order(
        self,
        ordered_list: List[Any],
        random_list: List[Any]
    ) -> List[Any]:
        """Randomly interleave two lists, preserving order of first list."""
        result = []
        i, j = 0, 0

        while i < len(ordered_list) or j < len(random_list):
            # Randomly decide whether to take from ordered or random
            if i >= len(ordered_list):
                result.append(random_list[j])
                j += 1
            elif j >= len(random_list):
                result.append(ordered_list[i])
                i += 1
            elif self.rng.random() < 0.5:
                result.append(ordered_list[i])
                i += 1
            else:
                result.append(random_list[j])
                j += 1

        return result

    def _encode_window(
        self,
        window: List[Dict[str, Any]],
        has_timestamp: List[bool]
    ) -> List[int]:
        """
        Encode measurements with delta timestamps.

        Delta timestamps skip over non-timestamped measurements:
        - M1(t=100) → M2(no ts) → M3(t=220)
        - M3's delta is 220-100=120s, not relative to M2
        """
        tokens = []
        prev_timestamped = None

        for meas, include_ts in zip(window, has_timestamp):
            # Encode measurement with optional timestamp
            meas_tokens = encode_measurement(
                meas,
                prev_timestamp=prev_timestamped if include_ts else None,
                include_timestamp=include_ts
            )
            tokens.extend(meas_tokens)

            # Update prev_timestamped only if this measurement had a timestamp
            if include_ts:
                prev_timestamped = meas['event_time']

        return tokens


def create_grain_pipeline(
    parquet_files: List[str],
    batch_size: int = 32,
    window_size: int = 64,
    max_tokens: int = 1024,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_workers: int = 4,
) -> grain.IterDataset:
    """
    Create Grain data pipeline for network measurements.

    Args:
        parquet_files: List of Parquet shard file paths
        batch_size: Batch size for training
        window_size: Number of measurements per context window
        max_tokens: Maximum tokens per sequence
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

    # Sample context windows with training modes
    sampler = ContextWindowSampler(
        window_size=window_size,
        max_tokens=max_tokens,
        seed=shuffle_seed,
    )
    dataset = dataset.map(sampler)

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
