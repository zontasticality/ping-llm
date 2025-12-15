"""
Grain DataSource for network measurement data (PLAN_2 - Full Implementation).

This module implements:
1. Memory-efficient Parquet DataSource with LRU caching
2. WindowedMeasurementDataSource - yields windows of consecutive measurements
3. ContextWindowSampler with 3 training modes (40/30/30 split):
   - Mode 1 (40%): Full timestamp, temporal order
   - Mode 2 (30%): No timestamp, random shuffle
   - Mode 3 (30%): Mixed timestamp with interleaving
4. Tokenization using PLAN_2 schema with delta timestamps
5. MaxText-compatible output format
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional
from functools import lru_cache
import threading
import grain.python as grain
import pyarrow.parquet as pq

# Import our tokenization
from tokenization import encode_measurement


class ParquetMeasurementDataSource(grain.RandomAccessDataSource):
    """
    Memory-efficient Grain DataSource for network measurement Parquet files.

    Uses LRU caching to avoid loading all files into memory at once.
    This is critical for datasets with 180+ shards (180M rows total).
    """

    def __init__(self, parquet_files: List[str], cache_size: int = 4):
        """
        Args:
            parquet_files: List of paths to Parquet shard files
            cache_size: Number of parquet files to keep in memory (default: 4)
        """
        self.parquet_files = sorted(parquet_files)  # Sort for deterministic ordering
        self._file_row_counts = []
        self._row_offsets = []  # Cumulative row counts for each file
        self._total_rows = 0
        self._cache = {}  # Simple cache: {file_idx: table}
        self._cache_order = []  # Track access order for LRU eviction
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()  # Thread-safe cache access for grain workers

        # Get row counts WITHOUT loading full tables (memory efficient)
        for parquet_file in self.parquet_files:
            # Just read metadata, not the full table
            metadata = pq.read_metadata(parquet_file)
            row_count = metadata.num_rows
            self._file_row_counts.append(row_count)
            self._total_rows += row_count
            self._row_offsets.append(self._total_rows)

    def __len__(self):
        return self._total_rows

    def _load_table(self, file_idx: int):
        """Load table with thread-safe LRU caching."""
        with self._cache_lock:
            if file_idx in self._cache:
                # Cache hit - move to end (most recently used)
                self._cache_order.remove(file_idx)
                self._cache_order.append(file_idx)
                return self._cache[file_idx]

            # Cache miss - load table (release lock during I/O)
            # This allows other workers to access cache while we read from disk

        # Load table outside the lock (I/O is slow, don't block other workers)
        table = pq.read_table(self.parquet_files[file_idx])

        with self._cache_lock:
            # Re-check cache in case another worker loaded it while we were reading
            if file_idx in self._cache:
                return self._cache[file_idx]

            # Add to cache
            self._cache[file_idx] = table
            self._cache_order.append(file_idx)

            # Evict oldest if cache is full
            if len(self._cache) > self._cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]

            return table

    def __getitem__(self, index):
        """Get a single measurement row by global index (loads file on-demand with caching)."""
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

        # Load table (cached)
        table = self._load_table(file_idx)

        # Extract row from PyArrow table
        row = {
            'src_addr': table['src_addr'][local_idx].as_py(),
            'dst_addr': table['dst_addr'][local_idx].as_py(),
            'ip_version': table['ip_version'][local_idx].as_py(),
            'rtt': table['rtt'][local_idx].as_py(),
            'event_time': table['event_time'][local_idx].as_py(),
        }

        return row


class WindowedMeasurementDataSource(grain.RandomAccessDataSource):
    """
    Grain DataSource that yields windows of consecutive measurements.

    This wraps ParquetMeasurementDataSource and samples windows of N consecutive
    measurements, enabling proper in-context learning as specified in PLAN_2.

    Key design decisions:
    - Windows are consecutive measurements (temporal locality)
    - Overlapping windows with stride (data augmentation)
    - Each window becomes one training example
    """

    def __init__(
        self,
        parquet_files: List[str],
        window_size: int = 64,
        stride: Optional[int] = None,
        cache_size: int = 4,
    ):
        """
        Args:
            parquet_files: List of Parquet shard file paths
            window_size: Number of measurements per context window (default: 64)
            stride: Step size between windows. If None, defaults to window_size (non-overlapping)
            cache_size: Number of parquet files to cache in memory
        """
        self.base_source = ParquetMeasurementDataSource(parquet_files, cache_size=cache_size)
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size

        # Calculate number of windows
        total_measurements = len(self.base_source)
        self._num_windows = max(1, (total_measurements - window_size) // self.stride + 1)

    def __len__(self):
        return self._num_windows

    def __getitem__(self, window_index):
        """
        Get a window of measurements by window index.

        Args:
            window_index: Index of the window to retrieve

        Returns:
            List of measurement dictionaries (length = window_size)
        """
        if window_index < 0 or window_index >= self._num_windows:
            raise IndexError(f"Window index {window_index} out of range [0, {self._num_windows})")

        # Calculate start index for this window
        start_idx = window_index * self.stride

        # Sample measurements for this window
        window = []
        for i in range(self.window_size):
            meas_idx = start_idx + i
            # Handle edge case: last window might be shorter
            if meas_idx >= len(self.base_source):
                # Wrap around (circular) to ensure full window
                meas_idx = meas_idx % len(self.base_source)
            window.append(self.base_source[meas_idx])

        return window


class ContextWindowTokenizer(grain.MapTransform):
    """
    Tokenize context windows with PLAN_2 training modes.

    This implements the 3 training modes:
    - Mode 1 (40%): Full timestamp, temporal order
    - Mode 2 (30%): No timestamp, random shuffle
    - Mode 3 (30%): Mixed timestamp with interleaving

    Produces MaxText-compatible output format.
    """

    def __init__(
        self,
        max_tokens: int = 1024,
        mode_weights: tuple = (0.40, 0.30, 0.30),
        seed: Optional[int] = None,
    ):
        """
        Args:
            max_tokens: Maximum tokens per sequence (hard limit)
            mode_weights: Tuple of (full_ts, no_ts, mixed_ts) probabilities
            seed: Random seed for reproducibility
        """
        self.max_tokens = max_tokens
        self.mode_weights = mode_weights
        self.rng = random.Random(seed)

    def map(self, window: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tokenize a window of measurements with training mode selection.

        Args:
            window: List of measurement dictionaries (from WindowedDataSource)

        Returns:
            Dictionary with MaxText-compatible fields:
                - inputs: token array (int32)
                - inputs_segmentation: mask for real tokens (int32)
                - inputs_position: position indices (int32)
                - targets: same as inputs (autoregressive)
                - targets_segmentation: same as inputs_segmentation
                - targets_position: same as inputs_position
        """
        # Decide training mode (40/30/30 split)
        mode_choice = self.rng.choices(
            ['full_timestamp', 'no_timestamp', 'mixed'],
            weights=self.mode_weights
        )[0]

        # Apply training mode
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

        # Store original length before padding/truncation
        original_length = len(tokens)

        # Truncate to max_tokens if needed
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            original_length = self.max_tokens

        # Pad to max_tokens for batching
        # Use token ID 0 (MEASUREMENT_START) as padding
        while len(tokens) < self.max_tokens:
            tokens.append(0)

        # Convert to numpy arrays (MaxText expects int32)
        tokens_array = np.array(tokens, dtype=np.int32)

        # Create segmentation (1 for real tokens, 0 for padding)
        segmentation = np.ones(self.max_tokens, dtype=np.int32)
        segmentation[original_length:] = 0

        # Create position IDs (0 to seq_len-1)
        positions = np.arange(self.max_tokens, dtype=np.int32)

        # For autoregressive training: inputs and targets are the same
        # (MaxText handles the shifting internally)
        return {
            "inputs": tokens_array,
            "inputs_segmentation": segmentation,
            "inputs_position": positions,
            "targets": tokens_array,
            "targets_segmentation": segmentation,
            "targets_position": positions,
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

        This implements the PLAN_2 delta timestamp specification.
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
    window_stride: Optional[int] = None,
    cache_size: int = 4,
) -> grain.IterDataset:
    """
    Create full PLAN_2 Grain data pipeline for network measurements.

    This implements the complete PLAN_2 specification:
    - Window-based sampling (64 measurements per context)
    - 3 training modes (40/30/30 split)
    - Delta timestamp encoding
    - MaxText-compatible output

    Args:
        parquet_files: List of Parquet shard file paths
        batch_size: Batch size for training
        window_size: Number of measurements per context window (default: 64)
        max_tokens: Maximum tokens per sequence (default: 1024)
        shuffle: Whether to shuffle the dataset
        shuffle_seed: Random seed for shuffling
        num_workers: Number of worker threads for data loading
        window_stride: Step between windows (default: window_size for non-overlapping)
        cache_size: Number of parquet files to cache in memory

    Returns:
        Grain IterDataset ready for MaxText training
    """
    # Create windowed data source
    source = WindowedMeasurementDataSource(
        parquet_files,
        window_size=window_size,
        stride=window_stride,
        cache_size=cache_size,
    )

    # Wrap in MapDataset
    dataset = grain.MapDataset.source(source)

    # Shuffle if requested (shuffles windows, not individual measurements)
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)

    # Tokenize windows with training mode selection
    tokenizer = ContextWindowTokenizer(
        max_tokens=max_tokens,
        seed=shuffle_seed,
    )
    dataset = dataset.map(tokenizer)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Convert to IterDataset with multiprocessing
    dataset = dataset.to_iter_dataset(
        read_options=grain.ReadOptions(
            num_threads=num_workers,
            prefetch_buffer_size=2,
        )
    )

    return dataset
