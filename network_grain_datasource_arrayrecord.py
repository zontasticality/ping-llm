"""
Grain DataSource for network measurement data using ArrayRecord.

This is a simple replacement for network_grain_datasource.py that reads from
ArrayRecord files instead of Parquet. ArrayRecord provides:
- Fast random access (100-1000× faster than Parquet)
- Efficient shuffling (no decompression overhead)
- Streaming support (constant memory usage)

Usage:
    from network_grain_datasource_arrayrecord import create_grain_pipeline

    dataset = create_grain_pipeline(
        arrayrecord_files=["data/arrayrecord/train.arrayrecord"],
        batch_size=32,
        window_size=64,
        max_tokens=1024,
        shuffle=True,
    )
"""

import numpy as np
from typing import List, Dict, Any, Optional
import random

import grain.python as grain
from array_record.python.array_record_data_source import ArrayRecordDataSource

# Import tokenization from existing module
from tokenization import encode_measurement


def deserialize_measurement(data: bytes) -> Dict[str, Any]:
    """
    Deserialize ArrayRecord bytes back to measurement dict.

    Current on-disk layout (produced by split_to_arrayrecord):
    [src_len:2][src][dst_len:2][dst][ip:8][rtt:4][time_us:8]
    - ip_version stored as int64 (little-endian)
    - event_time stored as microseconds since Unix epoch (int64)
    """
    import datetime

    offset = 0

    # Read src_addr
    src_len = int.from_bytes(data[offset:offset+2], 'little')
    offset += 2
    src_addr = data[offset:offset+src_len].decode('utf-8')
    offset += src_len

    # Read dst_addr
    dst_len = int.from_bytes(data[offset:offset+2], 'little')
    offset += 2
    dst_addr = data[offset:offset+dst_len].decode('utf-8')
    offset += dst_len

    # Read ip_version (8-byte little-endian int64)
    ip_version = int.from_bytes(data[offset:offset+8], 'little', signed=True)
    offset += 8

    # Read rtt
    rtt = np.frombuffer(data[offset:offset+4], dtype=np.float32)[0]
    offset += 4

    # Read event_time in microseconds and convert to timezone-aware datetime
    event_time_us = int.from_bytes(data[offset:offset+8], 'little', signed=True)
    event_time = datetime.datetime.fromtimestamp(
        event_time_us / 1_000_000, tz=datetime.timezone.utc
    )

    return {
        'src_addr': src_addr if src_addr else None,  # Convert empty string back to None
        'dst_addr': dst_addr if dst_addr else None,  # Convert empty string back to None
        'ip_version': ip_version,
        'rtt': float(rtt),
        'event_time': event_time,
    }


class ArrayRecordMeasurementDataSource(grain.RandomAccessDataSource):
    """
    Grain DataSource that wraps ArrayRecord files and deserializes measurements.

    This provides the same interface as ParquetMeasurementDataSource but with
    100-1000× faster random access.
    """

    def __init__(self, arrayrecord_files: List[str]):
        """
        Args:
            arrayrecord_files: List of ArrayRecord file paths
        """
        # Use ArrayRecord's built-in Grain datasource
        self._base_source = ArrayRecordDataSource(arrayrecord_files)
        self._total_measurements = len(self._base_source)

    def __len__(self):
        return self._total_measurements

    def __getitem__(self, index):
        """Get a single measurement by global index."""
        # Get raw bytes from ArrayRecord
        raw_bytes = self._base_source[index]

        # Deserialize to measurement dict
        return deserialize_measurement(raw_bytes)


class WindowedMeasurementDataSource(grain.RandomAccessDataSource):
    """
    Grain DataSource that yields windows of consecutive measurements.

    Identical to the Parquet version, but reads from ArrayRecord (much faster!).
    """

    def __init__(
        self,
        arrayrecord_files: List[str],
        window_size: int = 64,
        stride: Optional[int] = None,
    ):
        """
        Args:
            arrayrecord_files: List of ArrayRecord file paths
            window_size: Number of measurements per context window (default: 64)
            stride: Step size between windows. If None, defaults to window_size
        """
        self.base_source = ArrayRecordMeasurementDataSource(arrayrecord_files)
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

    Identical to Parquet version - just works with ArrayRecord data source.
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

        Returns:
            Dictionary with MaxText-compatible fields
        """
        # Decide training mode (40/30/30 split)
        mode_choice = self.rng.choices(
            ['full_timestamp', 'no_timestamp', 'mixed'],
            weights=self.mode_weights
        )[0]

        # Apply training mode
        if mode_choice == 'full_timestamp':
            window_sorted = sorted(window, key=lambda m: m['event_time'])
            has_timestamp = [True] * len(window_sorted)
            window_final = window_sorted

        elif mode_choice == 'no_timestamp':
            window_shuffled = window.copy()
            self.rng.shuffle(window_shuffled)
            has_timestamp = [False] * len(window_shuffled)
            window_final = window_shuffled

        else:  # mode_choice == 'mixed'
            has_timestamp = [self.rng.random() < 0.5 for _ in window]
            timestamped = [m for m, has_ts in zip(window, has_timestamp) if has_ts]
            non_timestamped = [m for m, has_ts in zip(window, has_timestamp) if not has_ts]

            timestamped_sorted = sorted(timestamped, key=lambda m: m['event_time'])
            self.rng.shuffle(non_timestamped)

            window_final = self._interleave_preserving_order(timestamped_sorted, non_timestamped)
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
        while len(tokens) < self.max_tokens:
            tokens.append(0)

        # Convert to numpy arrays (MaxText expects int32)
        tokens_array = np.array(tokens, dtype=np.int32)

        # Create segmentation (1 for real tokens, 0 for padding)
        segmentation = np.ones(self.max_tokens, dtype=np.int32)
        segmentation[original_length:] = 0

        # Create position IDs
        positions = np.arange(self.max_tokens, dtype=np.int32)

        return {
            "inputs": tokens_array,
            "inputs_segmentation": segmentation,
            "inputs_position": positions,
            "targets": tokens_array,
            "targets_segmentation": segmentation,
            "targets_position": positions,
        }

    def _interleave_preserving_order(self, ordered_list, random_list):
        """Randomly interleave two lists, preserving order of first list."""
        result = []
        i, j = 0, 0

        while i < len(ordered_list) or j < len(random_list):
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

    def _encode_window(self, window, has_timestamp):
        """Encode measurements with delta timestamps."""
        tokens = []
        prev_timestamped = None

        for meas, include_ts in zip(window, has_timestamp):
            meas_tokens = encode_measurement(
                meas,
                prev_timestamp=prev_timestamped if include_ts else None,
                include_timestamp=include_ts
            )
            tokens.extend(meas_tokens)

            if include_ts:
                prev_timestamped = meas['event_time']

        return tokens


def create_grain_pipeline(
    arrayrecord_files: List[str],
    batch_size: int = 32,
    window_size: int = 64,
    max_tokens: int = 1024,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_workers: int = 4,
    window_stride: Optional[int] = None,
) -> grain.IterDataset:
    """
    Create Grain data pipeline for network measurements using ArrayRecord.

    This is a drop-in replacement for the Parquet version, but 10-100× faster
    because ArrayRecord has efficient random access.

    Args:
        arrayrecord_files: List of ArrayRecord file paths
        batch_size: Batch size for training
        window_size: Number of measurements per context window (default: 64)
        max_tokens: Maximum tokens per sequence (default: 1024)
        shuffle: Whether to shuffle the dataset
        shuffle_seed: Random seed for shuffling
        num_workers: Number of worker threads for data loading
        window_stride: Step between windows (default: window_size)

    Returns:
        Grain IterDataset ready for MaxText training
    """
    # Create windowed data source
    source = WindowedMeasurementDataSource(
        arrayrecord_files,
        window_size=window_size,
        stride=window_stride,
    )

    # Wrap in MapDataset
    dataset = grain.MapDataset.source(source)

    # Shuffle if requested
    # With ArrayRecord, global shuffle is FAST (no decompression overhead!)
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
            prefetch_buffer_size=16,
        )
    )

    return dataset
