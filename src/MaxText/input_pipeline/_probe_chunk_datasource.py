"""
Grain DataSource for probe-centric measurement rows (DATA_LOADING_PLAN_3).

This module implements:
1. ProbeRowDataSource - reads big probe rows from ArrayRecord
2. ProbeRowSampler - generates K contexts per row with PLAN_3 sampling:
   - Multi-scale temporal sampling (log-uniform window sizes)
   - Timestamp mode selection (40% full, 30% partial, 30% none)
   - Field order randomization
   - Runtime tokenization with data augmentation
"""

import struct
import random
import numpy as np
from typing import List, Dict, Any, Optional
import grain.python as grain
from datetime import datetime
from math import ceil

try:
    import array_record.python.array_record_module as array_record_module
except ImportError:
    raise ImportError(
        "array_record not installed. Install with: pip install array_record\n"
        "Note: array_record requires tensorflow-datasets"
    )

import pyarrow as pa
import pyarrow.ipc as ipc

# Import tokenization (from same package)
from MaxText.input_pipeline.network_tokenization import encode_measurement


class ProbeRowDataSource(grain.RandomAccessDataSource):
    """
    Grain DataSource for probe rows stored in ArrayRecord (PLAN_3 format).

    Each element is a row containing one probe's measurement history (up to 8MB).

    Features:
    - Random access to rows via ArrayRecord
    - Efficient deserialization of PyArrow IPC measurements
    - Returns raw measurements for runtime tokenization
    - Pickleable for multiprocessing (recreates reader per process)
    """

    def __init__(self, arrayrecord_path: str):
        """
        Args:
            arrayrecord_path: Path to ArrayRecord file (e.g., data/probe_rows/train.arrayrecord)
        """
        self.arrayrecord_path = arrayrecord_path
        self._reader = None
        self._length = None

    @property
    def reader(self):
        """Lazy-load reader (recreated after unpickling)."""
        if self._reader is None:
            self._reader = array_record_module.ArrayRecordReader(self.arrayrecord_path)
            self._length = self._reader.num_records()
        return self._reader

    def __len__(self):
        if self._length is None:
            # Initialize reader to get length
            _ = self.reader
        return self._length

    def __getstate__(self):
        """Return state for pickling (exclude unpickleable reader)."""
        return {
            'arrayrecord_path': self.arrayrecord_path,
            '_length': self._length,
        }

    def __setstate__(self, state):
        """Restore state after unpickling (reader will be recreated lazily)."""
        self.arrayrecord_path = state['arrayrecord_path']
        self._length = state['_length']
        self._reader = None  # Will be recreated on first access

    def _deserialize_measurements(self, measurements_bytes: bytes) -> List[dict]:
        """Deserialize measurements from PyArrow IPC format."""
        reader = ipc.open_stream(measurements_bytes)
        table = reader.read_all()
        reader.close()
        return table.to_pylist()

    def _read_row(self, index: int) -> dict:
        """Read and deserialize a row from ArrayRecord."""
        # Read serialized record
        record_bytes = self.reader.read([index])[0]

        # Deserialize from PyArrow IPC format
        reader = ipc.open_stream(record_bytes)
        batch = reader.read_next_batch()
        reader.close()

        # Convert to dict
        record = {
            'src_id': batch.column('src_id')[0].as_py(),
            'measurements': batch.column('measurements')[0].as_py(),  # bytes
            'n_measurements': batch.column('n_measurements')[0].as_py(),
            'time_span_seconds': batch.column('time_span_seconds')[0].as_py(),
            'first_timestamp': batch.column('first_timestamp')[0].as_py(),
            'last_timestamp': batch.column('last_timestamp')[0].as_py(),
        }

        return record

    def __getitem__(self, index):
        """
        Get a row by index.

        Returns:
            Dictionary with:
                - src_id: int
                - measurements: List[dict] (deserialized)
                - n_measurements: int
                - metadata: dict with time_span, timestamps
        """
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range [0, {self._length})")

        row = self._read_row(index)

        # Deserialize measurements
        measurements = self._deserialize_measurements(row['measurements'])

        return {
            'src_id': row['src_id'],
            'measurements': measurements,
            'n_measurements': row['n_measurements'],
            'metadata': {
                'time_span_seconds': row['time_span_seconds'],
                'first_timestamp': row['first_timestamp'],
                'last_timestamp': row['last_timestamp'],
            }
        }


class ProbeRowSampler(grain.experimental.FlatMapTransform):
    """
    Generate K training contexts per row with PLAN_3 sampling.

    For each row, generates K contexts where K is based on row size.
    Each context uses different random sampling with data augmentation.

    Data augmentation ensures same row generates different contexts on each pass:
    - Random window sampling (log-uniform for large rows)
    - Random timestamp modes (40% full, 30% partial, 30% none)
    - Random field ordering

    For each context:
    1. Sample measurement window (small row: all, large row: log-uniform)
    2. Select timestamp mode (40% full, 30% partial, 30% none)
    3. Tokenize with field randomization
    4. Pad to crop_size
    """

    def __init__(
        self,
        crop_size: int = 1024,
        avg_tokens_per_measurement: int = 30,
        max_contexts_per_row: int = 16,
        mode_weights: tuple = (0.40, 0.30, 0.30),
        seed: Optional[int] = None,
    ):
        """
        Args:
            crop_size: Number of tokens per training example (default: 1024)
            avg_tokens_per_measurement: Estimated tokens per measurement (default: 30)
            max_contexts_per_row: Maximum contexts to generate per row (default: 16)
            mode_weights: Tuple of (full_ts, partial_ts, no_ts) probabilities
            seed: Random seed for reproducibility
        """
        self.crop_size = crop_size
        self.avg_tokens_per_meas = avg_tokens_per_measurement
        self.max_K = max_contexts_per_row
        self.mode_weights = mode_weights
        self.seed = seed
        self.max_fan_out = max_contexts_per_row  # Required by FlatMapTransform
        self._call_count = 0  # Track calls for seed variation

    def flat_map(self, row: dict) -> List[dict]:
        """
        Generate K contexts from a row.

        With repeat(), same row can generate different contexts on each epoch
        due to varying seed based on call count.

        Args:
            row: Row dictionary from ProbeRowDataSource

        Returns:
            List of K context dictionaries
        """
        measurements = row['measurements']
        n = len(measurements)

        # Calculate K contexts for this row
        K = min(ceil(n / self.avg_tokens_per_meas), self.max_K)
        K = max(K, 1)  # At least 1 context

        # Vary seed per call to generate different contexts on repeat
        # Combine: base_seed + src_id + call_count
        row_seed = self.seed if self.seed is not None else 0
        src_id = row.get('src_id', 0)
        combined_seed = (row_seed * 1000000) + (src_id * 1000) + self._call_count
        self._call_count += 1

        # Use numpy Generator for random sampling (has 'integers' method for RNG detection)
        rng = np.random.default_rng(combined_seed)
        contexts = []
        for _ in range(K):
            context = self._generate_one_context(row, rng)
            contexts.append(context)

        return contexts

    def _generate_one_context(self, row: dict, rng: random.Random) -> dict:
        """Generate a single training context from row."""
        measurements = row['measurements']
        n = len(measurements)

        # Calculate target measurements for ~crop_size tokens
        target_measurements = self.crop_size // self.avg_tokens_per_meas

        # Branch: small row vs large row
        if n < target_measurements:
            # Small row: use all measurements
            meas_buffer = measurements
        else:
            # Large row: sample window
            meas_buffer = self._sample_large_row(
                measurements, target_measurements, rng
            )

        # Select timestamp mode
        mode = self._select_timestamp_mode(rng)

        # Tokenize measurements
        tokens = self._tokenize_measurements(meas_buffer, mode, rng)

        # Pad and format
        return self._format_output(tokens)

    def _sample_large_row(
        self, measurements: List[dict], target: int, rng
    ) -> List[dict]:
        """
        Sample measurements from large row using log-uniform window.

        Args:
            measurements: Full list of measurements
            target: Target number of measurements
            rng: Random number generator (numpy or Python random)

        Returns:
            Sampled and sorted measurements
        """
        n = len(measurements)

        # Check RNG type and use appropriate methods
        is_numpy = hasattr(rng, 'integers')

        # Log-uniform window size
        log_min = 0  # log(1) = 0
        log_max = np.log(n)
        if is_numpy:
            log_size = rng.uniform(log_min, log_max)
        else:
            log_size = rng.uniform(log_min, log_max)
        window_size = min(n, int(np.exp(log_size)))
        window_size = max(window_size, 1)  # At least 1

        # Random offset
        if window_size >= n:
            offset = 0
        else:
            if is_numpy:
                offset = rng.integers(0, n - window_size + 1)
            else:
                offset = rng.randint(0, n - window_size)

        # Select window
        window = measurements[offset:offset + window_size]

        # Randomly subsample from window if needed
        if len(window) <= target:
            selected = window
        else:
            # Random sample without replacement
            if is_numpy:
                indices = rng.choice(len(window), size=target, replace=False)
                selected = [window[i] for i in indices]
            else:
                selected = rng.sample(window, target)

        # Sort by timestamp
        selected.sort(key=lambda m: m['event_time'])

        return selected

    def _select_timestamp_mode(self, rng) -> str:
        """Select timestamp mode: full, partial, or none."""
        is_numpy = hasattr(rng, 'integers')

        if is_numpy:
            r = rng.random()
        else:
            r = rng.random()

        if r < self.mode_weights[0]:
            return 'full'
        elif r < self.mode_weights[0] + self.mode_weights[1]:
            return 'partial'
        else:
            return 'none'

    def _tokenize_measurements(
        self,
        measurements: List[dict],
        mode: str,
        rng,
    ) -> List[int]:
        """
        Tokenize measurements with timestamp mode and field randomization.

        Args:
            measurements: List of measurement dicts
            mode: 'full', 'partial', or 'none'
            rng: Random number generator (numpy or Python random)

        Returns:
            List of token IDs
        """
        tokens = []
        prev_timestamp = None
        is_numpy = hasattr(rng, 'integers')

        if mode == 'full':
            # Include all timestamps
            for meas in measurements:
                meas_tokens = encode_measurement(
                    meas,
                    prev_timestamp=prev_timestamp,
                    include_timestamp=True,
                )
                tokens.extend(meas_tokens)
                prev_timestamp = meas['event_time']

        elif mode == 'partial':
            # Extract random percentage (10-90%)
            extract_pct = rng.uniform(0.1, 0.9)
            n_extract = max(1, int(len(measurements) * extract_pct))

            # Random sample to extract (remove timestamps)
            if is_numpy:
                extract_indices = set(rng.choice(
                    len(measurements), size=n_extract, replace=False
                ))
            else:
                extract_indices = set(rng.sample(
                    range(len(measurements)), n_extract
                ))

            # Build two lists: timestamped and non-timestamped
            timestamped = []
            non_timestamped = []

            for i, meas in enumerate(measurements):
                if i in extract_indices:
                    non_timestamped.append(meas)
                else:
                    timestamped.append(meas)

            # Randomize non-timestamped
            if is_numpy:
                rng.shuffle(non_timestamped)
            else:
                rng.shuffle(non_timestamped)

            # Merge: interleave randomly
            all_meas = []
            ts_idx = 0
            nts_idx = 0

            for _ in range(len(measurements)):
                if ts_idx >= len(timestamped):
                    all_meas.append(('nts', non_timestamped[nts_idx]))
                    nts_idx += 1
                elif nts_idx >= len(non_timestamped):
                    all_meas.append(('ts', timestamped[ts_idx]))
                    ts_idx += 1
                else:
                    # Random choice
                    if rng.random() < 0.5:
                        all_meas.append(('ts', timestamped[ts_idx]))
                        ts_idx += 1
                    else:
                        all_meas.append(('nts', non_timestamped[nts_idx]))
                        nts_idx += 1

            # Tokenize
            for typ, meas in all_meas:
                include_ts = (typ == 'ts')
                meas_tokens = encode_measurement(
                    meas,
                    prev_timestamp=prev_timestamp if include_ts else None,
                    include_timestamp=include_ts,
                )
                tokens.extend(meas_tokens)
                if include_ts:
                    prev_timestamp = meas['event_time']

        else:  # mode == 'none'
            # No timestamps, randomize order
            shuffled = measurements.copy()
            rng.shuffle(shuffled)

            for meas in shuffled:
                meas_tokens = encode_measurement(
                    meas,
                    prev_timestamp=None,
                    include_timestamp=False,
                )
                tokens.extend(meas_tokens)

        return tokens

    def _format_output(self, tokens: List[int]) -> dict:
        """Pad to crop_size and format for MaxText."""
        tokens = np.array(tokens, dtype=np.int32)

        # Truncate if too long
        if len(tokens) > self.crop_size:
            tokens = tokens[:self.crop_size]

        original_length = len(tokens)

        # Pad if too short
        if len(tokens) < self.crop_size:
            padding = np.zeros(
                self.crop_size - len(tokens), dtype=np.int32
            )
            tokens = np.concatenate([tokens, padding])

        # Segmentation mask
        segmentation = np.ones(self.crop_size, dtype=np.int32)
        segmentation[original_length:] = 0

        # Position IDs
        positions = np.arange(self.crop_size, dtype=np.int32)

        return {
            "inputs": tokens,
            "inputs_segmentation": segmentation,
            "inputs_position": positions,
            "targets": tokens,
            "targets_segmentation": segmentation,
            "targets_position": positions,
        }
