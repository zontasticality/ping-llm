# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline for network measurement data (probe_chunks and network_parquet).

This backend provides specialized data loading for network latency measurement datasets:
- probe_chunks: Probe-centric big-row data with runtime tokenization (DATA_LOADING_PLAN_3)
- network_parquet: Legacy parquet shards with window-based sampling (DATA_LOADING_PLAN_2)
"""

import ml_collections
# Note: JAX import moved to lazy import to avoid CUDA initialization in Grain workers
# import jax
import grain.python as grain

from MaxText import max_logging
from MaxText import multihost_dataloading
from MaxText.input_pipeline.probe_chunk_pipeline import build_probe_chunk_dataset


def _enable_grain_debugging(config: ml_collections.ConfigDict):
  """Enable grain debugging and visualization modes if configured."""
  # Enable debug mode for real-time execution metrics
  # grain.config.update() takes positional args, not kwargs
  if hasattr(config, 'grain_debug_mode') and config.grain_debug_mode:
    grain.config.update("py_debug_mode", True)
    max_logging.log("[GRAIN DEBUG] Enabled debug mode - will log execution summary every 60 seconds")

  # Enable visualization mode for pipeline structure
  if hasattr(config, 'grain_visualization_dir') and config.grain_visualization_dir:
    grain.config.update("py_dataset_visualization_output_dir", config.grain_visualization_dir)
    max_logging.log(f"[GRAIN DEBUG] Enabled visualization mode - output dir: {config.grain_visualization_dir}")


def make_network_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """
  Create training iterator for network measurement data.

  Supports two data formats:
  1. probe_chunks: ArrayRecord files with probe-centric rows (PLAN_3)
  2. network_parquet: Parquet shards with window sampling (PLAN_2)

  Args:
    config: Configuration dictionary with network-specific settings
    global_mesh: JAX mesh for distributed training
    process_indices: List of process indices participating in data loading

  Returns:
    MultiHostDataLoadIterator for training

  Required config fields:
    - network_data_format: "probe_chunks" or "network_parquet"
    - network_train_files: Path pattern to training data files
    - global_batch_size_to_load: Total batch size across all devices
    - max_target_length: Maximum sequence length (crop_size for probe_chunks)
    - enable_data_shuffling: Whether to shuffle data
    - data_shuffle_seed: Random seed for shuffling
    - num_epoch: Number of epochs to repeat dataset
    - grain_worker_count: Number of worker processes
    - grain_per_worker_buffer_size: Prefetch buffer size per worker

  Optional config fields:
    - grain_debug_mode: Enable grain debug mode (default: False)
    - grain_visualization_dir: Enable visualization mode with output directory
  """
  assert (
      config.global_batch_size_to_load % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."

  # Enable grain debugging if configured
  _enable_grain_debugging(config)

  # Determine data format
  if not hasattr(config, 'network_data_format'):
    raise ValueError(
        "network_data_format not specified in config. "
        "Must be 'probe_chunks' or 'network_parquet'"
    )

  # Lazy import JAX to avoid CUDA initialization in Grain worker processes
  import jax

  data_format = config.network_data_format
  batch_size = config.global_batch_size_to_load // jax.process_count()
  dataloading_host_index = process_indices.index(jax.process_index())
  dataloading_host_count = len(process_indices)

  max_logging.log(f"[NETWORK BACKEND] Creating {data_format} training iterator")
  max_logging.log(f"[NETWORK BACKEND] Batch size per host: {batch_size}")
  max_logging.log(f"[NETWORK BACKEND] Host {dataloading_host_index + 1}/{dataloading_host_count}")

  # Create dataset based on format
  if data_format == "probe_chunks":
    max_logging.log("[NETWORK BACKEND] Using probe-centric chunk dataset (DATA_LOADING_PLAN_3)")
    max_logging.log(f"[NETWORK BACKEND] Data files: {config.network_train_files}")
    max_logging.log(f"[NETWORK BACKEND] Crop size: {config.max_target_length} tokens")

    # Get RAM budget for multiprocessing (default to 8GB if not specified)
    ram_budget_mb = getattr(config, 'grain_ram_budget_mb', 8192)

    train_dataloader = build_probe_chunk_dataset(
        arrayrecord_path=config.network_train_files,
        batch_size=batch_size,
        crop_size=config.max_target_length,
        shuffle=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
        num_workers=config.grain_worker_count,
        prefetch_buffer_size=config.grain_per_worker_buffer_size,
        use_multiprocessing=True,  # Enable parallel processing
        ram_budget_mb=ram_budget_mb,
    )

    max_logging.log("[NETWORK BACKEND] Optimizations: parallel tokenization, mp_prefetch enabled")
    max_logging.log("[NETWORK BACKEND] Benefits: minimal padding (<5%), multi-scale temporal learning")

  elif data_format == "network_parquet":
    raise NotImplementedError(
        "network_parquet format not yet implemented in network backend. "
        "Use grain backend with grain_file_type='network_parquet' or implement here."
    )
  else:
    raise ValueError(
        f"Unknown network_data_format: {data_format}. "
        "Must be 'probe_chunks' or 'network_parquet'"
    )

  # Handle multi-epoch and distributed loading warnings
  if config.num_epoch > 1:
    max_logging.log(
        f"[NETWORK BACKEND] WARNING: Multi-epoch repeat (num_epoch={config.num_epoch}) "
        "not fully implemented for ArrayRecord. Consider implementing repeat in training loop."
    )

  if dataloading_host_count > 1:
    max_logging.log(
        f"[NETWORK BACKEND] WARNING: Distributed loading across {dataloading_host_count} hosts. "
        "ArrayRecord sharding not yet implemented - all hosts will read same data. "
        "For distributed training, pre-shard ArrayRecord files by host."
    )

  return multihost_dataloading.MultiHostDataLoadIterator(
      train_dataloader,
      global_mesh,
      config.generate_padding_batch_train,
      expansion_loading_factor_for_grain=config.expansion_factor_real_data,
  )


def make_network_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """
  Create evaluation iterator for network measurement data.

  Similar to make_network_train_iterator but for evaluation with:
  - No shuffling (deterministic evaluation)
  - Single epoch (num_epoch=1)
  - Separate eval config fields

  Args:
    config: Configuration dictionary
    global_mesh: JAX mesh for distributed training
    process_indices: List of process indices participating in data loading

  Returns:
    MultiHostDataLoadIterator for evaluation

  Required config fields:
    - network_data_format: "probe_chunks" or "network_parquet"
    - network_eval_files: Path pattern to evaluation data files
    - global_batch_size_to_load_eval: Total eval batch size across all devices
    - max_target_length: Maximum sequence length
    - data_shuffle_seed: Random seed (used but no shuffle for eval)
    - grain_worker_count_eval: Number of worker processes for eval
    - grain_per_worker_buffer_size_eval: Prefetch buffer size per worker for eval
  """
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Eval batch size should be divisible by number of global devices."

  # Enable grain debugging if configured
  _enable_grain_debugging(config)

  # Determine data format
  if not hasattr(config, 'network_data_format'):
    raise ValueError(
        "network_data_format not specified in config. "
        "Must be 'probe_chunks' or 'network_parquet'"
    )

  # Lazy import JAX to avoid CUDA initialization in Grain worker processes
  import jax

  data_format = config.network_data_format
  batch_size = config.global_batch_size_to_load_eval // jax.process_count()
  dataloading_host_index = process_indices.index(jax.process_index())
  dataloading_host_count = len(process_indices)

  max_logging.log(f"[NETWORK BACKEND] Creating {data_format} evaluation iterator")
  max_logging.log(f"[NETWORK BACKEND] Eval batch size per host: {batch_size}")

  # Create dataset based on format
  if data_format == "probe_chunks":
    max_logging.log("[NETWORK BACKEND] Using probe-centric chunk eval dataset (DATA_LOADING_PLAN_3)")
    max_logging.log(f"[NETWORK BACKEND] Eval files: {config.network_eval_files}")

    # Get RAM budget for multiprocessing (default to 8GB if not specified)
    ram_budget_mb = getattr(config, 'grain_ram_budget_mb', 8192)

    eval_dataloader = build_probe_chunk_dataset(
        arrayrecord_path=config.network_eval_files,
        batch_size=batch_size,
        crop_size=config.max_target_length,
        shuffle=False,  # No shuffle for eval
        shuffle_seed=config.data_shuffle_seed,
        num_workers=config.grain_worker_count_eval,
        prefetch_buffer_size=config.grain_per_worker_buffer_size_eval,
        use_multiprocessing=True,  # Enable for eval too
        ram_budget_mb=ram_budget_mb,
    )

  elif data_format == "network_parquet":
    raise NotImplementedError(
        "network_parquet format not yet implemented in network backend. "
        "Use grain backend with grain_file_type='network_parquet' or implement here."
    )
  else:
    raise ValueError(
        f"Unknown network_data_format: {data_format}. "
        "Must be 'probe_chunks' or 'network_parquet'"
    )

  if dataloading_host_count > 1:
    max_logging.log(
        f"[NETWORK BACKEND] WARNING: Distributed eval loading across {dataloading_host_count} hosts. "
        "For deterministic eval, ensure each host gets different data shards."
    )

  return multihost_dataloading.MultiHostDataLoadIterator(
      eval_dataloader,
      global_mesh,
      config.generate_padding_batch_eval,
  )
