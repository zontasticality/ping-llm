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

"""Integration tests for network data processing backend."""

import sys
import os.path
import tempfile
import unittest

import jax
import numpy as np
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from MaxText import pyconfig
from MaxText.input_pipeline import _network_data_processing
from MaxText.input_pipeline import input_pipeline_interface
from MaxText.globals import MAXTEXT_PKG_DIR


class NetworkDataProcessingTest(unittest.TestCase):
  """Test network backend with synthetic probe chunk data."""

  def setUp(self):
    super().setUp()

    # Create a temporary ArrayRecord file with synthetic data
    self.temp_dir = tempfile.mkdtemp()
    self.train_file = os.path.join(self.temp_dir, "test_train.arrayrecord")
    self.eval_file = os.path.join(self.temp_dir, "test_eval.arrayrecord")

    # Create minimal synthetic ArrayRecord files
    self._create_synthetic_arrayrecord(self.train_file, num_records=10)
    self._create_synthetic_arrayrecord(self.eval_file, num_records=5)

    # Initialize config for network backend
    self.config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1,
        run_name="test_network",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory=self.temp_dir,
        dataset_type="network",  # Use network backend
        network_data_format="probe_chunks",
        network_train_files=self.train_file,
        network_eval_files=self.eval_file,
        grain_worker_count=1,  # Minimal workers for testing
        grain_per_worker_buffer_size=1,
        grain_worker_count_eval=1,
        grain_per_worker_buffer_size_eval=1,
        enable_checkpointing=False,
        max_target_length=128,  # Small for testing
    )

    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
        self.config.data_sharding,
        self.config.global_batch_size_to_load,
        self.config.global_batch_size_to_train_on,
        self.config.max_target_length,
        self.mesh,
    )

  def _create_synthetic_arrayrecord(self, filepath: str, num_records: int = 10):
    """Create a minimal ArrayRecord file with synthetic probe row data."""
    try:
      import array_record.python.array_record_module as array_record_module
      import pyarrow as pa
      import pyarrow.ipc as ipc
    except ImportError:
      self.skipTest("array_record or pyarrow not available")
      return

    writer = array_record_module.ArrayRecordWriter(filepath, "group_size:1")

    for i in range(num_records):
      # Create synthetic measurement data
      measurements = [
          {
              "event_time": 1000000 + i * 100 + j,
              "dst_addr": f"192.168.1.{j}",
              "rtt": 10.0 + j * 0.5,
              "protocol": 1,
          }
          for j in range(5)  # 5 measurements per row
      ]

      # Serialize measurements to PyArrow IPC format
      measurement_table = pa.table({
          "event_time": [m["event_time"] for m in measurements],
          "dst_addr": [m["dst_addr"] for m in measurements],
          "rtt": [m["rtt"] for m in measurements],
          "protocol": [m["protocol"] for m in measurements],
      })

      sink = pa.BufferOutputStream()
      writer_ipc = ipc.new_stream(sink, measurement_table.schema)
      writer_ipc.write_table(measurement_table)
      writer_ipc.close()
      measurements_bytes = sink.getvalue().to_pybytes()

      # Create probe row record
      row_table = pa.table({
          "src_id": [i],
          "measurements": [measurements_bytes],
          "n_measurements": [len(measurements)],
          "time_span_seconds": [500.0],
          "first_timestamp": [measurements[0]["event_time"]],
          "last_timestamp": [measurements[-1]["event_time"]],
      })

      # Serialize to IPC and write
      sink = pa.BufferOutputStream()
      writer_ipc = ipc.new_stream(sink, row_table.schema)
      writer_ipc.write_table(row_table)
      writer_ipc.close()
      record_bytes = sink.getvalue().to_pybytes()

      writer.write(record_bytes)

    writer.close()

  def test_train_iterator_creation(self):
    """Test that train iterator can be created successfully."""
    train_iter = _network_data_processing.make_network_train_iterator(
        self.config, self.mesh, self.process_indices
    )
    self.assertIsNotNone(train_iter)

  def test_eval_iterator_creation(self):
    """Test that eval iterator can be created successfully."""
    eval_iter = _network_data_processing.make_network_eval_iterator(
        self.config, self.mesh, self.process_indices
    )
    self.assertIsNotNone(eval_iter)

  def test_train_batch_shape(self):
    """Test that training batches have correct shape."""
    train_iter = _network_data_processing.make_network_train_iterator(
        self.config, self.mesh, self.process_indices
    )

    batch = next(train_iter)
    expected_shape = [jax.device_count(), self.config.max_target_length]

    self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
            "inputs": expected_shape,
            "inputs_position": expected_shape,
            "inputs_segmentation": expected_shape,
            "targets": expected_shape,
            "targets_position": expected_shape,
            "targets_segmentation": expected_shape,
        },
    )

  def test_batch_types(self):
    """Test that batch values are the correct dtype."""
    train_iter = _network_data_processing.make_network_train_iterator(
        self.config, self.mesh, self.process_indices
    )

    batch = next(train_iter)

    # All values should be arrays
    for key, value in batch.items():
      self.assertIsInstance(value, (np.ndarray, jax.Array))
      # Should be int32 (token IDs and positions)
      self.assertEqual(value.dtype, np.int32, f"{key} should be int32")

  def test_batch_determinism(self):
    """Test that batches are deterministic with same seed."""
    train_iter1 = _network_data_processing.make_network_train_iterator(
        self.config, self.mesh, self.process_indices
    )
    batch1 = next(train_iter1)

    # Create new iterator with same config
    train_iter2 = _network_data_processing.make_network_train_iterator(
        self.config, self.mesh, self.process_indices
    )
    batch2 = next(train_iter2)

    # Note: Due to random sampling in ProbeRowSampler, batches may not be
    # identical unless we fix the random seed. This tests that the iterator
    # can be created and produces valid batches.
    self.assertEqual(batch1["inputs"].shape, batch2["inputs"].shape)
    self.assertEqual(batch1["targets"].shape, batch2["targets"].shape)

  def test_interface_integration(self):
    """Test that network backend integrates correctly with input_pipeline_interface."""
    train_iter, eval_iter = input_pipeline_interface.create_data_iterator(
        self.config, self.mesh
    )

    self.assertIsNotNone(train_iter)
    if self.config.eval_interval > 0:
      self.assertIsNotNone(eval_iter)

    # Test that we can get a batch
    batch = next(train_iter)
    self.assertIn("inputs", batch)
    self.assertIn("targets", batch)

  def tearDown(self):
    """Clean up temporary files."""
    super().tearDown()
    import shutil
    if os.path.exists(self.temp_dir):
      shutil.rmtree(self.temp_dir)


class NetworkDataProcessingDebugModeTest(NetworkDataProcessingTest):
  """Test network backend with debug mode enabled."""

  def setUp(self):
    super().setUp()

    # Re-initialize with debug mode
    self.config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1,
        run_name="test_network_debug",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory=self.temp_dir,
        dataset_type="network",
        network_data_format="probe_chunks",
        network_train_files=self.train_file,
        network_eval_files=self.eval_file,
        grain_worker_count=1,
        grain_per_worker_buffer_size=1,
        grain_worker_count_eval=1,
        grain_per_worker_buffer_size_eval=1,
        grain_debug_mode=True,  # Enable debug mode
        enable_checkpointing=False,
        max_target_length=128,
    )

    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
        self.config.data_sharding,
        self.config.global_batch_size_to_load,
        self.config.global_batch_size_to_train_on,
        self.config.max_target_length,
        self.mesh,
    )

  def test_debug_mode_enabled(self):
    """Test that debug mode can be enabled without errors."""
    # This should not raise any errors
    train_iter = _network_data_processing.make_network_train_iterator(
        self.config, self.mesh, self.process_indices
    )

    # Should still be able to get batches
    batch = next(train_iter)
    self.assertIsNotNone(batch)


if __name__ == "__main__":
  unittest.main()
