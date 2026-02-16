# Copyright 2024 Google LLC
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

"""Geometric Mixer (GT-Lite): local 1D depthwise convolution with gated residual.

Implements the local geometric mixer from Mahadevan's "Categories for AGI",
providing an approximate coend over local neighborhoods via causal depthwise
convolution and SiLU gating.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

from MaxText.common_types import Config
from MaxText.layers.normalizations import rms_norm


class GeometricMixer(nn.Module):
  """Local geometric mixer using causal depthwise convolution + gated residual.

  This module adds a smoothed local neighborhood representation after attention,
  approximating an approximate coend over local neighborhoods per the GT-Lite
  formulation.

  Attributes:
    config: Model configuration.
    mesh: Device mesh for sharding.
  """

  config: Config
  mesh: Mesh

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies the geometric mixer.

    Args:
      x: Input tensor of shape [batch, length, emb_dim].
      deterministic: If True, disables dropout.

    Returns:
      Output tensor of shape [batch, length, emb_dim].
    """
    cfg = self.config
    kernel_size = cfg.geom_mixer_kernel_size
    emb_dim = x.shape[-1]

    # Causal padding: pad left only so we never attend to future positions.
    # x shape: [batch, length, emb_dim] -> pad along length axis
    pad_width = ((0, 0), (kernel_size - 1, 0), (0, 0))
    x_padded = jnp.pad(x, pad_width)

    # Depthwise 1D convolution: each feature channel gets its own kernel.
    # Using nn.Conv with feature_group_count=emb_dim for depthwise behavior.
    conv_out = nn.Conv(
        features=emb_dim,
        kernel_size=(kernel_size,),
        padding="VALID",
        feature_group_count=emb_dim,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        name="depthwise_conv",
    )(x_padded)

    # SiLU gating: learn a gate from the input and apply element-wise.
    gate = nn.Dense(
        features=emb_dim,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        name="gate_proj",
    )(x)
    gate = nn.silu(gate)

    # Gated output
    gated_out = conv_out * gate

    # RMSNorm before output
    normed_out = rms_norm(
        num_features=emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="geom_mixer_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(gated_out)

    return normed_out
