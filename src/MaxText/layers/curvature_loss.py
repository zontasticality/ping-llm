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

"""Diagrammatic Curvature Loss (GT-Lite).

Implements Algorithm 2 from Mahadevan's "Categories for AGI": approximate
diagrammatic curvature energy for autoregressive training. Penalizes
violations of triangle commutativity in the learned representation space.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

from MaxText.common_types import Config


class DiagrammaticCurvatureLoss(nn.Module):
  """Computes diagrammatic curvature loss over sampled position triples.

  For each sampled triangle (i, j, k) of sequence positions, learns edge
  transform projections M_ij, M_jk, M_ik and penalizes the commutativity
  violation: ||M_ik(h_i) - M_jk(M_ij(h_i))||².

  Attributes:
    config: Model configuration.
    mesh: Device mesh for sharding.
  """

  config: Config
  mesh: Mesh

  @nn.compact
  def __call__(self, hidden_states):
    """Computes the diagrammatic curvature loss.

    Args:
      hidden_states: Tensor of shape [batch, length, emb_dim].

    Returns:
      Scalar curvature loss value.
    """
    cfg = self.config
    curvature_dim = cfg.geom_curvature_dim
    num_triangles = cfg.geom_num_triangles
    batch_size, seq_len, emb_dim = hidden_states.shape

    # Edge transform projections: project from emb_dim to curvature_dim.
    # M_ij: direct edge from i to j
    # M_jk: edge from j to k
    # M_ik: direct edge from i to k (the "shortcut")
    m_ij = nn.Dense(
        features=curvature_dim,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        name="edge_ij",
    )
    m_jk = nn.Dense(
        features=curvature_dim,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        name="edge_jk",
    )
    m_ik = nn.Dense(
        features=curvature_dim,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        name="edge_ik",
    )

    # Sample random triples (i, j, k) where i < j < k for causal consistency.
    # Use a deterministic-ish approach: linearly spaced triples to avoid
    # needing an RNG key in the loss computation.
    # Generate indices spread across the sequence length.
    tri_indices = jnp.arange(num_triangles)

    # Create three sets of indices that maintain i < j < k ordering
    # by dividing the sequence into thirds.
    third = seq_len // 3
    third = jnp.maximum(third, 1)  # Avoid zero division for very short sequences.

    idx_i = (tri_indices * 7) % third  # Spread across first third
    idx_j = third + (tri_indices * 11) % third  # Spread across second third
    idx_k = 2 * third + (tri_indices * 13) % (seq_len - 2 * third)  # Spread across last third

    # Gather hidden states at sampled positions: [batch, num_triangles, emb_dim]
    h_i = hidden_states[:, idx_i, :]

    # Compute the two paths through the simplicial diagram:
    # Direct path: M_ik(h_i) — the direct morphism from i to k
    direct = m_ik(h_i)  # [batch, num_triangles, curvature_dim]

    # Composed path: M_jk(M_ij(h_i)) — composition through intermediate j
    via_j = m_jk(m_ij(h_i))  # [batch, num_triangles, curvature_dim]

    # Commutativity violation: squared L2 norm of difference
    diff = direct - via_j
    curvature = jnp.mean(jnp.sum(diff ** 2, axis=-1))

    return curvature
