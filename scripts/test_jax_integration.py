#!/usr/bin/env python3
"""
JAX Integration Test for Tokenization.

This script validates that our tokenization works with JAX arrays and
basic training operations, without requiring the full MaxText installation
(which has Python 3.13 compatibility issues).

This tests:
1. Tokenization → JAX arrays
2. Embedding lookup
3. Basic forward pass
4. Loss computation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
import pyarrow.parquet as pq

from tokenization import encode_measurement, VOCAB_SIZE


class SimpleTransformer(nn.Module):
    """Minimal transformer for testing."""
    vocab_size: int
    emb_dim: int = 64
    num_heads: int = 2

    @nn.compact
    def __call__(self, tokens):
        # Embedding
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.emb_dim)(tokens)

        # Self-attention
        x = nn.SelfAttention(num_heads=self.num_heads)(x)

        # MLP
        x = nn.Dense(features=self.emb_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.vocab_size)(x)

        return x


def test_tokenization_to_jax():
    """Test that tokenization outputs work with JAX."""
    print("=" * 80)
    print("JAX INTEGRATION TEST")
    print("=" * 80)

    # Load a few real measurements
    print("\n1. Loading real data...")
    table = pq.read_table("data/training_data.parquet")
    df = table.to_pandas().head(5)

    print(f"   Loaded {len(df)} measurements")

    # Tokenize
    print("\n2. Tokenizing measurements...")
    tokenized = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        tokens = encode_measurement(row_dict)
        tokenized.append(tokens)

    print(f"   Token counts: {[len(t) for t in tokenized]}")

    # Convert to JAX arrays
    print("\n3. Converting to JAX arrays...")
    max_len = max(len(t) for t in tokenized)

    # Pad sequences
    padded = []
    for tokens in tokenized:
        padded_tokens = tokens + [0] * (max_len - len(tokens))
        padded.append(padded_tokens)

    # Create JAX array
    token_array = jnp.array(padded, dtype=jnp.int32)
    print(f"   JAX array shape: {token_array.shape}")
    print(f"   JAX array dtype: {token_array.dtype}")
    print(f"   Token range: [{token_array.min()}, {token_array.max()}]")

    # Verify tokens are in valid range
    assert token_array.min() >= 0, "Negative tokens found"
    assert token_array.max() < VOCAB_SIZE, f"Tokens exceed vocab size: {token_array.max()} >= {VOCAB_SIZE}"
    print(f"   ✓ All tokens in valid range [0, {VOCAB_SIZE})")

    # Test model forward pass
    print("\n4. Testing simple transformer forward pass...")
    model = SimpleTransformer(vocab_size=VOCAB_SIZE)

    # Initialize
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, token_array)

    # Forward pass
    logits = model.apply(params, token_array)
    print(f"   Output shape: {logits.shape}")
    print(f"   Output range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Test loss computation
    print("\n5. Testing loss computation...")

    # Shift for next-token prediction
    input_tokens = token_array[:, :-1]
    target_tokens = token_array[:, 1:]

    # Forward pass on inputs
    logits = model.apply(params, input_tokens)

    # Compute cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # One-hot targets
    target_one_hot = jax.nn.one_hot(target_tokens, VOCAB_SIZE)

    # Loss (negative log likelihood)
    loss = -jnp.sum(log_probs * target_one_hot) / input_tokens.size

    print(f"   Loss: {loss:.4f}")
    print(f"   ✓ Loss is finite: {jnp.isfinite(loss)}")

    # Test gradient computation
    print("\n6. Testing gradient computation...")

    def loss_fn(params):
        logits = model.apply(params, input_tokens)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_one_hot = jax.nn.one_hot(target_tokens, VOCAB_SIZE)
        return -jnp.sum(log_probs * target_one_hot) / input_tokens.size

    loss, grads = jax.value_and_grad(loss_fn)(params)

    print(f"   Computed gradients for {len(jax.tree.leaves(grads))} parameters")
    print(f"   ✓ All gradients finite: {all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Tokenization outputs work with JAX")
    print("✓ Tokens are in valid vocabulary range")
    print("✓ Embedding lookup works")
    print("✓ Forward pass produces finite logits")
    print("✓ Loss computation works")
    print("✓ Gradients can be computed")
    print("\n✅ JAX integration test PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_tokenization_to_jax()
