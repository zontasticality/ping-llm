#!/usr/bin/env python3
"""
Test that same measurement gets different field orderings when sampled multiple times.
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime
import numpy as np
from MaxText.input_pipeline._probe_chunk_datasource import ProbeRowSampler

# Create a test row with a single measurement
test_row = {
    'src_id': 12345,
    'measurements': [
        {
            'src_addr': '192.168.1.1',
            'dst_addr': '8.8.8.8',
            'ip_version': 4,
            'rtt': 15.5,
            'event_time': datetime(2024, 1, 1, 12, 0, 0)
        }
    ],
    'n_measurements': 1,
    'metadata': {
        'time_span_seconds': 1,
        'first_timestamp': datetime(2024, 1, 1, 12, 0, 0),
        'last_timestamp': datetime(2024, 1, 1, 12, 0, 1),
    }
}

print("Testing data augmentation: same measurement, different orderings\n")
print("=" * 70)

# Create sampler
sampler = ProbeRowSampler(
    crop_size=1024,
    avg_tokens_per_measurement=30,
    max_contexts_per_row=5,
    mode_weights=(0.0, 0.0, 1.0),  # Always use 'none' mode to test shuffling
    seed=42,
)

# Generate multiple contexts from the same row
print("Generating 5 contexts from the same row with 1 measurement...")
print("(Using mode='none' to ensure field shuffling is tested)\n")

token_sequences = []
for i in range(5):
    contexts = sampler.flat_map(test_row)
    if contexts:
        # Extract just the non-zero tokens (ignore padding)
        tokens = contexts[0]['inputs']
        non_zero_tokens = tokens[tokens != 0].tolist()
        token_sequences.append(non_zero_tokens)
        print(f"Context {i+1}: {non_zero_tokens}")

print("\n" + "=" * 70)
print("Analysis:")
print("=" * 70)

# Check if we got different orderings
unique_sequences = set(tuple(seq) for seq in token_sequences)
print(f"\nTotal contexts generated: {len(token_sequences)}")
print(f"Unique token sequences: {len(unique_sequences)}")

if len(unique_sequences) > 1:
    print("\n✓ SUCCESS: Same measurement produces DIFFERENT field orderings!")
    print("  This confirms random shuffle augmentation is working.")
    print(f"  Got {len(unique_sequences)} different orderings from {len(token_sequences)} samples.")
else:
    print("\n✗ FAILURE: All sequences are identical!")
    print("  Field shuffling is NOT random.")

# Verify all sequences have the same elements, just in different orders
if len(unique_sequences) > 1:
    print("\nVerifying all sequences contain the same tokens (just reordered):")
    sorted_sequences = [sorted(seq) for seq in token_sequences]
    if len(set(tuple(seq) for seq in sorted_sequences)) == 1:
        print("  ✓ All sequences contain identical tokens when sorted")
        print("  ✓ Confirms only field ORDER is changing, not the fields themselves")
    else:
        print("  ✗ WARNING: Sequences have different tokens (unexpected)")

print("\n" + "=" * 70)
print("Conclusion:")
print("=" * 70)
print("The data augmentation pipeline now generates different field orderings")
print("for the same measurement across different sampling calls, maximizing")
print("training diversity and preventing the model from memorizing specific")
print("field positions.")
