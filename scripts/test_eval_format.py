#!/usr/bin/env python3
"""
Test the evaluation formatting without requiring a full model.

This creates synthetic data to show what the eval output will look like.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from datetime import datetime
from MaxText.input_pipeline.network_tokenization import (
    encode_measurement,
    decode_token_stream_pretty,
    TOKEN_NAMES,
    token_to_byte,
    BYTE_TOKEN_OFFSET,
    VOCAB_SIZE,
)

def format_token_comparison(pos: int, actual: int, predicted: int, correct: bool) -> str:
    """Format a single token comparison for display."""
    actual_str = _token_to_str(actual)
    predicted_str = _token_to_str(predicted)

    status = "✓" if correct else "✗"
    color = "\033[92m" if correct else "\033[91m"  # Green if correct, red if wrong
    reset = "\033[0m"

    return f"{color}{status}{reset} Pos {pos:3d}: Actual={actual_str:20s} | Predicted={predicted_str:20s}"


def _token_to_str(token: int) -> str:
    """Convert a single token to a readable string."""
    if token in TOKEN_NAMES:
        return TOKEN_NAMES[token]
    elif BYTE_TOKEN_OFFSET <= token < VOCAB_SIZE:
        val = token_to_byte(token)
        return f"Byte(0x{val:02X}/{val:3d})"
    else:
        return f"Unknown({token})"


# Create a sample measurement sequence
measurements = [
    {
        'src_addr': '192.168.1.1',
        'dst_addr': '8.8.8.8',
        'ip_version': 4,
        'rtt': 15.5,
        'event_time': datetime(2024, 1, 1, 12, 0, 0)
    },
    {
        'src_addr': '192.168.1.1',
        'dst_addr': '1.1.1.1',
        'ip_version': 4,
        'rtt': 12.3,
        'event_time': datetime(2024, 1, 1, 12, 0, 1)
    },
]

# Encode measurements
tokens = []
prev_timestamp = None
for meas in measurements:
    meas_tokens = encode_measurement(
        meas,
        prev_timestamp=prev_timestamp,
        include_timestamp=True,
        shuffle_seed=42,  # Use fixed seed for reproducibility
    )
    tokens.extend(meas_tokens)
    prev_timestamp = meas['event_time']

print("="*80)
print("SAMPLE EVALUATION OUTPUT FORMAT")
print("="*80)
print()

# Show the sequence
pretty_tokens = decode_token_stream_pretty(tokens)
print(f"Sequence: {' '.join(pretty_tokens[:30])}")
if len(pretty_tokens) > 30:
    print(f"          ... (+{len(pretty_tokens) - 30} more tokens)")

print()
print(f"Total length: {len(tokens)} tokens")
print()

# Simulate predictions (for demo, we'll make some correct and some incorrect)
print("="*80)
print("NEXT-TOKEN PREDICTIONS (Simulated)")
print("="*80)
print()

np.random.seed(42)
correct_count = 0
total_count = min(30, len(tokens) - 1)

print("Showing first 30 predictions:")
print()

for pos in range(total_count):
    actual = tokens[pos + 1]

    # Simulate: 80% correct, 20% wrong
    if np.random.random() < 0.8:
        predicted = actual
        correct = True
    else:
        # Random wrong prediction
        predicted = np.random.randint(0, VOCAB_SIZE)
        correct = False

    if correct:
        correct_count += 1

    print(format_token_comparison(pos, actual, predicted, correct))

accuracy = correct_count / total_count * 100
print()
print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)")
print()

print("="*80)
print("COLOR LEGEND:")
print("="*80)
print("\033[92m✓\033[0m = Correct prediction (green check)")
print("\033[91m✗\033[0m = Incorrect prediction (red X)")
print()

print("="*80)
print("INTERPRETATION:")
print("="*80)
print("This format allows you to:")
print("1. See exactly what tokens the model predicted vs what was actual")
print("2. Identify patterns in where the model makes mistakes")
print("3. Track improvement over training by comparing checkpoints")
print("4. Understand which token types (IPs, RTTs, timestamps) are harder to predict")
print()

print("Run the full script with:")
print("  python scripts/eval_next_token_predictions.py \\")
print("    --checkpoint outputs/latency_network/full_run/full_run/checkpoints/2000/items \\")
print("    --data data/probe_rows/test.arrayrecord \\")
print("    --num-sequences 5 \\")
print("    --max-length 100")
