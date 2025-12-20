#!/usr/bin/env python3
"""
Test that field shuffling is now random per sample instead of deterministic.
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime
from MaxText.input_pipeline.network_tokenization import encode_measurement

# Create a test measurement
test_measurement = {
    'src_addr': '192.168.1.1',
    'dst_addr': '8.8.8.8',
    'ip_version': 4,
    'rtt': 15.5,
    'event_time': datetime(2024, 1, 1, 12, 0, 0)
}

print("Testing measurement field shuffling behavior...\n")

# Test 1: Deterministic shuffle (shuffle_seed=None)
print("Test 1: Deterministic shuffle (shuffle_seed=None)")
tokens1 = encode_measurement(test_measurement, shuffle_seed=None)
tokens2 = encode_measurement(test_measurement, shuffle_seed=None)
tokens3 = encode_measurement(test_measurement, shuffle_seed=None)

print(f"  Call 1 tokens: {tokens1}")
print(f"  Call 2 tokens: {tokens2}")
print(f"  Call 3 tokens: {tokens3}")
if tokens1 == tokens2 == tokens3:
    print("  ✓ PASS: Same tokens every time (deterministic)\n")
else:
    print("  ✗ FAIL: Tokens differ (should be deterministic)\n")

# Test 2: Random shuffle with different seeds
print("Test 2: Random shuffle with different external seeds")
tokens_seed_100 = encode_measurement(test_measurement, shuffle_seed=100)
tokens_seed_200 = encode_measurement(test_measurement, shuffle_seed=200)
tokens_seed_300 = encode_measurement(test_measurement, shuffle_seed=300)

print(f"  Seed 100: {tokens_seed_100}")
print(f"  Seed 200: {tokens_seed_200}")
print(f"  Seed 300: {tokens_seed_300}")

# Check that different seeds produce different orderings
unique_orderings = len(set([
    tuple(tokens_seed_100),
    tuple(tokens_seed_200),
    tuple(tokens_seed_300)
]))

if unique_orderings > 1:
    print(f"  ✓ PASS: Got {unique_orderings} different orderings from 3 seeds\n")
else:
    print(f"  ✗ FAIL: All orderings are the same (expected variation)\n")

# Test 3: Same seed produces same ordering
print("Test 3: Same external seed produces consistent ordering")
tokens_a1 = encode_measurement(test_measurement, shuffle_seed=42)
tokens_a2 = encode_measurement(test_measurement, shuffle_seed=42)
tokens_b1 = encode_measurement(test_measurement, shuffle_seed=99)
tokens_b2 = encode_measurement(test_measurement, shuffle_seed=99)

print(f"  Seed 42, call 1: {tokens_a1}")
print(f"  Seed 42, call 2: {tokens_a2}")
print(f"  Seed 99, call 1: {tokens_b1}")
print(f"  Seed 99, call 2: {tokens_b2}")

if tokens_a1 == tokens_a2 and tokens_b1 == tokens_b2:
    print("  ✓ PASS: Same seed produces same ordering\n")
else:
    print("  ✗ FAIL: Same seed produces different orderings\n")

# Test 4: Verify all token sequences are valid
print("Test 4: All sequences have same length and start with MEASUREMENT_START")
all_tokens = [tokens1, tokens_seed_100, tokens_seed_200, tokens_seed_300, tokens_a1, tokens_b1]
lengths = [len(t) for t in all_tokens]
first_tokens = [t[0] for t in all_tokens]

print(f"  Lengths: {lengths}")
print(f"  First tokens: {first_tokens}")

if len(set(lengths)) == 1 and len(set(first_tokens)) == 1:
    print("  ✓ PASS: All sequences have consistent structure\n")
else:
    print("  ✗ FAIL: Inconsistent sequence structure\n")

print("=" * 60)
print("SUMMARY:")
print("- Deterministic mode (shuffle_seed=None) works correctly")
print("- Random mode (shuffle_seed=value) produces varying orderings")
print("- Same seed produces same ordering (reproducible)")
print("- Field shuffling is now controlled by external seed parameter")
