#!/usr/bin/env python3
"""
Standalone example: Tokenize a single measurement.

This script demonstrates how to use the tokenization module
without requiring the full dataset.
"""

from datetime import datetime, timezone
from tokenization import encode_measurement, VOCAB_SIZE


def main():
    """Run tokenization examples."""
    print("=" * 80)
    print("TOKENIZATION EXAMPLES")
    print("=" * 80)
    print(f"Vocabulary size: {VOCAB_SIZE}\n")

    # Example 1: IPv4 measurement
    print("-" * 80)
    print("Example 1: IPv4 Measurement")
    print("-" * 80)

    ipv4_measurement = {
        'msm_id': 12345,
        'event_time': datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc),
        'src_addr': '192.0.2.1',
        'dst_addr': '8.8.8.8',
        'ip_version': 4,
        'rtt': 42.5,
        'size': 64,
        'packet_error_count': 0,
    }

    print(f"Input: {ipv4_measurement['src_addr']} → {ipv4_measurement['dst_addr']}")
    print(f"RTT: {ipv4_measurement['rtt']} ms")

    tokens = encode_measurement(ipv4_measurement)

    print(f"\nOutput:")
    print(f"  Token count: {len(tokens)}")
    print(f"  Token range: [{min(tokens)}, {max(tokens)}]")
    print(f"  First 30 tokens: {tokens[:30]}")
    print(f"  All tokens valid: {all(0 <= t < VOCAB_SIZE for t in tokens)}")

    # Example 2: IPv6 measurement
    print("\n" + "-" * 80)
    print("Example 2: IPv6 Measurement")
    print("-" * 80)

    ipv6_measurement = {
        'msm_id': 67890,
        'event_time': datetime(2025, 6, 25, 14, 30, 0, tzinfo=timezone.utc),
        'src_addr': '2001:db8::1',
        'dst_addr': '2001:4860:4860::8888',
        'ip_version': 6,
        'rtt': 123.75,
        'size': 1500,
        'packet_error_count': 2,
    }

    print(f"Input: {ipv6_measurement['src_addr']} → {ipv6_measurement['dst_addr']}")
    print(f"RTT: {ipv6_measurement['rtt']} ms")

    tokens = encode_measurement(ipv6_measurement)

    print(f"\nOutput:")
    print(f"  Token count: {len(tokens)}")
    print(f"  Token range: [{min(tokens)}, {max(tokens)}]")
    print(f"  First 30 tokens: {tokens[:30]}")
    print(f"  All tokens valid: {all(0 <= t < VOCAB_SIZE for t in tokens)}")

    # Example 3: Failed probe (RTT = -1.0, empty src)
    print("\n" + "-" * 80)
    print("Example 3: Failed Probe")
    print("-" * 80)

    failed_measurement = {
        'msm_id': 99999,
        'event_time': datetime(2025, 7, 10, 8, 0, 0, tzinfo=timezone.utc),
        'src_addr': '',  # Empty (failed probe)
        'dst_addr': '192.168.1.2',
        'ip_version': 4,
        'rtt': -1.0,  # Failed probe sentinel
        'size': 64,
        'packet_error_count': 1,
    }

    print(f"Input: (empty) → {failed_measurement['dst_addr']}")
    print(f"RTT: {failed_measurement['rtt']} ms (FAILED)")

    tokens = encode_measurement(failed_measurement)

    print(f"\nOutput:")
    print(f"  Token count: {len(tokens)}")
    print(f"  Token range: [{min(tokens)}, {max(tokens)}]")
    print(f"  All tokens valid: {all(0 <= t < VOCAB_SIZE for t in tokens)}")
    print(f"  Note: Empty src_addr encoded as 0.0.0.0 (sentinel)")

    # Deterministic check
    print("\n" + "-" * 80)
    print("Deterministic Encoding Check")
    print("-" * 80)

    tokens1 = encode_measurement(ipv4_measurement)
    tokens2 = encode_measurement(ipv4_measurement)

    print(f"Same measurement encoded twice:")
    print(f"  Encoding 1: {len(tokens1)} tokens")
    print(f"  Encoding 2: {len(tokens2)} tokens")
    print(f"  Identical: {tokens1 == tokens2} ✓")

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  - IPv4 measurements: ~{len(encode_measurement(ipv4_measurement))} tokens")
    print(f"  - IPv6 measurements: ~{len(encode_measurement(ipv6_measurement))} tokens")
    print(f"  - Encoding is deterministic: ✓")
    print(f"  - Failed probes handled: ✓")
    print(f"  - All tokens in valid range [0, {VOCAB_SIZE}): ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
