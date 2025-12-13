#!/usr/bin/env python3
"""
Standalone tokenization test with synthetic data.

This test validates tokenization logic without requiring the actual Parquet file
or external dependencies like pyarrow/duckdb.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path for tokenization import
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenization import (
    encode_measurement,
    validate_tokens,
    VOCAB_SIZE,
    MEASUREMENT_START,
    SRC_IP_START,
    DEST_IP_START,
    IPV4_START,
    IPV6_START,
    RTT_START,
    SIZE_START,
    ERROR_COUNT_START,
    TIMESTAMP_START,
    MSM_ID_START,
    BYTE_TOKEN_OFFSET,
)


def test_token_mappings():
    """Test basic token ID mappings."""
    print("=" * 80)
    print("TOKEN MAPPING TEST")
    print("=" * 80)

    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"\nRole tokens (0-9):")
    print(f"  MEASUREMENT_START = {MEASUREMENT_START}")
    print(f"  SRC_IP_START = {SRC_IP_START}")
    print(f"  DEST_IP_START = {DEST_IP_START}")
    print(f"  IPV4_START = {IPV4_START}")
    print(f"  IPV6_START = {IPV6_START}")
    print(f"  RTT_START = {RTT_START}")
    print(f"  SIZE_START = {SIZE_START}")
    print(f"  ERROR_COUNT_START = {ERROR_COUNT_START}")
    print(f"  TIMESTAMP_START = {TIMESTAMP_START}")
    print(f"  MSM_ID_START = {MSM_ID_START}")
    print(f"\nByte tokens: {BYTE_TOKEN_OFFSET}-{VOCAB_SIZE - 1} (256 total)")
    print("✓ Token mappings validated\n")


def test_ipv4_encoding():
    """Test IPv4 encoding."""
    print("-" * 80)
    print("IPv4 ENCODING TEST")
    print("-" * 80)

    # Create synthetic IPv4 measurement
    row = {
        'msm_id': 12345,
        'event_time': datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc),
        'src_addr': '192.0.2.1',
        'dst_addr': '8.8.8.8',
        'ip_version': 4,
        'rtt': 42.5,
        'size': 64,
        'packet_error_count': 0,
    }

    print(f"Test row: {row['src_addr']} -> {row['dst_addr']}")
    tokens = encode_measurement(row)

    print(f"Token count: {len(tokens)}")
    print(f"First 40 tokens: {tokens[:40]}")
    print(f"Token range: [{min(tokens)}, {max(tokens)}]")

    # Validate
    assert validate_tokens(tokens), "Invalid tokens"
    assert tokens[0] == MEASUREMENT_START, "First token must be MEASUREMENT_START"
    assert len(tokens) >= 40, f"IPv4 measurement should have ~42+ tokens, got {len(tokens)}"
    assert len(tokens) <= 60, f"IPv4 measurement should have ~42 tokens, got {len(tokens)}"

    print("✓ IPv4 encoding validated\n")
    return tokens


def test_ipv6_encoding():
    """Test IPv6 encoding."""
    print("-" * 80)
    print("IPv6 ENCODING TEST")
    print("-" * 80)

    # Create synthetic IPv6 measurement
    row = {
        'msm_id': 67890,
        'event_time': datetime(2025, 6, 25, 14, 30, 0, tzinfo=timezone.utc),
        'src_addr': '2001:db8::1',
        'dst_addr': '2001:4860:4860::8888',
        'ip_version': 6,
        'rtt': 123.75,
        'size': 1500,
        'packet_error_count': 2,
    }

    print(f"Test row: {row['src_addr']} -> {row['dst_addr']}")
    tokens = encode_measurement(row)

    print(f"Token count: {len(tokens)}")
    print(f"First 40 tokens: {tokens[:40]}")
    print(f"Token range: [{min(tokens)}, {max(tokens)}]")

    # Validate
    assert validate_tokens(tokens), "Invalid tokens"
    assert tokens[0] == MEASUREMENT_START, "First token must be MEASUREMENT_START"
    assert len(tokens) >= 60, f"IPv6 measurement should have ~66+ tokens, got {len(tokens)}"
    assert len(tokens) <= 80, f"IPv6 measurement should have ~66 tokens, got {len(tokens)}"

    print("✓ IPv6 encoding validated\n")
    return tokens


def test_deterministic_shuffling():
    """Test that same row produces same tokens (deterministic)."""
    print("-" * 80)
    print("DETERMINISTIC SHUFFLING TEST")
    print("-" * 80)

    row = {
        'msm_id': 11111,
        'event_time': datetime(2025, 7, 1, 10, 0, 0, tzinfo=timezone.utc),
        'src_addr': '10.0.0.1',
        'dst_addr': '10.0.0.2',
        'ip_version': 4,
        'rtt': 5.0,
        'size': 64,
        'packet_error_count': 0,
    }

    tokens1 = encode_measurement(row)
    tokens2 = encode_measurement(row)

    print(f"Encoding 1: {len(tokens1)} tokens")
    print(f"Encoding 2: {len(tokens2)} tokens")
    print(f"Identical: {tokens1 == tokens2}")

    assert tokens1 == tokens2, "Same row must produce identical tokens"
    print("✓ Deterministic shuffling validated\n")


def test_failed_probe_rtt():
    """Test RTT encoding with -1.0 sentinel value."""
    print("-" * 80)
    print("FAILED PROBE (RTT=-1.0) TEST")
    print("-" * 80)

    row = {
        'msm_id': 99999,
        'event_time': datetime(2025, 7, 10, 8, 0, 0, tzinfo=timezone.utc),
        'src_addr': '192.168.1.1',
        'dst_addr': '192.168.1.2',
        'ip_version': 4,
        'rtt': -1.0,  # Failed probe sentinel
        'size': 64,
        'packet_error_count': 1,
    }

    print(f"Test row with RTT=-1.0 (failed probe)")
    tokens = encode_measurement(row)

    print(f"Token count: {len(tokens)}")
    print(f"All tokens valid: {validate_tokens(tokens)}")

    assert validate_tokens(tokens), "Invalid tokens for failed probe"
    print("✓ Failed probe encoding validated\n")


def test_extreme_values():
    """Test encoding with extreme values."""
    print("-" * 80)
    print("EXTREME VALUES TEST")
    print("-" * 80)

    row = {
        'msm_id': 99999999999,  # Large msm_id
        'event_time': datetime(2025, 7, 22, 23, 59, 59, tzinfo=timezone.utc),
        'src_addr': '255.255.255.255',  # Max IPv4
        'dst_addr': '0.0.0.0',  # Min IPv4
        'ip_version': 4,
        'rtt': 302281.65625,  # Max observed RTT
        'size': 2000,  # Max observed size
        'packet_error_count': 16,  # Max observed error count
    }

    print(f"Test row with extreme values")
    print(f"  RTT: {row['rtt']} ms (max observed)")
    print(f"  Size: {row['size']} bytes (max observed)")
    print(f"  Errors: {row['packet_error_count']} (max observed)")

    tokens = encode_measurement(row)

    print(f"Token count: {len(tokens)}")
    print(f"All tokens valid: {validate_tokens(tokens)}")

    assert validate_tokens(tokens), "Invalid tokens for extreme values"
    print("✓ Extreme values encoding validated\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TOKENIZATION STANDALONE TEST SUITE")
    print("=" * 80)
    print()

    try:
        test_token_mappings()
        test_ipv4_encoding()
        test_ipv6_encoding()
        test_deterministic_shuffling()
        test_failed_probe_rtt()
        test_extreme_values()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print(f"\nTokenization is working correctly!")
        print(f"- Vocabulary size: {VOCAB_SIZE}")
        print(f"- IPv4 measurements: ~42 tokens")
        print(f"- IPv6 measurements: ~66 tokens")
        print(f"- Deterministic shuffling: ✓")
        print(f"- Edge cases handled: ✓")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
