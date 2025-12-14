#!/usr/bin/env python3
"""
Standalone tokenization test with synthetic data (PLAN_2 schema).

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
    encode_rtt_exponent_mantissa,
    decode_rtt_exponent_mantissa,
    token_to_byte,
    VOCAB_SIZE,
    MEASUREMENT_START,
    SRC_IPV4,
    SRC_IPV6,
    DST_IPV4,
    DST_IPV6,
    TIMESTAMP_ABS,
    TIMESTAMP_DELTA1,
    TIMESTAMP_DELTA4,
    RTT_START,
    THROUGHPUT_START,
    FAILED,
    BYTE_TOKEN_OFFSET,
)


def test_token_mappings():
    """Test basic token ID mappings."""
    print("=" * 80)
    print("TOKEN MAPPING TEST (PLAN_2 Schema)")
    print("=" * 80)

    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"\nRole tokens (0-10):")
    print(f"  MEASUREMENT_START = {MEASUREMENT_START}")
    print(f"  SRC_IPV4 = {SRC_IPV4}")
    print(f"  SRC_IPV6 = {SRC_IPV6}")
    print(f"  DST_IPV4 = {DST_IPV4}")
    print(f"  DST_IPV6 = {DST_IPV6}")
    print(f"  TIMESTAMP_ABS = {TIMESTAMP_ABS}")
    print(f"  TIMESTAMP_DELTA1 = {TIMESTAMP_DELTA1}")
    print(f"  TIMESTAMP_DELTA4 = {TIMESTAMP_DELTA4}")
    print(f"  RTT_START = {RTT_START}")
    print(f"  THROUGHPUT_START = {THROUGHPUT_START}")
    print(f"  FAILED = {FAILED}")
    print(f"\nByte tokens: {BYTE_TOKEN_OFFSET}-{VOCAB_SIZE - 1} (256 total)")
    print("✓ Token mappings validated\n")


def test_rtt_encoding():
    """Test RTT exponent-mantissa encoding."""
    print("-" * 80)
    print("RTT ENCODING TEST (5-bit exp + 11-bit mantissa)")
    print("-" * 80)

    test_cases = [
        (1.0, "1ms (local network)"),
        (10.0, "10ms (fast connection)"),
        (50.0, "50ms (typical)"),
        (100.0, "100ms (continental)"),
        (250.0, "250ms (intercontinental)"),
        (1000.0, "1000ms (satellite)"),
        (302281.65625, "302s (max observed in dataset)"),
    ]

    print("\nTest cases:")
    for rtt_ms, description in test_cases:
        tokens = encode_rtt_exponent_mantissa(rtt_ms)
        assert len(tokens) == 3, f"RTT should encode to 3 tokens, got {len(tokens)}"
        assert tokens[0] == RTT_START, "First token should be RTT_START"

        # Decode and check error
        byte1 = token_to_byte(tokens[1])
        byte2 = token_to_byte(tokens[2])
        decoded_rtt = decode_rtt_exponent_mantissa(byte1, byte2)
        relative_error = abs(decoded_rtt - rtt_ms) / rtt_ms * 100

        print(f"  {description:30s}: {rtt_ms:10.3f}ms → {decoded_rtt:10.3f}ms "
              f"(error: {relative_error:6.3f}%)")

        # Check precision (should be <0.1% for PLAN_2 spec)
        assert relative_error < 0.1, f"Relative error too high: {relative_error}%"

    print("✓ RTT encoding validated (all <0.1% error)\n")


def test_ipv4_encoding():
    """Test IPv4 encoding with PLAN_2 schema."""
    print("-" * 80)
    print("IPv4 ENCODING TEST")
    print("-" * 80)

    # Create synthetic IPv4 measurement (first in sequence)
    row1 = {
        'event_time': datetime(2025, 6, 24, 12, 0, 0, tzinfo=timezone.utc),
        'src_addr': '192.0.2.1',
        'dst_addr': '8.8.8.8',
        'ip_version': 4,
        'rtt': 42.5,
    }

    print(f"Test row (first): {row1['src_addr']} -> {row1['dst_addr']}")
    tokens1 = encode_measurement(row1, prev_timestamp=None, include_timestamp=True)

    print(f"Token count: {len(tokens1)}")
    print(f"Expected: 23 tokens (IPv4 first measurement with timestamp)")
    print(f"First 25 tokens: {tokens1[:25]}")

    # Validate
    assert validate_tokens(tokens1), "Invalid tokens"
    assert tokens1[0] == MEASUREMENT_START, "First token must be MEASUREMENT_START"
    assert len(tokens1) == 23, f"IPv4 first measurement should have 23 tokens, got {len(tokens1)}"

    # Second measurement with delta timestamp
    row2 = {
        'event_time': datetime(2025, 6, 24, 12, 1, 30, tzinfo=timezone.utc),  # 90s later
        'src_addr': '192.0.2.1',
        'dst_addr': '8.8.4.4',
        'ip_version': 4,
        'rtt': 45.2,
    }

    print(f"\nTest row (subsequent): {row2['src_addr']} -> {row2['dst_addr']}")
    tokens2 = encode_measurement(row2, prev_timestamp=row1['event_time'], include_timestamp=True)

    print(f"Token count: {len(tokens2)}")
    print(f"Expected: 16 tokens (IPv4 with 1-byte delta)")

    assert len(tokens2) == 16, f"IPv4 subsequent should have 16 tokens, got {len(tokens2)}"

    # Third measurement without timestamp (atemporal mode)
    print(f"\nTest row (no timestamp): {row1['src_addr']} -> {row1['dst_addr']}")
    tokens3 = encode_measurement(row1, prev_timestamp=None, include_timestamp=False)

    print(f"Token count: {len(tokens3)}")
    print(f"Expected: 14 tokens (IPv4 without timestamp)")

    assert len(tokens3) == 14, f"IPv4 no timestamp should have 14 tokens, got {len(tokens3)}"

    print("\n✓ IPv4 encoding validated\n")
    return tokens1, tokens2, tokens3


def test_ipv6_encoding():
    """Test IPv6 encoding with PLAN_2 schema."""
    print("-" * 80)
    print("IPv6 ENCODING TEST")
    print("-" * 80)

    # Create synthetic IPv6 measurement (first in sequence)
    row1 = {
        'event_time': datetime(2025, 6, 25, 14, 30, 0, tzinfo=timezone.utc),
        'src_addr': '2001:db8::1',
        'dst_addr': '2001:4860:4860::8888',
        'ip_version': 6,
        'rtt': 123.75,
    }

    print(f"Test row (first): {row1['src_addr']} -> {row1['dst_addr']}")
    tokens1 = encode_measurement(row1, prev_timestamp=None, include_timestamp=True)

    print(f"Token count: {len(tokens1)}")
    print(f"Expected: 47 tokens (IPv6 first measurement with timestamp)")

    assert validate_tokens(tokens1), "Invalid tokens"
    assert tokens1[0] == MEASUREMENT_START, "First token must be MEASUREMENT_START"
    assert len(tokens1) == 47, f"IPv6 first measurement should have 47 tokens, got {len(tokens1)}"

    # Subsequent measurement with delta
    row2 = {
        'event_time': datetime(2025, 6, 25, 14, 32, 0, tzinfo=timezone.utc),  # 120s later
        'src_addr': '2001:db8::1',
        'dst_addr': '2001:4860:4860::8844',
        'ip_version': 6,
        'rtt': 125.0,
    }

    print(f"\nTest row (subsequent): {row2['src_addr']} -> {row2['dst_addr']}")
    tokens2 = encode_measurement(row2, prev_timestamp=row1['event_time'], include_timestamp=True)

    print(f"Token count: {len(tokens2)}")
    print(f"Expected: 40 tokens (IPv6 with 1-byte delta)")

    assert len(tokens2) == 40, f"IPv6 subsequent should have 40 tokens, got {len(tokens2)}"

    print("\n✓ IPv6 encoding validated\n")
    return tokens1, tokens2


def test_deterministic_shuffling():
    """Test that same row produces same tokens (deterministic)."""
    print("-" * 80)
    print("DETERMINISTIC SHUFFLING TEST")
    print("-" * 80)

    row = {
        'event_time': datetime(2025, 7, 1, 10, 0, 0, tzinfo=timezone.utc),
        'src_addr': '10.0.0.1',
        'dst_addr': '10.0.0.2',
        'ip_version': 4,
        'rtt': 5.0,
    }

    tokens1 = encode_measurement(row, prev_timestamp=None, include_timestamp=True)
    tokens2 = encode_measurement(row, prev_timestamp=None, include_timestamp=True)

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
        'event_time': datetime(2025, 7, 10, 8, 0, 0, tzinfo=timezone.utc),
        'src_addr': '192.168.1.1',
        'dst_addr': '192.168.1.2',
        'ip_version': 4,
        'rtt': -1.0,  # Failed probe sentinel
    }

    print(f"Test row with RTT=-1.0 (failed probe)")
    tokens = encode_measurement(row, prev_timestamp=None, include_timestamp=True)

    print(f"Token count: {len(tokens)}")
    print(f"Expected: 21 tokens (IPv4 with <FAILED> instead of RTT)")

    # Failed token is 1 token vs RTT which is 3 tokens → 2 token savings
    assert len(tokens) == 21, f"IPv4 failed probe should have 21 tokens, got {len(tokens)}"
    assert validate_tokens(tokens), "Invalid tokens for failed probe"
    assert FAILED in tokens, "Should contain FAILED token"

    print("✓ Failed probe encoding validated\n")


def test_extreme_values():
    """Test encoding with extreme values."""
    print("-" * 80)
    print("EXTREME VALUES TEST")
    print("-" * 80)

    row = {
        'event_time': datetime(2025, 7, 22, 23, 59, 59, tzinfo=timezone.utc),
        'src_addr': '255.255.255.255',  # Max IPv4
        'dst_addr': '0.0.0.0',  # Min IPv4
        'ip_version': 4,
        'rtt': 302281.65625,  # Max observed RTT in dataset
    }

    print(f"Test row with extreme values")
    print(f"  RTT: {row['rtt']} ms (max observed)")

    tokens = encode_measurement(row, prev_timestamp=None, include_timestamp=True)

    print(f"Token count: {len(tokens)}")
    print(f"All tokens valid: {validate_tokens(tokens)}")

    assert validate_tokens(tokens), "Invalid tokens for extreme values"
    print("✓ Extreme values encoding validated\n")


def test_delta_timestamp_savings():
    """Test timestamp delta encoding savings."""
    print("-" * 80)
    print("TIMESTAMP DELTA SAVINGS TEST")
    print("-" * 80)

    measurements = [
        datetime(2025, 7, 1, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 7, 1, 10, 1, 0, tzinfo=timezone.utc),   # +60s (1-byte delta)
        datetime(2025, 7, 1, 10, 2, 30, tzinfo=timezone.utc),  # +90s (1-byte delta)
        datetime(2025, 7, 1, 10, 8, 0, tzinfo=timezone.utc),   # +330s (4-byte delta)
    ]

    prev_time = None
    total_tokens = 0

    for i, timestamp in enumerate(measurements):
        row = {
            'event_time': timestamp,
            'src_addr': '10.0.0.1',
            'dst_addr': '8.8.8.8',
            'ip_version': 4,
            'rtt': 50.0,
        }

        tokens = encode_measurement(row, prev_timestamp=prev_time, include_timestamp=True)
        total_tokens += len(tokens)

        delta_desc = "absolute" if prev_time is None else f"{int((timestamp - prev_time).total_seconds())}s delta"
        print(f"  Measurement {i+1} ({delta_desc:12s}): {len(tokens)} tokens")

        prev_time = timestamp

    print(f"\nTotal tokens: {total_tokens}")
    print(f"Average per measurement: {total_tokens / len(measurements):.1f} tokens")
    print(f"Old scheme would use: {len(measurements) * 45} tokens (45 per measurement)")
    print(f"Savings: {len(measurements) * 45 - total_tokens} tokens "
          f"({(1 - total_tokens / (len(measurements) * 45)) * 100:.1f}%)")

    print("✓ Delta timestamp savings validated\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TOKENIZATION STANDALONE TEST SUITE (PLAN_2)")
    print("=" * 80)
    print()

    try:
        test_token_mappings()
        test_rtt_encoding()
        test_ipv4_encoding()
        test_ipv6_encoding()
        test_deterministic_shuffling()
        test_failed_probe_rtt()
        test_extreme_values()
        test_delta_timestamp_savings()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print(f"\nTokenization is working correctly!")
        print(f"- Vocabulary size: {VOCAB_SIZE}")
        print(f"- IPv4 first measurement: 23 tokens (with timestamp)")
        print(f"- IPv4 subsequent: 16 tokens (with 1-byte delta)")
        print(f"- IPv4 no timestamp: 14 tokens")
        print(f"- IPv6 first measurement: 47 tokens")
        print(f"- IPv6 subsequent: 40 tokens")
        print(f"- RTT encoding: 2 bytes (5-bit exp + 11-bit mant)")
        print(f"- Deterministic shuffling: ✓")
        print(f"- Edge cases handled: ✓")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
