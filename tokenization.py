"""
Tokenization for network measurement data (PLAN_2 schema).

This module converts Parquet rows (network measurements) into token sequences
for training a decoder-only Transformer with MaxText.

Schema: event_time, src_addr, dst_addr, ip_version, rtt
(Note: msm_id, size, packet_error_count removed from training data)

Token vocabulary (267 total):
- IDs 0-10: Role tokens (11 total)
- IDs 11-266: Byte tokens (256 total, Byte0..Byte255)
"""

import struct
import ipaddress
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import random


# =============================================================================
# Token ID Mappings (PLAN_2 Schema)
# =============================================================================

# Role tokens (IDs 0-10)
MEASUREMENT_START = 0
SRC_IPV4 = 1
SRC_IPV6 = 2
DST_IPV4 = 3
DST_IPV6 = 4
TIMESTAMP_ABS = 5
TIMESTAMP_DELTA1 = 6
TIMESTAMP_DELTA4 = 7
RTT_START = 8
THROUGHPUT_START = 9  # Reserved for future datasets
FAILED = 10

# Byte tokens (IDs 11-266)
BYTE_TOKEN_OFFSET = 11

VOCAB_SIZE = 267


def byte_to_token(byte_val: int) -> int:
    """Convert a byte value (0-255) to its token ID (11-266)."""
    assert 0 <= byte_val <= 255, f"Invalid byte value: {byte_val}"
    return BYTE_TOKEN_OFFSET + byte_val


def token_to_byte(token_id: int) -> int:
    """Convert a token ID (11-266) back to byte value (0-255)."""
    assert BYTE_TOKEN_OFFSET <= token_id < VOCAB_SIZE, f"Invalid token ID: {token_id}"
    return token_id - BYTE_TOKEN_OFFSET


# =============================================================================
# IP Address Encoding
# =============================================================================

def parse_ipv4(ip_str: str) -> List[int]:
    """
    Parse IPv4 address string to 4 byte tokens.

    Args:
        ip_str: IPv4 address string (e.g., "192.0.2.1")

    Returns:
        List of 4 byte token IDs

    Note: Empty/invalid addresses are replaced with 0.0.0.0
    """
    if not ip_str or ip_str.strip() == '':
        ip_str = '0.0.0.0'

    ip_obj = ipaddress.IPv4Address(ip_str)
    ip_bytes = ip_obj.packed  # Returns 4 bytes
    return [byte_to_token(b) for b in ip_bytes]


def parse_ipv6(ip_str: str) -> List[int]:
    """
    Parse IPv6 address string to 16 byte tokens.

    Args:
        ip_str: IPv6 address string (e.g., "2001:db8::1")

    Returns:
        List of 16 byte token IDs

    Note: Empty/invalid addresses are replaced with ::
    """
    if not ip_str or ip_str.strip() == '':
        ip_str = '::'

    ip_obj = ipaddress.IPv6Address(ip_str)
    ip_bytes = ip_obj.packed  # Returns 16 bytes
    return [byte_to_token(b) for b in ip_bytes]


def encode_ip_merged(ip_str: str, ip_version: int, is_src: bool) -> List[int]:
    """
    Encode IP address with merged role token (SrcIPv4/SrcIPv6/DstIPv4/DstIPv6).

    Args:
        ip_str: IP address string
        ip_version: 4 or 6
        is_src: True for source IP, False for destination IP

    Returns:
        List of token IDs: [SRC_IPV4/SRC_IPV6/DST_IPV4/DST_IPV6, byte_tokens...]

    Token savings vs old scheme:
        Old: [SRC_IP_START, IPV4_START, bytes...] = 2 + 4 = 6 tokens
        New: [SRC_IPV4, bytes...] = 1 + 4 = 5 tokens
        Savings: 1 token per IP address
    """
    if ip_version == 4:
        role_token = SRC_IPV4 if is_src else DST_IPV4
        return [role_token] + parse_ipv4(ip_str)
    elif ip_version == 6:
        role_token = SRC_IPV6 if is_src else DST_IPV6
        return [role_token] + parse_ipv6(ip_str)
    else:
        raise ValueError(f"Invalid ip_version: {ip_version}")


# =============================================================================
# RTT Encoding (5-bit exponent + 11-bit mantissa)
# =============================================================================

def encode_rtt_exponent_mantissa(rtt_ms: float) -> List[int]:
    """
    Encode RTT as 5-bit exponent + 11-bit mantissa (microseconds).

    Format: value_μs = mantissa × 2^exponent
    Range: 1μs to 51 days
    Precision: ~0.049% relative error (1/2047)

    Args:
        rtt_ms: RTT value in milliseconds. Negative values indicate failed probe.

    Returns:
        List of token IDs: [RTT_START, byte1, byte2] or [FAILED]

    Encoding:
        byte1 = EEEEE MMM  (5-bit exponent, 3 MSB of mantissa)
        byte2 = MMMM MMMM  (8 LSB of mantissa)

    Examples:
        1ms (1000μs) → exp=9, mant=1000 → 1000 × 2^9 = 512000μs ❌
        Actually: 1000μs → exp=0, mant=1000 → 1000 × 2^0 = 1000μs ✓
        100ms (100000μs) → exp=6, mant=1562 → 1562 × 2^6 = 99968μs ✓
    """
    if rtt_ms < 0:
        return [FAILED]  # Use dedicated token for failed probes

    rtt_us = max(1.0, rtt_ms * 1000)  # Convert ms to μs, min 1μs

    # Find exponent (powers of 2)
    exponent = 0
    mantissa_float = rtt_us
    while mantissa_float >= 2048 and exponent < 31:
        mantissa_float /= 2
        exponent += 1

    mantissa = min(int(mantissa_float), 2047)

    # Pack: EEEEE MMM | MMMM MMMM
    #       5exp  3msb  8lsb
    byte1 = (exponent << 3) | (mantissa >> 8)
    byte2 = mantissa & 0xFF

    return [RTT_START, byte_to_token(byte1), byte_to_token(byte2)]


def decode_rtt_exponent_mantissa(byte1: int, byte2: int) -> float:
    """
    Decode RTT from 5-bit exponent + 11-bit mantissa.

    Args:
        byte1: First byte (EEEEE MMM)
        byte2: Second byte (MMMM MMMM)

    Returns:
        RTT in milliseconds
    """
    exponent = byte1 >> 3
    mantissa = ((byte1 & 0x07) << 8) | byte2

    rtt_us = mantissa * (2 ** exponent)
    rtt_ms = rtt_us / 1000

    return rtt_ms


# =============================================================================
# Timestamp Encoding (Absolute + Deltas)
# =============================================================================

def encode_u64(value: int) -> List[int]:
    """Encode 64-bit unsigned integer as 8 bytes (big-endian)."""
    value_bytes = struct.pack('>Q', value)  # Q = unsigned long long (8 bytes)
    return [byte_to_token(b) for b in value_bytes]


def encode_u32(value: int) -> List[int]:
    """Encode 32-bit unsigned integer as 4 bytes (big-endian)."""
    value_bytes = struct.pack('>I', value)  # I = unsigned int (4 bytes)
    return [byte_to_token(b) for b in value_bytes]


def encode_timestamp_delta(
    event_time,
    prev_time: Optional[datetime],
    dataset_start: Optional[datetime] = None
) -> List[int]:
    """
    Encode timestamp as absolute or delta.

    Args:
        event_time: Current measurement timestamp (pandas Timestamp or datetime)
        prev_time: Previous measurement timestamp (for delta encoding)
        dataset_start: Dataset start time (unused, kept for API compatibility)

    Returns:
        List of token IDs:
            - First measurement: [TIMESTAMP_ABS, 8 bytes] (9 tokens)
            - Delta <256s: [TIMESTAMP_DELTA1, 1 byte] (2 tokens)
            - Delta ≥256s: [TIMESTAMP_DELTA4, 4 bytes] (5 tokens)

    Optimization: 95%+ of consecutive measurements are 60-300s apart → 1-byte delta
    """
    if prev_time is None:
        # First measurement: absolute timestamp
        timestamp_sec = int(event_time.timestamp())
        return [TIMESTAMP_ABS] + encode_u64(timestamp_sec)

    delta_sec = int((event_time - prev_time).total_seconds())

    if delta_sec < 0:
        # Handle out-of-order timestamps (shouldn't happen with sorted data)
        delta_sec = 0

    if delta_sec < 256:
        # 1-byte delta (most common case)
        return [TIMESTAMP_DELTA1, byte_to_token(delta_sec)]
    else:
        # 4-byte delta (rare)
        return [TIMESTAMP_DELTA4] + encode_u32(delta_sec)


# =============================================================================
# Deterministic Shuffling
# =============================================================================

def compute_shuffle_seed(src_addr: str, dst_addr: str, event_time) -> int:
    """
    Compute deterministic shuffle seed from (src, dst, timestamp).

    Args:
        src_addr: Source IP address string
        dst_addr: Destination IP address string
        event_time: Event timestamp

    Returns:
        32-bit integer seed for RNG

    Note: Uses hash instead of simple arithmetic to avoid collisions
          for high-frequency repeated pairs.
    """
    timestamp_sec = int(event_time.timestamp())
    seed_str = f"{src_addr}|{dst_addr}|{timestamp_sec}"
    hash_bytes = hashlib.md5(seed_str.encode()).digest()
    seed = struct.unpack('>I', hash_bytes[:4])[0]  # First 4 bytes as uint32
    return seed


def shuffle_blocks_deterministic(blocks: List[List[int]], seed: int) -> List[List[int]]:
    """
    Shuffle field blocks deterministically using a seed.

    Args:
        blocks: List of token block lists
        seed: RNG seed for reproducibility

    Returns:
        Shuffled list of blocks
    """
    rng = random.Random(seed)
    shuffled = blocks.copy()
    rng.shuffle(shuffled)
    return shuffled


# =============================================================================
# Main Encoding Function
# =============================================================================

def encode_measurement(
    row: Dict[str, Any],
    prev_timestamp: Optional[datetime] = None,
    include_timestamp: bool = True
) -> List[int]:
    """
    Encode a single network measurement.

    Args:
        row: Parquet row with src_addr, dst_addr, ip_version, rtt, event_time
        prev_timestamp: Previous measurement timestamp (for delta encoding)
        include_timestamp: Whether to include timestamp in this measurement

    Returns:
        List of token IDs

    Token sequence structure:
        [MEASUREMENT_START] + shuffled_field_blocks

    Field blocks (before shuffling):
        - Source IP: [SRC_IPV4/SRC_IPV6, byte_tokens...]
        - Destination IP: [DST_IPV4/DST_IPV6, byte_tokens...]
        - Result: [RTT_START, 2 bytes] or [FAILED]
        - Timestamp (optional): [TIMESTAMP_ABS/DELTA1/DELTA4, bytes...]

    The field blocks are shuffled deterministically using (src, dst, timestamp)
    as the RNG seed to force joint distribution learning.
    """
    # Extract fields
    src_addr = row['src_addr']
    dst_addr = row['dst_addr']
    ip_version = row['ip_version']
    rtt = row['rtt']
    event_time = row['event_time']

    # Handle None/NaN addresses (pandas can return NaN for null values)
    try:
        import pandas as pd
        if pd.isna(src_addr):
            src_addr = ''
        if pd.isna(dst_addr):
            dst_addr = ''
    except ImportError:
        # pandas not available (standalone mode)
        if src_addr is None:
            src_addr = ''
        if dst_addr is None:
            dst_addr = ''

    # Encode IP addresses (merged src/dst with IPv4/IPv6)
    src_ip_block = encode_ip_merged(src_addr, ip_version, is_src=True)
    dst_ip_block = encode_ip_merged(dst_addr, ip_version, is_src=False)

    # Encode result (RTT or failed)
    if rtt < 0:
        result_block = [FAILED]
    else:
        result_block = encode_rtt_exponent_mantissa(rtt)

    # Collect field blocks (timestamp is optional)
    field_blocks = [
        src_ip_block,
        dst_ip_block,
        result_block,
    ]

    if include_timestamp:
        timestamp_block = encode_timestamp_delta(event_time, prev_timestamp)
        field_blocks.append(timestamp_block)

    # Deterministic shuffle using (src_ip, dst_ip, timestamp) as seed
    shuffle_seed = compute_shuffle_seed(src_addr, dst_addr, event_time)
    shuffled_blocks = shuffle_blocks_deterministic(field_blocks, shuffle_seed)

    # Build final sequence
    tokens = [MEASUREMENT_START]
    for block in shuffled_blocks:
        tokens.extend(block)

    return tokens


# =============================================================================
# Utility Functions
# =============================================================================

def get_vocab_size() -> int:
    """Return the vocabulary size."""
    return VOCAB_SIZE


def validate_tokens(tokens: List[int]) -> bool:
    """
    Validate that all token IDs are within valid range.

    Args:
        tokens: List of token IDs

    Returns:
        True if all tokens are valid, False otherwise
    """
    return all(0 <= t < VOCAB_SIZE for t in tokens)
