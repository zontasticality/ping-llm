"""
Tokenization for network measurement data.

This module converts Parquet rows (network measurements) into token sequences
for training a decoder-only Transformer with MaxText.

Schema: msm_id, event_time, src_addr, dst_addr, ip_version, rtt, size, packet_error_count

Token vocabulary (266 total):
- IDs 0-9: Role tokens (10 total)
- IDs 10-265: Byte tokens (256 total, Byte0..Byte255)
"""

import struct
import ipaddress
from typing import List, Dict, Any
import random


# =============================================================================
# Token ID Mappings
# =============================================================================

# Role tokens (IDs 0-9)
MEASUREMENT_START = 0
SRC_IP_START = 1
DEST_IP_START = 2
IPV4_START = 3
IPV6_START = 4
RTT_START = 5
SIZE_START = 6
ERROR_COUNT_START = 7
TIMESTAMP_START = 8
MSM_ID_START = 9

# Byte tokens (IDs 10-265)
BYTE_TOKEN_OFFSET = 10

VOCAB_SIZE = 266


def byte_to_token(byte_val: int) -> int:
    """Convert a byte value (0-255) to its token ID (10-265)."""
    assert 0 <= byte_val <= 255, f"Invalid byte value: {byte_val}"
    return BYTE_TOKEN_OFFSET + byte_val


def token_to_byte(token_id: int) -> int:
    """Convert a token ID (10-265) back to byte value (0-255)."""
    assert BYTE_TOKEN_OFFSET <= token_id < VOCAB_SIZE, f"Invalid token ID: {token_id}"
    return token_id - BYTE_TOKEN_OFFSET


# =============================================================================
# IP Address Parsing
# =============================================================================

def parse_ipv4(ip_str: str) -> List[int]:
    """
    Parse IPv4 address string to 4 bytes.

    Args:
        ip_str: IPv4 address string (e.g., "192.0.2.1")

    Returns:
        List of 4 byte token IDs

    Note: Empty/invalid addresses are replaced with 0.0.0.0 (failed probe sentinel)
    """
    # Handle empty or invalid IP addresses (from failed probes)
    if not ip_str or ip_str.strip() == '':
        ip_str = '0.0.0.0'

    ip_obj = ipaddress.IPv4Address(ip_str)
    ip_bytes = ip_obj.packed  # Returns 4 bytes
    return [byte_to_token(b) for b in ip_bytes]


def parse_ipv6(ip_str: str) -> List[int]:
    """
    Parse IPv6 address string to 16 bytes.

    Args:
        ip_str: IPv6 address string (e.g., "2001:db8::1")

    Returns:
        List of 16 byte token IDs

    Note: Empty/invalid addresses are replaced with :: (failed probe sentinel)
    """
    # Handle empty or invalid IP addresses (from failed probes)
    if not ip_str or ip_str.strip() == '':
        ip_str = '::'

    ip_obj = ipaddress.IPv6Address(ip_str)
    ip_bytes = ip_obj.packed  # Returns 16 bytes
    return [byte_to_token(b) for b in ip_bytes]


def encode_ip_address(ip_str: str, ip_version: int) -> List[int]:
    """
    Encode an IP address (IPv4 or IPv6) to token sequence.

    Args:
        ip_str: IP address string
        ip_version: 4 or 6

    Returns:
        List of token IDs: [IPV4_START/IPV6_START, byte_tokens...]
    """
    if ip_version == 4:
        return [IPV4_START] + parse_ipv4(ip_str)
    elif ip_version == 6:
        return [IPV6_START] + parse_ipv6(ip_str)
    else:
        raise ValueError(f"Invalid ip_version: {ip_version}")


# =============================================================================
# Field Encoding Functions
# =============================================================================

def encode_rtt(rtt: float) -> List[int]:
    """
    Encode RTT (round-trip time) as 8 bytes (float64, big-endian).

    Args:
        rtt: RTT value in milliseconds. -1.0 indicates failed probe.

    Returns:
        List of token IDs: [RTT_START, 8 byte tokens]

    Note: Currently uses raw IEEE 754 encoding. See PLAN.md section 8.1
          for alternative encoding strategies (fixed-point, log-scale, clipped).
    """
    # Convert to Python float (handles numpy float32/float64)
    rtt_val = float(rtt)
    # Pack as big-endian float64
    rtt_bytes = struct.pack('>d', rtt_val)
    return [RTT_START] + [byte_to_token(b) for b in rtt_bytes]


def encode_size(size: int) -> List[int]:
    """
    Encode packet size as 2 bytes (uint16, big-endian).

    Args:
        size: Packet size in bytes (0-65535)

    Returns:
        List of token IDs: [SIZE_START, 2 byte tokens]
    """
    # Convert to Python int (handles numpy int64, float32 from Parquet)
    size_val = int(size)
    assert 0 <= size_val <= 65535, f"Size out of range: {size_val}"
    size_bytes = struct.pack('>H', size_val)  # H = unsigned short (2 bytes)
    return [SIZE_START] + [byte_to_token(b) for b in size_bytes]


def encode_error_count(error_count: int) -> List[int]:
    """
    Encode packet error count as 1 byte (uint8).

    Args:
        error_count: Number of packet errors (0-255)

    Returns:
        List of token IDs: [ERROR_COUNT_START, 1 byte token]
    """
    # Convert to Python int (handles numpy int64)
    error_val = int(error_count)
    assert 0 <= error_val <= 255, f"Error count out of range: {error_val}"
    return [ERROR_COUNT_START, byte_to_token(error_val)]


def encode_timestamp(event_time) -> List[int]:
    """
    Encode event timestamp as 8 bytes (int64, seconds since epoch, big-endian).

    Args:
        event_time: Timestamp (pandas Timestamp or datetime-like object)

    Returns:
        List of token IDs: [TIMESTAMP_START, 8 byte tokens]
    """
    # Convert to Unix timestamp (seconds since epoch)
    timestamp_sec = int(event_time.timestamp())
    ts_bytes = struct.pack('>q', timestamp_sec)  # q = long long (8 bytes)
    return [TIMESTAMP_START] + [byte_to_token(b) for b in ts_bytes]


def encode_msm_id(msm_id: int) -> List[int]:
    """
    Encode measurement ID as 8 bytes (int64, big-endian).

    Args:
        msm_id: Measurement identifier

    Returns:
        List of token IDs: [MSM_ID_START, 8 byte tokens]
    """
    # Convert to Python int (handles numpy int64)
    msm_val = int(msm_id)
    msm_bytes = struct.pack('>q', msm_val)  # q = long long (8 bytes)
    return [MSM_ID_START] + [byte_to_token(b) for b in msm_bytes]


# =============================================================================
# Deterministic Shuffling
# =============================================================================

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

def encode_measurement(row: Dict[str, Any]) -> List[int]:
    """
    Encode a single measurement row into a token sequence.

    Args:
        row: Dictionary with keys: msm_id, event_time, src_addr, dst_addr,
             ip_version, rtt, size, packet_error_count

    Returns:
        List of token IDs representing the measurement

    Token sequence structure:
        [MEASUREMENT_START] + shuffled_field_blocks

    Each field block is a list of tokens, e.g.:
        - Source IP: [SRC_IP_START, IPV4_START/IPV6_START, byte_tokens...]
        - Destination IP: [DEST_IP_START, IPV4_START/IPV6_START, byte_tokens...]
        - RTT: [RTT_START, 8 byte tokens]
        - Size: [SIZE_START, 2 byte tokens]
        - Error count: [ERROR_COUNT_START, 1 byte token]
        - Timestamp: [TIMESTAMP_START, 8 byte tokens]
        - Measurement ID: [MSM_ID_START, 8 byte tokens]

    The field blocks are shuffled deterministically using (msm_id, timestamp)
    as the RNG seed to ensure the model learns the joint distribution.
    """
    # Extract fields
    msm_id = row['msm_id']
    event_time = row['event_time']
    src_addr = row['src_addr']
    dst_addr = row['dst_addr']
    ip_version = row['ip_version']
    rtt = row['rtt']
    size = row['size']
    packet_error_count = row['packet_error_count']

    # Handle None/NaN addresses (pandas can return NaN for null values)
    import pandas as pd
    if pd.isna(src_addr):
        src_addr = ''
    if pd.isna(dst_addr):
        dst_addr = ''

    # Encode each field as a block
    src_ip_block = [SRC_IP_START] + encode_ip_address(src_addr, ip_version)
    dst_ip_block = [DEST_IP_START] + encode_ip_address(dst_addr, ip_version)
    rtt_block = encode_rtt(rtt)
    size_block = encode_size(size)
    error_block = encode_error_count(packet_error_count)
    timestamp_block = encode_timestamp(event_time)
    msm_id_block = encode_msm_id(msm_id)

    # Collect all field blocks
    field_blocks = [
        src_ip_block,
        dst_ip_block,
        rtt_block,
        size_block,
        error_block,
        timestamp_block,
        msm_id_block,
    ]

    # Deterministic shuffle using (msm_id, timestamp) as seed
    # Combine msm_id and timestamp seconds for seed
    timestamp_sec = int(event_time.timestamp())
    shuffle_seed = (msm_id * 31 + timestamp_sec) % (2**32)

    shuffled_blocks = shuffle_blocks_deterministic(field_blocks, shuffle_seed)

    # Flatten blocks and prepend MEASUREMENT_START
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
