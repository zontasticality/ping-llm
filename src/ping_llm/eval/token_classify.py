"""
State machine to classify each token position by semantic type.

Given a token sequence, returns a parallel array of type labels:
  role, src_ip_byte, dst_ip_byte, rtt_byte, timestamp_byte
"""

from ping_llm.data.tokenization import (
    MEASUREMENT_START, SRC_IPV4, SRC_IPV6, DST_IPV4, DST_IPV6,
    TIMESTAMP_ABS, TIMESTAMP_DELTA1, TIMESTAMP_DELTA4,
    RTT_START, FAILED, THROUGHPUT_START,
)

# How many data bytes follow each role token
ROLE_BYTE_COUNTS = {
    MEASUREMENT_START: 0,
    SRC_IPV4: 4,
    SRC_IPV6: 16,
    DST_IPV4: 4,
    DST_IPV6: 16,
    TIMESTAMP_ABS: 8,
    TIMESTAMP_DELTA1: 1,
    TIMESTAMP_DELTA4: 4,
    RTT_START: 2,
    FAILED: 0,
    THROUGHPUT_START: 0,  # reserved
}

# Map role tokens to the label for their following bytes
ROLE_TO_BYTE_LABEL = {
    SRC_IPV4: "src_ip_byte",
    SRC_IPV6: "src_ip_byte",
    DST_IPV4: "dst_ip_byte",
    DST_IPV6: "dst_ip_byte",
    TIMESTAMP_ABS: "timestamp_byte",
    TIMESTAMP_DELTA1: "timestamp_byte",
    TIMESTAMP_DELTA4: "timestamp_byte",
    RTT_START: "rtt_byte",
}

ALL_ROLE_TOKENS = set(ROLE_BYTE_COUNTS.keys())


def classify_tokens(tokens):
    """
    Classify each token position in a sequence.

    Args:
        tokens: list/array of token IDs

    Returns:
        list of str labels, same length as tokens.
        Labels: "role", "src_ip_byte", "dst_ip_byte", "rtt_byte", "timestamp_byte"
    """
    n = len(tokens)
    labels = ["unknown"] * n
    i = 0
    while i < n:
        t = int(tokens[i])
        if t in ALL_ROLE_TOKENS:
            labels[i] = "role"
            byte_count = ROLE_BYTE_COUNTS[t]
            byte_label = ROLE_TO_BYTE_LABEL.get(t)
            if byte_label and byte_count > 0:
                for j in range(1, byte_count + 1):
                    if i + j < n:
                        labels[i + j] = byte_label
                i += 1 + byte_count
            else:
                i += 1
        else:
            # Byte token outside expected structure — mark unknown
            labels[i] = "unknown"
            i += 1
    return labels
