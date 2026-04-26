"""
RTT-aware Wasserstein loss for ping-llm.

Adds ordinal-aware Wasserstein-1 (Earth Mover's Distance) loss on RTT byte
tokens alongside standard cross-entropy. The key insight: an exponent error
in RTT byte 1 (EEEEE_MMM) means 2x the actual RTT, while a low-mantissa
error in byte 2 is ~0.05%. Standard CE treats both equally wrong.

Loss = CE + lambda1 * WAS(rtt_byte1, log_scale) + lambda2 * WAS(rtt_byte2, linear)
"""

import math
import torch
import torch.nn.functional as F

from ping_llm.data.tokenization import RTT_START, BYTE_TOKEN_OFFSET


def _build_rtt_byte1_sort_order():
    """
    Build sort order for RTT byte 1 values by log2(representative RTT).

    Byte 1 = EEEEE_MMM: the 256 possible values are NOT linearly ordered
    in RTT-space. We sort them by their actual RTT contribution so that
    the CDF-based Wasserstein respects the true metric.
    """
    log_values = torch.zeros(256)
    for b in range(256):
        exp = b >> 3
        hmant = b & 0x07
        mantissa = hmant * 256 + 128  # midpoint of byte2 range
        if mantissa == 0:
            mantissa = 1
        rtt_us = mantissa * (2.0 ** exp)
        log_values[b] = math.log2(max(rtt_us, 1.0))

    sort_order = torch.argsort(log_values)
    inv_order = torch.zeros(256, dtype=torch.long)
    inv_order[sort_order] = torch.arange(256)
    return sort_order, inv_order


_BYTE1_SORT, _BYTE1_INV = _build_rtt_byte1_sort_order()
_BYTE2_SORT = torch.arange(256)
_BYTE2_INV = torch.arange(256, dtype=torch.long)


def compute_rtt_wasserstein(logits, targets, targets_mask, inputs,
                            lambda1=0.5, lambda2=0.1):
    """
    Compute Wasserstein-1 loss on RTT byte positions.

    Args:
        logits: [B, T, V] model output logits
        targets: [B, T] target token IDs
        targets_mask: [B, T] segmentation mask (1=valid, 0=padding)
        inputs: [B, T] input token IDs (used to identify RTT positions)
        lambda1: weight for byte 1 Wasserstein (exponent + high mantissa)
        lambda2: weight for byte 2 Wasserstein (low mantissa)

    Returns:
        scalar loss = lambda1 * was_byte1 + lambda2 * was_byte2
    """
    if lambda1 == 0 and lambda2 == 0:
        return torch.tensor(0.0, device=logits.device)

    B, T = inputs.shape
    device = logits.device

    # Classify RTT byte positions from inputs tensor (fully vectorized)
    rtt_start_mask = (inputs == RTT_START)  # [B, T]
    valid = targets_mask.bool() if targets_mask is not None else torch.ones(B, T, dtype=torch.bool, device=device)

    # byte1 positions: where inputs[i] == RTT_START -> targets[i] is byte 1
    byte1_mask = rtt_start_mask & valid
    # byte2 positions: where inputs[i-1] == RTT_START -> targets[i] is byte 2
    byte2_mask = torch.zeros_like(valid)
    if T > 1:
        byte2_mask[:, 1:] = rtt_start_mask[:, :-1] & valid[:, 1:]

    total_loss = torch.tensor(0.0, device=device)

    if lambda1 > 0 and byte1_mask.any():
        total_loss = total_loss + lambda1 * _wasserstein_at_positions(
            logits, targets, byte1_mask,
            _BYTE1_SORT.to(device), _BYTE1_INV.to(device),
        )

    if lambda2 > 0 and byte2_mask.any():
        total_loss = total_loss + lambda2 * _wasserstein_at_positions(
            logits, targets, byte2_mask,
            _BYTE2_SORT.to(device), _BYTE2_INV.to(device),
        )

    return total_loss


def _wasserstein_at_positions(logits, targets, mask, sort_order, inv_order):
    """
    CDF-based Wasserstein-1 on byte tokens at masked positions.

    Only considers the 256 byte token logits (indices BYTE_TOKEN_OFFSET:BYTE_TOKEN_OFFSET+256),
    reordered by sort_order for proper ordinal comparison.
    """
    sel_logits = logits[mask][:, BYTE_TOKEN_OFFSET:BYTE_TOKEN_OFFSET + 256]  # [N, 256]
    sel_targets = targets[mask] - BYTE_TOKEN_OFFSET  # byte values 0-255

    # Reorder by semantic value
    sel_logits_sorted = sel_logits[:, sort_order]
    pred_cdf = torch.cumsum(F.softmax(sel_logits_sorted, dim=-1), dim=-1)

    # Target CDF (step function at the sorted position of the true byte)
    target_sorted_pos = inv_order[sel_targets.long()]
    target_onehot = F.one_hot(target_sorted_pos, num_classes=256).float()
    true_cdf = torch.cumsum(target_onehot, dim=-1)

    return torch.sum(torch.abs(pred_cdf - true_cdf), dim=-1).mean()
