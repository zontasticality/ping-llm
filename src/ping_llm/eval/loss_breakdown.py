"""
Test 1: Loss breakdown by token type.

Decomposes eval cross-entropy into per-token-type losses:
  role, src_ip_byte, dst_ip_byte, rtt_byte, timestamp_byte
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from ping_llm.eval.token_classify import classify_tokens


def eval_loss_breakdown(model, sequences, device="cpu", max_sequences=200):
    """
    Compute per-token-type cross-entropy on evaluation sequences.

    Args:
        model: Loaded GPT model (eval mode)
        sequences: list of np.ndarray token sequences (variable length, non-padded)
        device: torch device string
        max_sequences: cap on sequences to process

    Returns:
        dict with per-type stats: {type: {count, total_ce, mean_ce, accuracy}}
    """
    buckets = defaultdict(lambda: {"count": 0, "total_ce": 0.0, "correct": 0})

    for seq_idx, tokens in enumerate(sequences[:max_sequences]):
        if len(tokens) < 2:
            continue

        token_list = [int(t) for t in tokens]
        labels = classify_tokens(token_list)

        # Forward pass
        idx = torch.tensor([token_list], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx)  # [1, T, V]
        logits = logits[0]  # [T, V]

        # Per-position CE: logits[i] predicts token[i+1]
        # So the label for position i in the loss is labels[i+1] (the type of the target token)
        log_probs = F.log_softmax(logits[:-1], dim=-1)  # [T-1, V]
        targets = torch.tensor(token_list[1:], dtype=torch.long, device=device)
        per_token_ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [T-1]
        predictions = logits[:-1].argmax(dim=-1)  # [T-1]
        correct = (predictions == targets)

        ce_np = per_token_ce.cpu().numpy()
        correct_np = correct.cpu().numpy()

        # labels[i+1] is the type of the token being predicted at position i
        for i in range(len(token_list) - 1):
            token_type = labels[i + 1]
            buckets[token_type]["count"] += 1
            buckets[token_type]["total_ce"] += float(ce_np[i])
            buckets[token_type]["correct"] += int(correct_np[i])

        if (seq_idx + 1) % 50 == 0:
            print(f"  Processed {seq_idx + 1}/{min(len(sequences), max_sequences)} sequences")

    # Compute means
    results = {}
    total_count = 0
    total_ce = 0.0
    total_correct = 0
    for token_type, bucket in sorted(buckets.items()):
        count = bucket["count"]
        if count == 0:
            continue
        mean_ce = bucket["total_ce"] / count
        accuracy = bucket["correct"] / count
        results[token_type] = {
            "count": count,
            "mean_ce": round(mean_ce, 4),
            "accuracy": round(accuracy, 4),
        }
        total_count += count
        total_ce += bucket["total_ce"]
        total_correct += bucket["correct"]

    if total_count > 0:
        results["overall"] = {
            "count": total_count,
            "mean_ce": round(total_ce / total_count, 4),
            "accuracy": round(total_correct / total_count, 4),
        }

    return results


def print_loss_breakdown(results):
    """Pretty-print loss breakdown results."""
    print("\n" + "=" * 65)
    print("LOSS BREAKDOWN BY TOKEN TYPE")
    print("=" * 65)
    print(f"{'Token Type':<18} {'Count':>8} {'CE Loss':>10} {'Top-1 Acc':>10}")
    print("-" * 65)

    display_order = ["role", "src_ip_byte", "dst_ip_byte", "rtt_byte", "timestamp_byte", "unknown", "overall"]
    for token_type in display_order:
        if token_type not in results:
            continue
        r = results[token_type]
        prefix = ">> " if token_type == "overall" else "   "
        print(f"{prefix}{token_type:<15} {r['count']:>8,} {r['mean_ce']:>10.4f} {r['accuracy']:>9.1%}")
    print("=" * 65)
