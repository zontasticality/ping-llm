"""
Inference utilities for ping-llm PyTorch model.

Provides:
- load_model: Load model from checkpoint
- get_logits: Get logits for full token sequence
- generate: Autoregressive token generation
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from ping_llm.config import ModelConfig
from ping_llm.model import GPT


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[GPT, ModelConfig]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Target device
        dtype: Model dtype (float32 for eval is fine)

    Returns:
        (model, model_config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(model_cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


@torch.no_grad()
def get_logits(
    model: GPT,
    tokens: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Get logits for a full token sequence.

    Args:
        model: Loaded GPT model
        tokens: List of token IDs
        device: Device string

    Returns:
        Logits tensor of shape [len(tokens), vocab_size]
    """
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    logits, _ = model(idx)
    return logits[0]  # [T, V]


@torch.no_grad()
def get_log_probs(
    model: GPT,
    tokens: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Get per-token log probabilities for next-token prediction.

    For tokens [t0, t1, t2, ...], returns log P(t_{i+1} | t_0..t_i)
    for i in [0, len-2]. Output has length len(tokens)-1.

    Args:
        model: Loaded GPT model
        tokens: List of token IDs
        device: Device string

    Returns:
        Log probabilities tensor of shape [len(tokens)-1]
    """
    logits = get_logits(model, tokens, device)  # [T, V]
    # logits[i] predicts token[i+1]
    log_probs = F.log_softmax(logits[:-1], dim=-1)  # [T-1, V]
    target_tokens = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    return log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)  # [T-1]


@torch.no_grad()
def generate(
    model: GPT,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    stop_token: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """
    Autoregressive token generation.

    Args:
        model: Loaded GPT model
        prompt_tokens: Initial token IDs
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: If set, only sample from top-k logits
        stop_token: If set, stop generation when this token is produced
        device: Device string

    Returns:
        List of all tokens (prompt + generated)
    """
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    seq_len = model.config.seq_len

    for _ in range(max_new_tokens):
        # Crop to max sequence length
        idx_cond = idx[:, -seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # [1, V]

        if temperature == 0:
            # Greedy
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if stop_token is not None and next_token.item() == stop_token:
            break

    return idx[0].tolist()
