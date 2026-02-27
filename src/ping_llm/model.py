"""
GPT model for ping-llm, following nanochat architecture.

Architecture:
- RMSNorm (no learnable params)
- RoPE (precomputed cos/sin buffers)
- Separate Q/K/V projections with QK norm
- ReLU^2 activation (configurable)
- Logit softcap at 15
- Small init for output projections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ping_llm.config import ModelConfig


class RMSNorm(nn.Module):
    """RMS normalization without learnable parameters."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,))


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin buffers."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        half_dim = head_dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # [seq_len, half_dim]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embeddings. x: [B, n_head, T, head_dim]"""
        T = x.shape[2]
        cos = self.cos[offset : offset + T]  # [T, half_dim]
        sin = self.sin[offset : offset + T]
        # Split into pairs for rotation
        x1, x2 = x.chunk(2, dim=-1)
        # Apply rotation
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        inner_dim = config.n_head * config.head_dim

        self.q_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.n_embd, bias=False)

        # QK norm (per-head RMS norm on q and k)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)

        # Small init for output projection
        self.o_proj.weight.data.mul_(1.0 / math.sqrt(2 * config.n_layer))

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = rope(q)
        k = rope(k)

        # Flash attention (causal)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, config.mlp_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_dim, config.n_embd, bias=False)
        self.activation = config.activation

        # Small init for down projection
        self.down_proj.weight.data.mul_(1.0 / math.sqrt(2 * config.n_layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(x)
        if self.activation == "relu_sq":
            h = F.relu(h).square()
        elif self.activation == "gelu":
            h = F.gelu(h)
        elif self.activation == "silu":
            h = F.silu(h)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return self.down_proj(h)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.n_layer)]
        )
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Shared rope for all layers
        self.rope = RotaryEmbedding(config.head_dim, config.seq_len, config.rope_theta)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        targets_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: input token ids [B, T]
            targets: target token ids [B, T] (optional, for loss computation)
            targets_mask: segmentation mask [B, T] for ignoring padding in loss

        Returns:
            (logits [B, T, V], loss or None)
        """
        B, T = idx.shape
        x = self.wte(idx)  # [B, T, C]

        for block in self.blocks:
            x = block(x, self.rope)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, V]

        # Logit softcap
        logits = self.config.softcap * torch.tanh(logits / self.config.softcap)

        loss = None
        if targets is not None:
            # Flatten for cross entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            if targets_mask is not None:
                # Masked loss: ignore padding tokens
                mask_flat = targets_mask.view(-1).float()
                per_token_loss = F.cross_entropy(
                    logits_flat, targets_flat, reduction="none"
                )
                loss = (per_token_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_params_non_embedding(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.wte.weight.numel()
