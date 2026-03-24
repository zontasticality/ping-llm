"""
GPT model for ping-llm, following nanochat architecture.

Architecture:
- RMSNorm (no learnable params)
- RoPE (precomputed cos/sin buffers)
- Q/K/V projections with QK norm (fused QKV optional)
- ReLU^2 activation (configurable)
- Logit softcap at 15
- Optional: zero init, residual scalars, value embeddings
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
        x1, x2 = x.chunk(2, dim=-1)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx
        self.use_fused_qkv = config.use_fused_qkv
        inner_dim = config.n_head * config.head_dim

        if self.use_fused_qkv:
            self.qkv_proj = nn.Linear(config.n_embd, 3 * inner_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
            self.k_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
            self.v_proj = nn.Linear(config.n_embd, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.n_embd, bias=False)

        # QK norm (per-head RMS norm on q and k)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)

        # Output projection init
        if config.use_zero_init:
            self.o_proj.weight.data.zero_()
        else:
            self.o_proj.weight.data.mul_(1.0 / math.sqrt(2 * config.n_layer))

        # Value embeddings on alternating layers
        self.use_value_embeds = config.use_value_embeds and (layer_idx % 2 == 1)
        if self.use_value_embeds:
            self.ve_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding,
                value_embed: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        if self.use_fused_qkv:
            qkv = self.qkv_proj(x)
            inner_dim = self.n_head * self.head_dim
            q, k, v = qkv.split(inner_dim, dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = rope(q)
        k = rope(k)

        # Value embeddings
        if self.use_value_embeds and value_embed is not None:
            v = v + self.ve_gate * value_embed[:, :T, :]

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

        # Down projection init
        if config.use_zero_init:
            self.down_proj.weight.data.zero_()
        else:
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

        # Per-layer residual scalars + x0 injection
        self.use_resid_scalars = config.use_resid_scalars
        if self.use_resid_scalars:
            self.resid_lambda = nn.Parameter(torch.tensor(1.05))
            self.x0_lambda = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding,
                x0: torch.Tensor | None = None,
                value_embed: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_resid_scalars and x0 is not None:
            x = self.resid_lambda * x + self.attn(self.attn_norm(x), rope, value_embed)
            x = self.resid_lambda * x + self.mlp(self.mlp_norm(x)) + self.x0_lambda * x0
        else:
            x = x + self.attn(self.attn_norm(x), rope, value_embed)
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

        # Value embeddings (alternating layers)
        if config.use_value_embeds:
            n_ve_layers = config.n_layer // 2
            self.value_embeds = nn.Parameter(
                0.01 * torch.randn(n_ve_layers, config.n_head, config.seq_len, config.head_dim)
            )
        else:
            self.value_embeds = None

        # Initialize
        self.apply(self._init_weights)
        if config.use_zero_init:
            self.lm_head.weight.data.zero_()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = self.config.embed_init_scale
            if std == 0:  # auto: sqrt(3/n_embd)
                std = math.sqrt(3.0 / self.config.n_embd)
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        targets_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        x = self.wte(idx)  # [B, T, C]

        x0 = x if self.config.use_resid_scalars else None

        ve_idx = 0
        for block in self.blocks:
            ve = None
            if self.value_embeds is not None and block.attn.use_value_embeds:
                ve = self.value_embeds[ve_idx]
                ve_idx += 1
            x = block(x, self.rope, x0=x0, value_embed=ve)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, V]

        # Logit softcap
        logits = self.config.softcap * torch.tanh(logits / self.config.softcap)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            if targets_mask is not None:
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
