"""Configuration dataclasses for ping-llm training."""

from dataclasses import dataclass, field
import argparse
import math


@dataclass
class ModelConfig:
    vocab_size: int = 267
    n_layer: int = 20
    n_embd: int = 640
    n_head: int = 10
    head_dim: int = 64
    seq_len: int = 1024
    rope_theta: float = 10000.0
    softcap: float = 15.0
    activation: str = "relu_sq"  # relu_sq, gelu, silu

    # Nanochat architecture flags (autoresearch-validated winners enabled by default)
    use_fused_qkv: bool = True      # fused Q/K/V projection (1 matmul vs 3) — kept: -0.15 loss
    use_zero_init: bool = False     # zero init for output projections + lm_head — discarded at 5min
    use_resid_scalars: bool = False # per-layer residual scalars + x0 injection — discarded at 5min
    use_value_embeds: bool = False  # ResFormer-style value embeddings — discarded at 5min
    embed_init_scale: float = 0     # embedding init std (0=auto sqrt(3/n_embd)) — kept: -0.49 loss

    @property
    def mlp_dim(self) -> int:
        return 4 * self.n_embd

    @property
    def num_params(self) -> int:
        """Rough parameter count."""
        emb = self.vocab_size * self.n_embd
        attn_per_layer = 4 * self.n_embd * (self.n_head * self.head_dim)
        mlp_per_layer = 2 * self.n_embd * self.mlp_dim
        layers = self.n_layer * (attn_per_layer + mlp_per_layer)
        head = self.vocab_size * self.n_embd
        return emb + layers + head


@dataclass
class TrainConfig:
    # Data
    train_data: str | list[str] = "data/probe_rows/train.arrayrecord"
    eval_data: str = "data/probe_rows/test.arrayrecord"
    grain_workers: int = 6
    grain_prefetch: int = 4
    grain_ram_budget_mb: int = 32768
    use_multiprocessing: bool = True

    # Training
    batch_size: int = 256
    total_steps: int = 14000
    warmup_ratio: float = 0.01
    warmdown_ratio: float = 0.50
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    compile: bool = True
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    max_time_seconds: int = 0  # 0 = disabled, >0 = stop after this many seconds

    # Optimizer: per-group LRs (scaled by (n_embd/768)**-0.5)
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    weight_decay: float = 0.1

    # Muon
    muon_momentum: float = 0.95
    muon_nesterov_warmup_steps: int = 300
    muon_nesterov_start: float = 0.85

    # AdamW
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95

    # Checkpointing
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_interval: int = 200
    run_name: str = "default"

    # Logging
    wandb_project: str = "ping-llm"
    wandb_mode: str = "online"  # online, offline, disabled
    log_interval: int = 10
    eval_interval: int = 100
    eval_steps: int = 5

    def lr_scale(self, n_embd: int) -> float:
        return (n_embd / 768) ** -0.5

    def scaled_embedding_lr(self, n_embd: int) -> float:
        return self.embedding_lr * self.lr_scale(n_embd)

    def scaled_unembedding_lr(self, n_embd: int) -> float:
        return self.unembedding_lr * self.lr_scale(n_embd)

    def scaled_matrix_lr(self, n_embd: int) -> float:
        return self.matrix_lr * self.lr_scale(n_embd)


def parse_args() -> tuple[ModelConfig, TrainConfig]:
    """Parse CLI args into ModelConfig + TrainConfig with override support."""
    parser = argparse.ArgumentParser(description="ping-llm training")

    # Model args
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=None)
    parser.add_argument("--softcap", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--use-fused-qkv", action="store_true")
    parser.add_argument("--use-zero-init", action="store_true")
    parser.add_argument("--use-resid-scalars", action="store_true")
    parser.add_argument("--use-value-embeds", action="store_true")
    parser.add_argument("--embed-init-scale", type=float, default=None)

    # Train args
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--eval-data", type=str, default=None)
    parser.add_argument("--grain-workers", type=int, default=None)
    parser.add_argument("--grain-prefetch", type=int, default=None)
    parser.add_argument("--grain-ram-budget-mb", type=int, default=None)
    parser.add_argument("--no-multiprocessing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--max-time-seconds", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--warmdown-ratio", type=float, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--embedding-lr", type=float, default=None)
    parser.add_argument("--unembedding-lr", type=float, default=None)
    parser.add_argument("--matrix-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--muon-momentum", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)

    args = parser.parse_args()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Apply overrides
    model_fields = {
        "vocab_size", "n_layer", "n_embd", "n_head", "head_dim",
        "seq_len", "rope_theta", "softcap", "activation",
        "embed_init_scale",
    }
    for f in model_fields:
        cli_name = f.replace("_", "-")
        val = getattr(args, f.replace("-", "_"), None)
        if val is not None:
            setattr(model_cfg, f, val)

    train_fields = {
        "train_data", "eval_data", "grain_workers", "grain_prefetch",
        "grain_ram_budget_mb", "batch_size", "total_steps", "max_time_seconds",
        "warmup_ratio",
        "warmdown_ratio", "dtype", "grad_clip", "gradient_accumulation_steps",
        "embedding_lr",
        "unembedding_lr", "matrix_lr", "weight_decay", "muon_momentum",
        "checkpoint_dir", "checkpoint_interval", "run_name", "wandb_project",
        "wandb_mode", "log_interval", "eval_interval", "eval_steps",
    }
    for f in train_fields:
        val = getattr(args, f.replace("-", "_"), None)
        if val is not None:
            setattr(train_cfg, f, val)

    if args.no_multiprocessing:
        train_cfg.use_multiprocessing = False
    if args.no_compile:
        train_cfg.compile = False
    if args.use_fused_qkv:
        model_cfg.use_fused_qkv = True
    if args.use_zero_init:
        model_cfg.use_zero_init = True
    if args.use_resid_scalars:
        model_cfg.use_resid_scalars = True
    if args.use_value_embeds:
        model_cfg.use_value_embeds = True

    return model_cfg, train_cfg
