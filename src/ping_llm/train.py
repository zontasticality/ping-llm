"""
Raw PyTorch training loop for ping-llm.

Features:
- Split optimizer: AdamW for embeddings/unembeddings, Muon for weight matrices
- WSD learning rate schedule (warmup-stable-decay)
- Muon Nesterov momentum warmup
- Linear weight decay to 0
- Masked cross-entropy loss (ignores padding)
- Wandb logging
- Checkpointing with SIGINT handling
"""

import os
import sys
import math
import time
import signal
import contextlib
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn.functional as F

from ping_llm.config import ModelConfig, TrainConfig, parse_args
from ping_llm.model import GPT
from ping_llm.data.loader import create_loader
from ping_llm.muon import Muon


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def wsd_schedule(step: int, total_steps: int, warmup_ratio: float,
                 warmdown_ratio: float) -> float:
    """Warmup-Stable-Decay schedule. Returns multiplier in [0, 1]."""
    warmup_steps = int(total_steps * warmup_ratio)
    warmdown_steps = int(total_steps * warmdown_ratio)
    stable_end = total_steps - warmdown_steps

    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    elif step < stable_end:
        return 1.0
    else:
        progress = (step - stable_end) / max(warmdown_steps, 1)
        return max(0.0, 1.0 - progress)


def muon_momentum_schedule(step: int, warmup_steps: int,
                           start: float, end: float) -> float:
    """Linear warmup of Nesterov momentum."""
    if step >= warmup_steps:
        return end
    return start + (end - start) * step / max(warmup_steps, 1)


def weight_decay_schedule(step: int, total_steps: int, base_wd: float) -> float:
    """Linear decay of weight decay to 0."""
    return base_wd * max(0.0, 1.0 - step / max(total_steps, 1))


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------

def create_optimizers(model: GPT, train_cfg: TrainConfig, model_cfg: ModelConfig):
    """
    Create split optimizer groups following nanochat pattern.

    Returns list of (optimizer, param_group_name, is_muon) tuples.
    """
    embed_params = [model.wte.weight]
    unembed_params = [model.lm_head.weight]
    matrix_params = []

    for name, p in model.named_parameters():
        if name == "wte.weight" or name == "lm_head.weight":
            continue
        if p.ndim == 2:
            matrix_params.append(p)
        # 1D params (e.g. from norms) — no optimizer needed for parameterless RMSNorm

    scale = train_cfg.lr_scale(model_cfg.n_embd)

    # AdamW for embedding
    embed_opt = torch.optim.AdamW(
        embed_params,
        lr=train_cfg.embedding_lr * scale,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.weight_decay,
    )

    # AdamW for unembedding
    unembed_opt = torch.optim.AdamW(
        unembed_params,
        lr=train_cfg.unembedding_lr * scale,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.weight_decay,
    )

    # Muon for weight matrices
    muon_opt = Muon(
        matrix_params,
        lr=train_cfg.matrix_lr * scale,
        momentum=train_cfg.muon_nesterov_start,  # Will be warmed up
    )

    return [
        (embed_opt, "embedding", False),
        (unembed_opt, "unembedding", False),
        (muon_opt, "matrix", True),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    model_cfg, train_cfg = parse_args()

    # Support comma-separated train data paths from CLI
    if isinstance(train_cfg.train_data, str) and ',' in train_cfg.train_data:
        train_cfg.train_data = [p.strip() for p in train_cfg.train_data.split(',')]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    amp_dtype = dtype_map[train_cfg.dtype]
    use_amp = train_cfg.dtype != "float32"

    # Model
    model = GPT(model_cfg).to(device)
    print(f"Model params: {model.num_params:,} ({model.num_params/1e6:.1f}M)")
    print(f"  Embedding: {model.wte.weight.numel():,}")
    print(f"  Non-embedding: {model.num_params_non_embedding:,}")

    # Compile
    if train_cfg.compile and device == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizers
    optimizers = create_optimizers(model, train_cfg, model_cfg)

    # Checkpoint directory
    ckpt_dir = Path(train_cfg.checkpoint_dir) / train_cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_step = 0
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        # Handle compiled model
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model"])
        for (opt, name, _), opt_state in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(opt_state)
        start_step = ckpt["step"]
        print(f"  Resumed at step {start_step}, loss={ckpt.get('loss', '?')}")

    # Wandb
    wandb_run = None
    if train_cfg.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            name=train_cfg.run_name,
            config={
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
            },
            mode=train_cfg.wandb_mode,
            resume="allow",
        )

    # Data loaders
    print("Creating data loaders...")
    train_loader = create_loader(
        arrayrecord_path=train_cfg.train_data,
        batch_size=train_cfg.batch_size,
        crop_size=model_cfg.seq_len,
        shuffle=True,
        shuffle_seed=42,
        num_workers=train_cfg.grain_workers,
        prefetch_buffer_size=train_cfg.grain_prefetch,
        use_multiprocessing=train_cfg.use_multiprocessing,
        ram_budget_mb=train_cfg.grain_ram_budget_mb,
        device=device,
    )
    train_iter = iter(train_loader)

    # SIGINT handling for clean checkpoint save
    interrupted = False
    def sigint_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            print("\nForce quit.")
            sys.exit(1)
        interrupted = True
        print("\nInterrupted. Saving checkpoint...")
    signal.signal(signal.SIGINT, sigint_handler)

    # Training loop
    accum_steps = train_cfg.gradient_accumulation_steps
    print(f"\nStarting training: {train_cfg.total_steps} steps, "
          f"batch_size={train_cfg.batch_size}, seq_len={model_cfg.seq_len}, "
          f"grad_accum={accum_steps}")
    tokens_per_step = train_cfg.batch_size * model_cfg.seq_len * accum_steps
    print(f"Tokens/step: {tokens_per_step:,} "
          f"(effective batch size: {train_cfg.batch_size * accum_steps})")

    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    t0 = time.time()
    running_loss = 0.0
    log_steps = 0

    for step in range(start_step, train_cfg.total_steps):
        if interrupted:
            break

        # --- LR + schedule updates ---
        lr_mult = wsd_schedule(
            step, train_cfg.total_steps,
            train_cfg.warmup_ratio, train_cfg.warmdown_ratio,
        )
        wd = weight_decay_schedule(step, train_cfg.total_steps, train_cfg.weight_decay)
        muon_mom = muon_momentum_schedule(
            step, train_cfg.muon_nesterov_warmup_steps,
            train_cfg.muon_nesterov_start, train_cfg.muon_momentum,
        )

        for opt, name, is_muon in optimizers:
            for pg in opt.param_groups:
                if is_muon:
                    base_lr = train_cfg.matrix_lr * train_cfg.lr_scale(model_cfg.n_embd)
                    pg["lr"] = base_lr * lr_mult
                    pg["momentum"] = muon_mom
                elif name == "embedding":
                    base_lr = train_cfg.embedding_lr * train_cfg.lr_scale(model_cfg.n_embd)
                    pg["lr"] = base_lr * lr_mult
                    pg["weight_decay"] = wd
                else:  # unembedding
                    base_lr = train_cfg.unembedding_lr * train_cfg.lr_scale(model_cfg.n_embd)
                    pg["lr"] = base_lr * lr_mult
                    pg["weight_decay"] = wd

        # --- Gradient accumulation loop ---
        for opt, _, _ in optimizers:
            opt.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for _micro in range(accum_steps):
            batch = next(train_iter)
            inputs = batch["inputs"]
            targets = batch["targets"]
            targets_mask = batch["targets_segmentation"]

            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(inputs, targets, targets_mask)
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        # Gradient clipping
        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizers[0][0])
            scaler.unscale_(optimizers[1][0])
            scaler.unscale_(optimizers[2][0])
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip)

        for opt, _, _ in optimizers:
            scaler.step(opt)
        scaler.update()

        # --- Logging ---
        loss_val = accumulated_loss
        running_loss += loss_val
        log_steps += 1

        if (step + 1) % train_cfg.log_interval == 0:
            avg_loss = running_loss / log_steps
            elapsed = time.time() - t0
            tokens_sec = (log_steps * tokens_per_step) / elapsed
            eta_sec = (train_cfg.total_steps - step - 1) / (log_steps / elapsed)

            print(f"step {step+1}/{train_cfg.total_steps} | "
                  f"loss {avg_loss:.4f} | "
                  f"lr {lr_mult:.4f} | "
                  f"tok/s {tokens_sec:,.0f} | "
                  f"eta {eta_sec/3600:.1f}h")

            if wandb_run:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr_mult": lr_mult,
                    "train/tokens_per_sec": tokens_sec,
                    "train/weight_decay": wd,
                    "train/muon_momentum": muon_mom,
                    "train/step": step + 1,
                }, step=step + 1)

            running_loss = 0.0
            log_steps = 0
            t0 = time.time()

        # --- Eval ---
        if (step + 1) % train_cfg.eval_interval == 0:
            eval_loss = run_eval(
                model, train_cfg, model_cfg, device, amp_dtype, use_amp
            )
            print(f"  eval loss: {eval_loss:.4f}")
            if wandb_run:
                import wandb
                wandb.log({"eval/loss": eval_loss}, step=step + 1)

        # --- Checkpoint ---
        if (step + 1) % train_cfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizers, step + 1, model_cfg, train_cfg,
                            loss_val, ckpt_dir)

    # Final checkpoint
    save_checkpoint(model, optimizers, step + 1, model_cfg, train_cfg,
                    loss_val, ckpt_dir)
    print(f"\nTraining complete. Final loss: {loss_val:.4f}")

    if wandb_run:
        import wandb
        wandb.finish()


def run_eval(model, train_cfg: TrainConfig, model_cfg: ModelConfig,
             device: str, amp_dtype, use_amp: bool) -> float:
    """Run evaluation and return average loss."""
    model.eval()
    eval_loader = create_loader(
        arrayrecord_path=train_cfg.eval_data,
        batch_size=train_cfg.batch_size,
        crop_size=model_cfg.seq_len,
        shuffle=False,
        shuffle_seed=0,
        num_workers=4,
        prefetch_buffer_size=4,
        use_multiprocessing=False,
        device=device,
    )
    total_loss = 0.0
    count = 0
    eval_iter = iter(eval_loader)
    with torch.no_grad():
        for _ in range(train_cfg.eval_steps):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            inputs = batch["inputs"]
            targets = batch["targets"]
            targets_mask = batch["targets_segmentation"]
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(inputs, targets, targets_mask)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


def save_checkpoint(model, optimizers, step, model_cfg, train_cfg, loss, ckpt_dir):
    """Save model checkpoint."""
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizers": [opt.state_dict() for opt, _, _ in optimizers],
        "step": step,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "loss": loss,
    }
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(ckpt, path)
    # Symlink latest
    latest = ckpt_dir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    print(f"  checkpoint saved: {path}")


if __name__ == "__main__":
    train()
