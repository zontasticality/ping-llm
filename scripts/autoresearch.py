"""Autoresearch-style experiment runner for ping-llm.

Runs a series of architecture experiments, testing one change at a time.
Keeps improvements, reverts regressions. Inspired by karpathy/autoresearch.

Usage:
  modal run scripts/autoresearch.py
  modal run scripts/autoresearch.py --time-budget 180  # 3 min per experiment
"""

import os
import subprocess
import time
import json

import modal

APP_NAME = "ping-llm-autoresearch"
WORKDIR = "/workspace"

DATA_VOLUME = "ping-llm-data"
OUTPUTS_VOLUME = "ping-llm"
DATA_MOUNT = "/mnt/data"
OUTPUTS_MOUNT = "/mnt/outputs"

TRAIN_SHARDS = [
    f"{DATA_MOUNT}/train_shards/train.arrayrecord-0000{i}-of-00004"
    for i in range(4)
]
EVAL_PATH = f"{DATA_MOUNT}/test.arrayrecord"

IGNORE_PATTERNS = [
    ".git", ".venv", ".train_venv", ".slurm_venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", "outputs", "logs", "data",
    "local_datasets", "archive", "*.parquet", "*.arrayrecord",
    ".DS_Store", ".claude", "docs",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("git", "build-essential", "tzdata")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system "
        "torch --index-url https://download.pytorch.org/whl/cu124 && "
        "uv pip install --system "
        "pyarrow numpy grain array_record wandb"
    )
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME)
outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Experiment definitions: each is (name, description, config_overrides)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    (
        "fused_qkv",
        "Fused Q/K/V projection (1 matmul instead of 3)",
        {"use_fused_qkv": True},
    ),
    (
        "zero_init",
        "Zero init for output projections + lm_head",
        {"use_zero_init": True},
    ),
    (
        "embed_init",
        "Embedding init std = sqrt(3/n_embd) instead of 0.02",
        {"embed_init_scale": 0},  # 0 = auto sqrt(3/n_embd)
    ),
    (
        "resid_scalars",
        "Per-layer residual scalars + x0 injection",
        {"use_resid_scalars": True},
    ),
    (
        "value_embeds",
        "ResFormer-style value embeddings on alternating layers",
        {"use_value_embeds": True},
    ),
]


def run_experiment(
    name: str,
    model_flags: dict,
    time_budget: int,
    batch_size: int,
) -> dict:
    """Run a single training experiment and return metrics."""
    train_data = ",".join(TRAIN_SHARDS)

    cmd = [
        "python", "-m", "ping_llm.train",
        "--run-name", f"auto-{name}",
        "--total-steps", "100000",  # large number; max-time-seconds is the real limit
        "--max-time-seconds", str(time_budget),
        "--batch-size", str(batch_size),
        "--wandb-mode", "disabled",
        "--train-data", train_data,
        "--eval-data", EVAL_PATH,
        "--checkpoint-dir", f"/tmp/autoresearch_ckpt",  # temp dir, no resume from stale checkpoints
        "--log-interval", "50",
        "--eval-interval", "999999",  # skip periodic eval, we do final eval
        "--checkpoint-interval", "999999",  # skip checkpoints
    ]

    # Apply model flags
    for flag, value in model_flags.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{flag.replace('_', '-')}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{flag.replace('_', '-')}", str(value)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKDIR}/src"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONUNBUFFERED"] = "1"
    # Compile cache
    cache_dir = f"{OUTPUTS_MOUNT}/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"Flags: {model_flags}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    process = subprocess.Popen(
        cmd, cwd=WORKDIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1,
    )

    # Collect output, kill after time budget + grace period
    output_lines = []
    deadline = time.time() + time_budget + 360  # extra 6 min for compile + data warmup
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
        if time.time() > deadline:
            print(f"\n  TIMEOUT after {time_budget+120}s, killing...")
            process.kill()
            break

    process.wait()

    # Parse metrics from output
    result = {
        "name": name,
        "flags": model_flags,
        "exit_code": process.returncode,
        "train_loss": None,
        "eval_loss": None,
        "tok_s": None,
        "steps": None,
    }

    for line in output_lines:
        if "Training complete. Final loss:" in line:
            try:
                result["train_loss"] = float(line.strip().split(":")[-1])
            except ValueError:
                pass
        if "eval loss:" in line:
            try:
                result["eval_loss"] = float(line.strip().split(":")[-1])
            except ValueError:
                pass
        if "tok/s" in line and "step" in line:
            try:
                for part in line.split("|"):
                    if "tok/s" in part:
                        result["tok_s"] = float(part.split("tok/s")[0].strip().split()[-1].replace(",", ""))
                    if "step" in part and "/" in part:
                        step_str = part.strip().split()[1].split("/")[0]
                        result["steps"] = int(step_str)
            except (ValueError, IndexError):
                pass

    return result


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    volumes={DATA_MOUNT: data_vol, OUTPUTS_MOUNT: outputs_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 4,  # 4 hours max
)
def autoresearch(time_budget: int = 300, batch_size: int = 32):
    """Run the autoresearch experiment loop."""
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Inject max_time_seconds into all commands via env
    os.environ["PING_LLM_MAX_TIME"] = str(time_budget)

    results = []
    active_flags = {}  # Flags that have been kept

    # --- Step 1: Baseline ---
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE (no changes)")
    print("=" * 60)

    baseline = run_experiment("baseline", active_flags, time_budget, batch_size)
    results.append(baseline)

    if baseline["train_loss"] is None:
        print("\nBASELINE FAILED - aborting")
        return results

    best_loss = baseline["train_loss"]
    print(f"\n>>> BASELINE: train_loss={best_loss:.4f}, "
          f"tok/s={baseline.get('tok_s', '?')}, steps={baseline.get('steps', '?')}")

    # --- Step 2: Test each experiment ---
    for exp_name, exp_desc, exp_flags in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Testing '{exp_name}' — {exp_desc}")
        print(f"Active flags so far: {active_flags}")
        print(f"New flag: {exp_flags}")
        print(f"{'='*60}")

        # Merge with currently active flags
        test_flags = {**active_flags, **exp_flags}

        result = run_experiment(exp_name, test_flags, time_budget, batch_size)
        results.append(result)

        if result["train_loss"] is None:
            print(f"\n>>> {exp_name}: CRASHED (exit={result['exit_code']}) — DISCARDED")
            continue

        improved = result["train_loss"] < best_loss
        delta = result["train_loss"] - best_loss

        if improved:
            print(f"\n>>> {exp_name}: train_loss={result['train_loss']:.4f} "
                  f"(delta={delta:+.4f}) — KEEPING")
            active_flags.update(exp_flags)
            best_loss = result["train_loss"]
        else:
            print(f"\n>>> {exp_name}: train_loss={result['train_loss']:.4f} "
                  f"(delta={delta:+.4f}) — DISCARDED")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("AUTORESEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline train_loss: {baseline['train_loss']:.4f}")
    print(f"Best train_loss:     {best_loss:.4f}")
    print(f"Active flags:        {active_flags}")
    print(f"\nAll results:")
    print(f"{'Name':<20s} {'Loss':>8s} {'tok/s':>8s} {'Steps':>6s} {'Status':<10s}")
    print("-" * 60)
    for r in results:
        loss_str = f"{r['train_loss']:.4f}" if r['train_loss'] else "CRASH"
        tok_str = f"{r['tok_s']:,.0f}" if r['tok_s'] else "?"
        step_str = str(r.get('steps', '?'))
        status = "KEPT" if all(k in active_flags and active_flags[k] == v
                               for k, v in r['flags'].items()) and r['flags'] else "baseline" if not r['flags'] else "discarded"
        print(f"{r['name']:<20s} {loss_str:>8s} {tok_str:>8s} {step_str:>6s} {status:<10s}")

    # Save results to volume
    results_path = f"{OUTPUTS_MOUNT}/autoresearch_results.json"
    with open(results_path, "w") as f:
        json.dump({"results": results, "active_flags": active_flags, "best_loss": best_loss}, f, indent=2)
    outputs_vol.commit()
    print(f"\nResults saved to {results_path}")

    return results


@app.local_entrypoint()
def run(time_budget: int = 300, batch_size: int = 32):
    autoresearch.remote(time_budget=time_budget, batch_size=batch_size)
