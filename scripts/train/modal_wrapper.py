"""Modal wrapper to run scripts/train.py on a single GPU with a shared volume.

Note: This file is named modal_wrapper.py (not modal.py) to avoid shadowing
      the modal Python module when Modal CLI runs the script.

Prereqs:
  - Modal volume `ping-llm` (or set MODAL_VOLUME) with:
      data/probe_rows/train.arrayrecord  (PLAN_3)
      data/probe_rows/test.arrayrecord   (PLAN_3)
  - WANDB_API_KEY provided via environment or Modal secret (set MODAL_WANDB_SECRET name).

Usage:
  modal run scripts/train/modal.py::run \
    --run-name plan3_modal_test \
    --steps 5000 \
    --batch-size 128 \
    --wandb-project ping-llm-plan3

Note:
  - Uses latency_network.yml (64 workers for 64-core A100-80GB)
  - Multiprocessing enabled for parallel tokenization (PLAN_3)
  - Expected throughput: ~320K-850K tokens/sec
"""

import os
import subprocess
from typing import Optional

import modal

APP_NAME = "ping-llm-maxtext-wandb-sync"
WORKDIR = "/workspace"
CONFIG_PATH = f"{WORKDIR}/src/MaxText/configs/latency_network.yml"
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")

# Ignore heavy paths when copying code into the image
IGNORE_PATTERNS = [
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "outputs",
    "logs",
    "data",
    "local_datasets",
    "archive",
    "tests",
    "docs",
    "benchmarks",
    "end_to_end",
    "*.parquet",
    ".DS_Store",
]

# Build image in stages to optimize caching:
# 1. Copy only files needed for dependency installation
# 2. Install dependencies (expensive, only rebuilds when dependencies change)
# 3. Copy the rest of the code (cheap, rebuilds on code changes)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "ninja-build",
    )
    .pip_install("uv")
    # Stage 1: Copy ONLY files needed for dependency resolution
    .add_local_file("pyproject.toml", f"{WORKDIR}/pyproject.toml", copy=True)
    .add_local_file("README.md", f"{WORKDIR}/README.md", copy=True)
    .add_local_file("build_hooks.py", f"{WORKDIR}/build_hooks.py", copy=True)
    .add_local_dir("dependencies", f"{WORKDIR}/dependencies", copy=True)
    .add_local_file(
        "src/MaxText/__init__.py", f"{WORKDIR}/src/MaxText/__init__.py", copy=True
    )  # Only __init__.py for version
    .add_local_dir(
        "src/install_maxtext_extra_deps",
        f"{WORKDIR}/src/install_maxtext_extra_deps",
        copy=True,
    )
    # Stage 2: Install dependencies (this layer is cached unless above files change)
    .run_commands(
        f"cd {WORKDIR} && CC=gcc CXX=g++ uv pip install --system -e '.[cuda12]' --resolution=lowest",
        f"cd {WORKDIR} && install_maxtext_github_deps",
    )
    .pip_install("wandb")
    # Stage 3: Copy the rest of the code (fast layer that rebuilds on code changes)
    # CACHE BUST: 2025-12-18-04 - Fix Grain mp_prefetch worker state initialization
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App(APP_NAME)
shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    cpu=8,  # A100-80GB with 64 CPUs for parallel data loading
    volumes={"/mnt": shared_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 24,  # 24 hours (Modal max 86400s)
)
def run(
    run_name: str = "full_run",
    steps: int = 5_000,
    batch_size: int = 128,
    wandb_project: str = "full_run",
):
    import signal
    import atexit

    # Create symlinks so config's relative paths work
    # Config expects: data/sharded/... and outputs/...
    # Volume provides: /mnt/data/sharded/... and /mnt/outputs/...
    os.symlink("/mnt/data", f"{WORKDIR}/data")
    os.makedirs("/mnt/outputs", exist_ok=True)
    os.symlink("/mnt/outputs", f"{WORKDIR}/outputs")

    cmd = [
        "python",
        "scripts/train.py",
        "--config",
        CONFIG_PATH,
        "--project",
        wandb_project,
        "--name",
        run_name,
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--hardware",
        "gpu",
        "--enable-checkpointing",  # Enable checkpointing to save progress
    ]

    # Start the training subprocess
    process = subprocess.Popen(cmd, cwd=WORKDIR)

    # Register cleanup handler to send SIGINT on exit
    def cleanup_handler():
        if process.poll() is None:  # Process still running
            print("\n" + "=" * 80)
            print(
                "⚠️  Container shutting down - sending interrupt to training process..."
            )
            print("=" * 80)
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=25)  # Modal gives 30s grace period
                print("✓ Training process exited gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️  Training did not exit in time")
                process.kill()
            print("=" * 80)

    atexit.register(cleanup_handler)

    # Wait for the process to complete
    exit_code = process.wait()

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, cmd)
