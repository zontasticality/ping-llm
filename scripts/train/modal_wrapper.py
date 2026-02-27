"""Modal wrapper to run ping-llm PyTorch training on a single GPU.

Note: This file is named modal_wrapper.py (not modal.py) to avoid shadowing
      the modal Python module when Modal CLI runs the script.

Prereqs:
  - Modal volume `ping-llm` (or set MODAL_VOLUME) with:
      data/probe_rows/train.arrayrecord
      data/probe_rows/test.arrayrecord
  - WANDB_API_KEY provided via Modal secret (wandb-secret).

Usage:
  # Basic usage:
  modal run scripts/train/modal_wrapper.py::run

  # Custom parameters:
  modal run scripts/train/modal_wrapper.py::run \
    --run-name my-test \
    --steps 5000 \
    --batch-size 128 \
    --gpu H100
"""

import os
import subprocess
from typing import Optional

import modal

APP_NAME = "ping-llm-pytorch"
WORKDIR = "/workspace"
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")

IGNORE_PATTERNS = [
    ".git",
    ".venv",
    ".train_venv",
    ".slurm_venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "outputs",
    "logs",
    "data",
    "local_datasets",
    "archive",
    "*.parquet",
    "*.arrayrecord",
    ".DS_Store",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git", "build-essential")
    .pip_install("uv")
    # Stage 1: Install Python dependencies (cached unless list changes)
    .run_commands(
        "uv pip install --system "
        "torch "
        "pyarrow numpy grain array_record "
        "wandb "
    )
    # Stage 2: Copy code (rebuilds on code changes)
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App(APP_NAME)
shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    volumes={"/mnt": shared_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 24,  # 24 hours
)
def run(
    run_name: str = "full-run",
    steps: int = 14000,
    batch_size: int = 256,
    wandb_project: str = "ping-llm",
    gpu: str = "A100",
):
    import signal
    import atexit

    # Symlinks for relative data paths
    os.symlink("/mnt/data", f"{WORKDIR}/data")
    os.makedirs("/mnt/outputs", exist_ok=True)
    os.symlink("/mnt/outputs", f"{WORKDIR}/outputs")

    cmd = [
        "python", "-m", "ping_llm.train",
        "--run-name", run_name,
        "--total-steps", str(steps),
        "--batch-size", str(batch_size),
        "--wandb-project", wandb_project,
        "--checkpoint-dir", "outputs/checkpoints",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKDIR}/src"
    env["PYTHONWARNINGS"] = "ignore"

    process = subprocess.Popen(
        cmd,
        cwd=WORKDIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    def cleanup_handler():
        if process.poll() is None:
            print("\nContainer shutting down - sending interrupt to training...")
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=25)
                print("Training process exited gracefully")
            except subprocess.TimeoutExpired:
                print("Training did not exit in time")
                process.kill()

    atexit.register(cleanup_handler)

    for line in process.stdout:
        print(line, end="", flush=True)

    exit_code = process.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, cmd)
