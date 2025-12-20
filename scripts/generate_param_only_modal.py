"""Modal wrapper to generate a param-only decode checkpoint from the latest full-state checkpoint.

Usage:
  modal run scripts/generate_param_only_modal.py::run \
    --run-name full_run
"""

import os
import subprocess
from typing import Optional

import modal

APP_NAME = "ping-llm-generate-param-only"
WORKDIR = "/workspace"
CONFIG_PATH = f"{WORKDIR}/src/MaxText/configs/latency_network.yml"
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")
OUTPUT_BASE = "/mnt/outputs/latency_network"
PARAM_ONLY_RUN_NAME = "param_only_checkpoint"
PARAM_ONLY_DIR = f"{OUTPUT_BASE}/{PARAM_ONLY_RUN_NAME}/checkpoints"

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
    # Stage 3: Copy the rest of the code (fast layer that rebuilds on code changes)
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App(APP_NAME)
shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _find_latest_checkpoint_items(run_name: str) -> str:
    """Find the latest numeric checkpoint directory that contains an items subdir."""
    candidates = [
        os.path.join(OUTPUT_BASE, run_name, "checkpoints"),
        os.path.join(OUTPUT_BASE, run_name, run_name, "checkpoints"),
    ]
    for root in candidates:
        if not os.path.isdir(root):
            continue
        numeric = []
        for entry in os.listdir(root):
            if entry.isdigit():
                step_dir = os.path.join(root, entry)
                items_dir = os.path.join(step_dir, "items")
                if os.path.isdir(items_dir):
                    numeric.append(int(entry))
        if numeric:
            latest = str(max(numeric))
            return os.path.join(root, latest, "items")
    raise FileNotFoundError(
        f"No checkpoints with items/ found for run_name='{run_name}' in: {', '.join(candidates)}"
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=4,
    volumes={"/mnt": shared_vol},
    timeout=60 * 60 * 2,
    env={
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=1",
    },
)
def run(run_name: str = "full_run"):
    # Create symlinks so config's relative paths work
    if not os.path.exists(f"{WORKDIR}/outputs"):
        os.symlink("/mnt/outputs", f"{WORKDIR}/outputs")

    full_checkpoint = _find_latest_checkpoint_items(run_name)
    print(f"Using latest full-state checkpoint: {full_checkpoint}")
    print(f"Saving param-only checkpoint to: {PARAM_ONLY_DIR}")

    cmd = [
        "python",
        "-m",
        "MaxText.generate_param_only_checkpoint",
        CONFIG_PATH,
        f"load_full_state_path={full_checkpoint}",
        f"base_output_directory={OUTPUT_BASE}",
        f"run_name={PARAM_ONLY_RUN_NAME}",
        "enable_checkpointing=true",
        "hardware=gpu",
    ]
    subprocess.run(cmd, cwd=WORKDIR, check=True)
