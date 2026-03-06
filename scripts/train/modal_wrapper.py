"""Modal wrapper to run ping-llm PyTorch training on a single GPU.

Note: This file is named modal_wrapper.py (not modal.py) to avoid shadowing
      the modal Python module when Modal CLI runs the script.

Prereqs:
  - Modal volume `ping-llm-data` with train shards + test.arrayrecord
  - Modal volume `ping-llm` for outputs (checkpoints, etc.)
  - WANDB_API_KEY provided via Modal secret (wandb-secret).

Usage:
  # Smoke test (10 steps):
  modal run scripts/train/modal_wrapper.py::run \
    --steps 10 --run-name smoke-test --batch-size 8 --no-compile

  # Smoke test with gradient accumulation (effective BS=256):
  modal run scripts/train/modal_wrapper.py::run \
    --steps 10 --run-name smoke-accum --batch-size 32 \
    --gradient-accumulation-steps 8 --no-compile

  # Full run:
  modal run scripts/train/modal_wrapper.py::run \
    --run-name my-run
"""

import os
import subprocess

import modal

APP_NAME = "ping-llm-pytorch"
WORKDIR = "/workspace"

DATA_VOLUME = "ping-llm-data"
OUTPUTS_VOLUME = "ping-llm"

DATA_MOUNT = "/mnt/data"
OUTPUTS_MOUNT = "/mnt/outputs"

# Shard paths on the data volume
TRAIN_SHARDS = [
    f"{DATA_MOUNT}/train_shards/train.arrayrecord-0000{i}-of-00004"
    for i in range(4)
]
EVAL_PATH = f"{DATA_MOUNT}/test.arrayrecord"

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
    ".claude",
    "docs",
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
    # Stage 1: Install Python deps (cached unless list changes)
    .run_commands(
        "uv pip install --system "
        "torch --index-url https://download.pytorch.org/whl/cu124 && "
        "uv pip install --system "
        "pyarrow numpy grain array_record wandb"
    )
    # Stage 2: Copy code (rebuilds on code changes)
    .add_local_dir(".", WORKDIR, ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME)
outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    volumes={DATA_MOUNT: data_vol, OUTPUTS_MOUNT: outputs_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 24,  # 24 hours
)
def _run(
    run_name: str,
    steps: int,
    batch_size: int,
    wandb_project: str,
    no_compile: bool = False,
    no_multiprocessing: bool = False,
    gradient_accumulation_steps: int = 1,
):
    import signal
    import atexit

    # Preflight: verify data files exist
    print("=== Preflight checks ===")
    for p in TRAIN_SHARDS + [EVAL_PATH]:
        exists = os.path.exists(p)
        print(f"  {'OK' if exists else 'MISSING'}: {p}")
        if not exists:
            raise FileNotFoundError(f"Data file not found: {p}")
    print("All data files present.\n")

    checkpoint_dir = f"{OUTPUTS_MOUNT}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_data = ",".join(TRAIN_SHARDS)

    cmd = [
        "python", "-m", "ping_llm.train",
        "--run-name", run_name,
        "--total-steps", str(steps),
        "--batch-size", str(batch_size),
        "--wandb-project", wandb_project,
        "--train-data", train_data,
        "--eval-data", EVAL_PATH,
        "--checkpoint-dir", checkpoint_dir,
    ]

    if no_compile:
        cmd.append("--no-compile")
    if no_multiprocessing:
        cmd.append("--no-multiprocessing")
    if gradient_accumulation_steps > 1:
        cmd.extend(["--gradient-accumulation-steps", str(gradient_accumulation_steps)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKDIR}/src"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONUNBUFFERED"] = "1"

    # Cache torch.compile graphs to the outputs volume so recompilation is skipped
    cache_dir = f"{OUTPUTS_MOUNT}/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    print(f"CMD: {' '.join(cmd)}\n", flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=WORKDIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    committed = False

    def cleanup_handler():
        nonlocal committed
        if process.poll() is None:
            print("\nContainer shutting down - sending interrupt to training...")
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=25)
                print("Training process exited gracefully")
            except subprocess.TimeoutExpired:
                print("Training did not exit in time")
                process.kill()
        if not committed:
            outputs_vol.commit()
            committed = True

    atexit.register(cleanup_handler)

    for line in process.stdout:
        print(line, end="", flush=True)

    exit_code = process.wait()
    outputs_vol.commit()
    committed = True
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, cmd)


@app.local_entrypoint()
def run(
    run_name: str = "full-run",
    steps: int = 14000,
    batch_size: int = 256,
    wandb_project: str = "ping-llm",
    no_compile: bool = False,
    no_multiprocessing: bool = False,
    gradient_accumulation_steps: int = 1,
):
    _run.remote(
        run_name=run_name,
        steps=steps,
        batch_size=batch_size,
        wandb_project=wandb_project,
        no_compile=no_compile,
        no_multiprocessing=no_multiprocessing,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
