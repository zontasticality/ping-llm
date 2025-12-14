"""Modal runner for MaxText PLAN_2 on a single A100.

Usage (single volume with data + outputs):
  modal volume create ping-llm
  modal volume put ping-llm data/sharded
  modal run scripts/train/modal_maxtext.py::train --run-name plan2_modal
"""

import os
import subprocess

import modal

APP_NAME = "ping-llm-maxtext"
WORKDIR = "/workspace"
CONFIG_PATH = f"{WORKDIR}/src/MaxText/configs/latency_network.yml"

VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")

# Base image with CUDA 12.4 + Python 3.12
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .copy_local_dir(".", WORKDIR)
    .run_commands(
        "pip install --upgrade pip",
        "pip install -r /workspace/dependencies/requirements/generated_requirements/cuda12-requirements.txt",
        "pip install -e /workspace",
    )
)

app = modal.App(APP_NAME)

shared_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=modal.gpu.A100(),
    volumes={"/mnt": shared_vol},
    timeout=60 * 60 * 48,  # 48 hours
)
def train(
    run_name: str = "plan2_modal",
    steps: int = 200_000,
    batch_size: int = 32,
    grain_workers: int = 8,
):
    """Launch MaxText training on a single A100."""
    os.environ["DECOUPLE_GCLOUD"] = "TRUE"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True",
    )

    # Shared volume layout:
    # /mnt/data/sharded/train/*.parquet
    # /mnt/data/sharded/test/*.parquet
    # /mnt/outputs/<run_name>/
    os.makedirs("/mnt/outputs", exist_ok=True)

    cmd = [
        "python",
        "-m",
        "MaxText.train",
        CONFIG_PATH,
        f"run_name={run_name}",
        "hardware=gpu",
        "base_output_directory=/mnt/outputs",
        "dataset_type=grain",
        "grain_train_files=/mnt/data/sharded/train/*.parquet",
        "grain_eval_files=/mnt/data/sharded/test/*.parquet",
        f"grain_worker_count={grain_workers}",
        f"per_device_batch_size={batch_size}",
        f"steps={steps}",
        "checkpoint_period=5000",
        "eval_interval=1000",
        "eval_steps=100",
        "log_period=100",
    ]

    subprocess.run(cmd, check=True, cwd=WORKDIR)
