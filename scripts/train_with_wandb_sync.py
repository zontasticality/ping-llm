#!/usr/bin/env python3
"""
MaxText training with Wandb integration via TensorBoard sync.

This approach uses wandb.tensorboard.patch() to automatically sync TensorBoard
logs to Wandb, which is more reliable than parsing stdout.

Usage:
    # First time setup (one-time):
    wandb login

    # Run training with wandb:
    python scripts/train_with_wandb_sync.py \
        --config src/MaxText/configs/latency_network.yml \
        --project ping-llm-plan2 \
        --name test_run \
        --steps 10 \
        --batch-size 2

    # Or for full training:
    python scripts/train_with_wandb_sync.py \
        --config src/MaxText/configs/latency_network.yml \
        --project ping-llm-plan2 \
        --name full_plan2_training \
        --steps 200000 \
        --batch-size 32
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Install with: pip install pyyaml")
    sys.exit(1)


def check_wandb_login():
    """Check if user is logged into wandb."""
    try:
        # Method 1: Check if API key exists in settings
        settings = wandb.Settings()
        if settings.api_key:
            return True

        # Method 2: Try to create API instance (faster than viewer())
        api = wandb.Api(timeout=10)
        return True
    except Exception as e:
        print(f"Debug: wandb login check failed with: {e}")
        return False


def load_config(config_file):
    """Load MaxText YAML config."""
    with open(config_file) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run MaxText with Wandb TensorBoard sync")
    parser.add_argument("--config", required=True, help="MaxText config file")
    parser.add_argument("--project", default="ping-llm-plan2", help="Wandb project name")
    parser.add_argument("--entity", default=None, help="Wandb entity (team name)")
    parser.add_argument("--name", default=None, help="Run name (auto-generated if not provided)")
    parser.add_argument("--tags", nargs="+", default=[], help="Wandb tags")
    parser.add_argument("--steps", type=int, default=200000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--hardware", default="gpu", choices=["gpu", "cpu"], help="Hardware to use")
    parser.add_argument("--enable-checkpointing", action="store_true", help="Enable checkpointing")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                        help="Wandb mode")
    args = parser.parse_args()

    # Check wandb login (soft check - wandb.init will fail with better error if not logged in)
    if args.wandb_mode != "disabled":
        if not check_wandb_login():
            print("\n" + "="*60)
            print("WARNING: wandb login check uncertain")
            print("="*60)
            print("wandb says you're logged in, but API check failed.")
            print("Proceeding anyway - wandb.init() will show proper error if needed.")
            print("="*60 + "\n")

    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Auto-generate run name if not provided
    run_name = args.name or f"maxtext_plan2_{wandb.util.generate_id()}"

    # Initialize wandb with TensorBoard sync
    print(f"\nInitializing Wandb (mode: {args.wandb_mode})...")

    # Patch TensorBoard to sync to wandb
    if args.wandb_mode != "disabled":
        wandb.tensorboard.patch(root_logdir="outputs/latency_network", pytorch=False, tensorboard_x=False)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        mode=args.wandb_mode,
        config={
            **config,
            "steps": args.steps,
            "per_device_batch_size": args.batch_size,
            "hardware": args.hardware,
            "enable_checkpointing": args.enable_checkpointing,
        },
        tags=["maxtext", "plan2", "network-measurement"] + args.tags,
        sync_tensorboard=True,  # Enable TensorBoard sync
    )

    print(f"✓ Wandb run initialized")
    if args.wandb_mode == "online":
        print(f"  Run URL: {run.url}")
    print(f"  Project: {run.project}")
    print(f"  Run name: {run.name}")
    print(f"  Run ID: {run.id}")
    print()

    # Build MaxText command
    project_root = Path(__file__).parent.parent

    maxtext_cmd = [
        "python", "-m", "MaxText.train",
        args.config,
        f"run_name={run_name}",
        f"hardware={args.hardware}",
        f"steps={args.steps}",
        f"per_device_batch_size={args.batch_size}",
        f"enable_checkpointing={'true' if args.enable_checkpointing else 'false'}",
    ]

    # Set environment variables
    env = os.environ.copy()
    env["DECOUPLE_GCLOUD"] = "TRUE"
    env["WANDB_RUN_ID"] = run.id

    # Add src to PYTHONPATH
    src_path = str(project_root / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    print("=" * 80)
    print("Starting MaxText Training with Wandb TensorBoard Sync")
    print("=" * 80)
    print(f"Command: {' '.join(maxtext_cmd)}")
    print(f"TensorBoard logs will automatically sync to Wandb")
    print("=" * 80)
    print()

    # Run MaxText training with stdout parsing for real-time metrics
    try:
        process = subprocess.Popen(
            maxtext_cmd,
            env=env,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output and parse for metrics
        for line in process.stdout:
            print(line, end='')  # Print to console

            # Parse MaxText metrics from stdout
            # Example: "completed step: 0, seconds: 9.020, TFLOP/s/device: 0.170, Tokens/s/device: 227.052, total_weights: 70, loss: 5.544"
            if "completed step:" in line:
                try:
                    # Extract step number
                    step_str = line.split("completed step:")[1].split(",")[0].strip()
                    step = int(step_str)

                    metrics = {}

                    # Parse loss
                    if "loss:" in line:
                        loss_str = line.split("loss:")[1].split(",")[0].strip()
                        metrics["train/loss"] = float(loss_str)

                    # Parse TFLOP/s/device
                    if "TFLOP/s/device:" in line:
                        tflops_str = line.split("TFLOP/s/device:")[1].split(",")[0].strip()
                        metrics["train/tflops_per_device"] = float(tflops_str)

                    # Parse Tokens/s/device
                    if "Tokens/s/device:" in line:
                        tokens_str = line.split("Tokens/s/device:")[1].split(",")[0].strip()
                        metrics["train/tokens_per_sec_per_device"] = float(tokens_str)

                    # Parse total_weights
                    if "total_weights:" in line:
                        weights_str = line.split("total_weights:")[1].split(",")[0].strip()
                        metrics["train/total_weights"] = int(weights_str)

                    # Parse seconds (step time)
                    if "seconds:" in line:
                        seconds_str = line.split("seconds:")[1].split(",")[0].strip()
                        metrics["train/step_time_seconds"] = float(seconds_str)

                    # Log to wandb
                    if metrics:
                        wandb.log(metrics, step=step)

                except Exception as e:
                    # Parsing failed, skip this line
                    pass

        process.wait()
        exit_code = process.returncode

        if exit_code == 0:
            print("\n✓ Training completed successfully!")
            wandb.log({"training_status": "completed"})
        else:
            print(f"\n✗ Training failed with exit code {exit_code}")
            wandb.log({"training_status": "failed", "exit_code": exit_code})

        wandb.finish()
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        wandb.log({"training_status": "interrupted"})
        wandb.finish()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        wandb.log({"training_status": "error", "error": str(e)})
        wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
