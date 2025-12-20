#!/usr/bin/env python3
"""
MaxText training with Wandb integration via TensorBoard sync.

This approach uses wandb.tensorboard.patch() to automatically sync TensorBoard
logs to Wandb, which is more reliable than parsing stdout.

Usage:
    # First time setup (one-time):
    wandb login

    # Run training with wandb:
    python scripts/train.py \
        --config src/MaxText/configs/latency_network.yml \
        --steps 10 \
        --batch-size 2

    # Or for full training (uses defaults: project=ping-llm, name=full-run):
    python scripts/train.py \
        --config src/MaxText/configs/latency_network.yml \
        --steps 5000 \
        --batch-size 128
"""

# CRITICAL: Set logging environment variables BEFORE any imports
# This must come before importing TensorFlow, JAX, or any Google libraries
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Suppress TF warnings (3 = errors only)
os.environ.setdefault("JAX_LOG_COMPILES", "0")  # Disable JAX compilation logs
os.environ.setdefault("JAX_PLATFORMS", "")  # Let JAX auto-detect

# DEBUG: Verify environment variables are set
print(f"[DEBUG] TF_CPP_MIN_LOG_LEVEL={os.environ.get('TF_CPP_MIN_LOG_LEVEL')}")
print(f"[DEBUG] JAX_LOG_COMPILES={os.environ.get('JAX_LOG_COMPILES')}")
print(f"[DEBUG] PYTHONWARNINGS={os.environ.get('PYTHONWARNINGS')}")

import argparse
import sys
import subprocess
from pathlib import Path
import logging

# Configure Python logging to suppress noisy libraries
logging.getLogger("absl").setLevel(logging.ERROR)  # Suppress ABSL (used by JAX/TF)
logging.getLogger("grain").setLevel(logging.WARNING)  # Suppress Grain info logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.WARNING)

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

try:
    import jax
except ImportError:
    print("ERROR: jax not installed. Install with: pip install jax")
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


def get_peak_tflops():
    """
    Get peak TFLOPs for the current GPU.

    Returns peak TFLOPS for BF16/FP16 operations (typical for LLM training).
    If GPU type cannot be determined, returns None.
    """
    try:
        device = jax.devices()[0]
        device_kind = device.device_kind

        # Peak TFLOPs for common GPUs (BF16/FP16)
        # Source: NVIDIA specs
        gpu_specs = {
            "NVIDIA A100-SXM4-40GB": 312.0,
            "NVIDIA A100-SXM4-80GB": 312.0,
            "NVIDIA A100-PCIE-40GB": 312.0,
            "NVIDIA A100-PCIE-80GB": 312.0,
            "NVIDIA A100 80GB": 312.0,
            "NVIDIA A100": 312.0,
            "Tesla A100-SXM4-40GB": 312.0,
            "Tesla A100-SXM4-80GB": 312.0,
            "NVIDIA H100 80GB HBM3": 989.0,  # SXM
            "NVIDIA H100": 989.0,
            "NVIDIA H100 PCIe": 756.0,
            "Tesla H100": 989.0,
            "NVIDIA V100-SXM2-16GB": 125.0,
            "NVIDIA V100": 125.0,
            "Tesla V100-SXM2-16GB": 125.0,
            "Tesla V100-SXM2-32GB": 125.0,
        }

        # Try exact match first
        if device_kind in gpu_specs:
            return gpu_specs[device_kind]

        # Try partial matches
        device_kind_lower = device_kind.lower()
        for gpu_name, peak_tflops in gpu_specs.items():
            if gpu_name.lower() in device_kind_lower or device_kind_lower in gpu_name.lower():
                print(f"Matched GPU '{device_kind}' to '{gpu_name}' (Peak: {peak_tflops} TFLOPS)")
                return peak_tflops

        print(f"WARNING: Unknown GPU type '{device_kind}'. MFU will not be calculated.")
        print("Known GPU types:", list(gpu_specs.keys()))
        return None

    except Exception as e:
        print(f"WARNING: Failed to detect GPU type: {e}. MFU will not be calculated.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run MaxText with Wandb TensorBoard sync")
    parser.add_argument("--config", required=True, help="MaxText config file")
    parser.add_argument("--project", default="ping-llm", help="Wandb project name")
    parser.add_argument("--entity", default=None, help="Wandb entity (team name)")
    parser.add_argument("--name", default="full-run", help="Run name")
    parser.add_argument("--tags", nargs="+", default=[], help="Wandb tags")
    parser.add_argument("--steps", type=int, default=200000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--hardware", default="gpu", choices=["gpu", "cpu"], help="Hardware to use")
    parser.add_argument("--enable-checkpointing", action="store_true", help="Enable checkpointing")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh training even if checkpoints exist")
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

    # Print full config at startup for debugging
    print("\n" + "=" * 80)
    print("FULL CONFIGURATION:")
    print("=" * 80)
    import pprint
    pprint.pprint(config, width=100, compact=False)
    print("=" * 80 + "\n")

    # Get peak TFLOPs for MFU calculation
    peak_tflops = get_peak_tflops()
    if peak_tflops:
        print(f"Detected GPU peak TFLOPs: {peak_tflops} (BF16/FP16)")
    else:
        print("WARNING: Could not detect GPU peak TFLOPs. MFU will not be tracked.")

    # Use provided run name (defaults to "full-run")
    run_name = args.name

    # Build output directory path BEFORE wandb.init
    project_root = Path(__file__).parent.parent
    output_dir = (project_root / "outputs" / "latency_network" / run_name).resolve()
    tensorboard_dir = output_dir / "tensorboard"

    # Initialize wandb with TensorBoard sync
    print(f"\nInitializing Wandb (mode: {args.wandb_mode})...")
    print(f"TensorBoard directory: {tensorboard_dir}")

    # Patch TensorBoard to sync to wandb (use correct path!)
    if args.wandb_mode != "disabled":
        wandb.tensorboard.patch(root_logdir=str(tensorboard_dir), pytorch=False, tensorboard_x=False)

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
            "peak_tflops_bf16": peak_tflops if peak_tflops else "unknown",
        },
        tags=["maxtext", "network-measurement"] + args.tags,
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
    # (output_dir already created above before wandb.init)

    # Auto-resume: Check for existing checkpoints (unless --no-resume is set)
    checkpoint_path = None
    if not args.no_resume:
        checkpoints_dir = output_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find the most recent checkpoint (highest numbered directory)
            checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()],
                                    key=lambda x: int(x.name) if x.name.isdigit() else -1)
            if checkpoint_dirs:
                latest_checkpoint = checkpoint_dirs[-1]
                checkpoint_path = str(latest_checkpoint)
                print(f"✓ Found existing checkpoint: {checkpoint_path}")
                print(f"  Resuming training from step {latest_checkpoint.name}")
                print()
    elif args.no_resume:
        print("ℹ  --no-resume flag set: Starting fresh training (ignoring existing checkpoints)")
        print()

    maxtext_cmd = [
        "python", "-m", "MaxText.train",
        args.config,
        f"run_name={run_name}",
        f"base_output_directory={output_dir}",  # Absolute path for Orbax
        f"hardware={args.hardware}",
        f"steps={args.steps}",
        f"per_device_batch_size={args.batch_size}",
        f"enable_checkpointing={'true' if args.enable_checkpointing else 'false'}",
    ]

    # Add checkpoint loading if we found one
    if checkpoint_path:
        maxtext_cmd.append(f"load_full_state_path={checkpoint_path}")
        print(f"  → Resuming from: {checkpoint_path}")
        print()

    # Set environment variables
    env = os.environ.copy()
    env["DECOUPLE_GCLOUD"] = "TRUE"
    env["WANDB_RUN_ID"] = run.id

    # Fix JAX memory allocation to allow larger batches
    # Without this, JAX pre-allocates 80% of GPU memory, preventing batch_size > 32
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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

        # Track timing for ETA calculation
        from collections import deque
        step_times = deque(maxlen=50)  # Track last 50 step times for moving average
        checkpoint_period = 5000  # From config (TODO: parse from config if needed)

        # Stream output and parse for metrics
        for line in process.stdout:
            print(line, end='')  # Print to console

            # Parse MaxText training metrics from stdout
            # Example: "completed step: 0, seconds: 9.020, TFLOP/s/device: 0.170, Tokens/s/device: 227.052, total_weights: 70, loss: 5.544"
            if "completed step:" in line:
                try:
                    # Extract step number
                    step_str = line.split("completed step:")[1].split(",")[0].strip()
                    step = int(step_str)

                    metrics = {}

                    # Parse step time first (needed for ETA)
                    step_time = None
                    if "seconds:" in line:
                        seconds_str = line.split("seconds:")[1].split(",")[0].strip()
                        step_time = float(seconds_str)
                        step_times.append(step_time)
                        metrics["train/step_time_seconds"] = step_time

                    # Parse loss
                    if "loss:" in line:
                        loss_str = line.split("loss:")[1].split(",")[0].strip()
                        metrics["train/loss"] = float(loss_str)

                    # Parse learning rate
                    if "learning_rate:" in line:
                        lr_str = line.split("learning_rate:")[1].split(",")[0].strip()
                        metrics["train/learning_rate"] = float(lr_str)

                    # Parse TFLOP/s/device
                    if "TFLOP/s/device:" in line:
                        tflops_str = line.split("TFLOP/s/device:")[1].split(",")[0].strip()
                        achieved_tflops = float(tflops_str)
                        metrics["train/tflops_per_device"] = achieved_tflops

                        # Calculate MFU (Model FLOP Utilization)
                        if peak_tflops:
                            mfu = (achieved_tflops / peak_tflops) * 100  # as percentage
                            metrics["train/mfu_percent"] = mfu

                    # Parse Tokens/s/device
                    if "Tokens/s/device:" in line:
                        tokens_str = line.split("Tokens/s/device:")[1].split(",")[0].strip()
                        metrics["train/tokens_per_sec_per_device"] = float(tokens_str)

                    # Parse total_weights
                    if "total_weights:" in line:
                        weights_str = line.split("total_weights:")[1].split(",")[0].strip()
                        metrics["train/total_weights"] = int(weights_str)

                    # Calculate and print ETA
                    if step_times and step > 0:
                        avg_step_time = sum(step_times) / len(step_times)
                        remaining_steps = args.steps - step - 1
                        eta_seconds = remaining_steps * avg_step_time

                        # Time to next checkpoint
                        steps_to_checkpoint = checkpoint_period - (step % checkpoint_period)
                        checkpoint_eta_seconds = steps_to_checkpoint * avg_step_time

                        # Format ETAs
                        def format_time(seconds):
                            if seconds < 60:
                                return f"{seconds:.0f}s"
                            elif seconds < 3600:
                                return f"{seconds/60:.1f}m"
                            elif seconds < 86400:
                                return f"{seconds/3600:.1f}h"
                            else:
                                days = seconds / 86400
                                hours = (seconds % 86400) / 3600
                                return f"{days:.0f}d {hours:.0f}h"

                        eta_str = format_time(eta_seconds)
                        checkpoint_eta_str = format_time(checkpoint_eta_seconds)

                        # Print enhanced progress
                        print(f"  → Step {step+1}/{args.steps} | ETA: {eta_str} | Next checkpoint: {checkpoint_eta_str}")

                        # Log ETAs to wandb
                        metrics["train/eta_seconds"] = eta_seconds
                        metrics["train/eta_to_checkpoint_seconds"] = checkpoint_eta_seconds

                    # Log to wandb
                    if metrics:
                        wandb.log(metrics, step=step)

                except Exception as e:
                    # Parsing failed, skip this line
                    pass

            # Parse MaxText eval metrics from stdout
            # Example: "eval metrics after step: 39, loss=4.733, total_weights=8751"
            if "eval metrics after step:" in line:
                try:
                    # Extract step number
                    step_str = line.split("eval metrics after step:")[1].split(",")[0].strip()
                    step = int(step_str)

                    eval_metrics = {}

                    # Parse eval loss
                    if "loss=" in line:
                        loss_str = line.split("loss=")[1].split(",")[0].strip()
                        eval_metrics["eval/avg_loss"] = float(loss_str)

                    # Parse total weights
                    if "total_weights=" in line:
                        weights_str = line.split("total_weights=")[1].split(",")[0].strip()
                        eval_metrics["eval/total_weights"] = float(weights_str)

                    # Log to wandb
                    if eval_metrics:
                        wandb.log(eval_metrics, step=step)
                        print(f"  → Logged eval metrics to wandb at step {step}: {eval_metrics}")

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
        print("\n\n" + "="*80)
        print("⚠️  TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("="*80)
        print("Sending interrupt signal to MaxText training process...")
        print("Waiting for checkpoint save...")

        # Send SIGINT (not SIGTERM) so Python can catch it as KeyboardInterrupt
        # This allows MaxText's exception handler to save a checkpoint
        import signal
        process.send_signal(signal.SIGINT)

        # Wait for the process to exit (with timeout)
        try:
            process.wait(timeout=120)  # Give it 2 minutes to save checkpoint
            print("✓ Training process saved checkpoint and exited")
        except subprocess.TimeoutExpired:
            print("⚠️  Checkpoint save timed out, forcing termination...")
            process.kill()
            process.wait()

        print("\n" + "="*80)
        wandb.log({"training_status": "interrupted"})
        wandb.finish()
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n✗ Error: {e}")
        wandb.log({"training_status": "error", "error": str(e)})
        wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
