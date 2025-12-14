#!/usr/bin/env python3
"""
Wrapper to run MaxText training with Wandb logging.

This script:
1. Initializes a wandb run
2. Launches MaxText training
3. Monitors TensorBoard logs and syncs to wandb
4. Logs system metrics (GPU usage, throughput, etc.)

Usage:
    python scripts/train_with_wandb.py \
        --config src/MaxText/configs/latency_network.yml \
        --project ping-llm-plan2 \
        --name smoke_test \
        --wandb-mode online  # or 'offline' or 'disabled'
"""

import argparse
import os
import sys
import subprocess
import wandb
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run MaxText with Wandb logging")
    parser.add_argument("--config", required=True, help="MaxText config file")
    parser.add_argument("--project", default="ping-llm-plan2", help="Wandb project name")
    parser.add_argument("--name", default=None, help="Run name (auto-generated if not provided)")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                        help="Wandb mode: online (sync immediately), offline (sync later), disabled (no wandb)")
    parser.add_argument("--tags", nargs="+", default=[], help="Wandb tags")
    parser.add_argument("maxtext_args", nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to MaxText.train")
    return parser.parse_args()

def load_config(config_file):
    """Load MaxText config to extract key parameters."""
    import yaml
    with open(config_file) as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()

    # Load MaxText config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Initialize wandb
    print(f"\nInitializing Wandb (mode: {args.wandb_mode})...")
    run = wandb.init(
        project=args.project,
        name=args.name,
        mode=args.wandb_mode,
        config=config,
        tags=["maxtext", "plan2", "network-measurement"] + args.tags,
    )

    print(f"✓ Wandb run initialized")
    if args.wandb_mode == "online":
        print(f"  Run URL: {run.url}")
    print(f"  Project: {run.project}")
    print(f"  Run ID: {run.id}\n")

    # Build MaxText command
    maxtext_cmd = [
        "python", "-m", "MaxText.train",
        args.config,
    ]

    # Add any additional MaxText arguments
    if args.maxtext_args:
        # Remove leading '--' if present
        maxtext_args_clean = [a for a in args.maxtext_args if a != '--']
        maxtext_cmd.extend(maxtext_args_clean)

    # Set environment variables
    env = os.environ.copy()
    env["DECOUPLE_GCLOUD"] = "TRUE"
    env["WANDB_RUN_ID"] = run.id  # Link to this wandb run

    # Add src to PYTHONPATH so MaxText can be imported
    project_root = Path(__file__).parent.parent
    src_path = str(project_root / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    print("=" * 60)
    print("Starting MaxText Training")
    print("=" * 60)
    print(f"Command: {' '.join(maxtext_cmd)}\n")

    # Run MaxText training
    try:
        process = subprocess.Popen(
            maxtext_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output and parse for metrics
        for line in process.stdout:
            print(line, end='')

            # Parse training metrics from output
            # Example line: "Step 100: loss=8.234, learning_rate=0.0003, tokens/sec=45123"
            if "Step" in line and "loss=" in line:
                try:
                    parts = line.split(",")
                    step = int(parts[0].split("Step")[1].split(":")[0].strip())

                    metrics = {}
                    for part in parts:
                        if "loss=" in part:
                            metrics["train/loss"] = float(part.split("=")[1].strip())
                        elif "learning_rate=" in part:
                            metrics["train/learning_rate"] = float(part.split("=")[1].strip())
                        elif "tokens/sec=" in part:
                            metrics["train/tokens_per_sec"] = float(part.split("=")[1].strip())

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
