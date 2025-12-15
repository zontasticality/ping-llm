#!/usr/bin/env python3
"""
Optional Weights & Biases (wandb) integration for MaxText training.

This script adds wandb logging alongside MaxText's built-in TensorBoard logging.

Setup:
    1. Install wandb: pip install wandb
    2. Login: wandb login
    3. Set WANDB_PROJECT env var in SLURM script
    4. This script will be called automatically if ENABLE_WANDB=true

Usage (in SLURM script):
    export ENABLE_WANDB=true
    export WANDB_PROJECT="ping-llm-plan2"
    export WANDB_ENTITY="your-username"  # Optional
"""

import os
import sys
import json
from pathlib import Path
import time

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def init_wandb(config_file: str, run_name: str):
    """Initialize wandb run with MaxText config."""

    # Load MaxText config
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    # Parse YAML config (simple approach)
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "ping-llm"),
        entity=os.getenv("WANDB_ENTITY", None),
        name=run_name,
        config=config,
        tags=["plan2", "network-measurement", "maxtext"],
    )

    print(f"✓ wandb initialized: {wandb.run.url}")
    return wandb.run


def watch_tensorboard_logs(tensorboard_dir: str, run):
    """
    Watch TensorBoard event files and sync to wandb.

    This is a simple approach that parses TensorBoard event files.
    For production, consider using wandb's TensorBoard integration.
    """
    from tensorboard.backend.event_processing import event_accumulator

    # Find latest event file
    event_files = sorted(Path(tensorboard_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"WARNING: No TensorBoard event files found in {tensorboard_dir}")
        return

    latest_event = event_files[-1]
    print(f"Watching TensorBoard events: {latest_event}")

    ea = event_accumulator.EventAccumulator(str(latest_event.parent))
    ea.Reload()

    last_step = 0
    while True:
        ea.Reload()

        # Log scalars
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            for event in events:
                if event.step > last_step:
                    wandb.log({tag: event.value}, step=event.step)
                    last_step = max(last_step, event.step)

        time.sleep(60)  # Check every minute


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="wandb integration for MaxText")
    parser.add_argument("--config", required=True, help="MaxText config file")
    parser.add_argument("--run-name", required=True, help="Run name")
    parser.add_argument("--tensorboard-dir", required=True, help="TensorBoard log directory")
    parser.add_argument("--mode", choices=["init", "watch"], default="init",
                        help="Mode: init (initialize only) or watch (sync TensorBoard logs)")
    args = parser.parse_args()

    run = init_wandb(args.config, args.run_name)

    if args.mode == "watch":
        try:
            watch_tensorboard_logs(args.tensorboard_dir, run)
        except KeyboardInterrupt:
            print("\n✓ wandb sync stopped")
            wandb.finish()


if __name__ == "__main__":
    main()
