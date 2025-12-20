#!/usr/bin/env python3
"""
Inspect checkpoints stored on Modal volume.

This script lists all checkpoints across all training runs, their sizes,
training steps, and metadata. It helps track training progress and manage
checkpoint storage.

Usage:
    # List all checkpoints across all runs
    modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints

    # List checkpoints for a specific run
    modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints --run-name my_run

    # Get details of a specific checkpoint
    modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoint --run-name my_run --step 1000

Note:
    Checkpoint directory structure: /mnt/outputs/latency_network/<run_name>/checkpoints/
    Each run has its own checkpoints directory.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from datetime import datetime

try:
    import modal
except ImportError:
    print("ERROR: modal not installed. Install with: pip install modal")
    exit(1)


VOLUME_NAME = os.environ.get("MODAL_VOLUME", "ping-llm")
# Base path contains run-specific subdirectories
# Structure: /mnt/outputs/latency_network/<run_name>/checkpoints/
OUTPUT_BASE_PATH = "/mnt/outputs/latency_network"

app = modal.App("inspect-checkpoints")
volume = modal.Volume.from_name(VOLUME_NAME)

# Simple image with just Python
image = modal.Image.debian_slim(python_version="3.12")


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    step: int
    path: str
    size_mb: float
    num_files: int
    created_time: Optional[datetime] = None
    has_metadata: bool = False
    metadata: Optional[Dict] = None


def get_directory_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"Warning: Error calculating size for {path}: {e}")
    return total


def count_files(path: Path) -> int:
    """Count number of files in a directory."""
    count = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                count += 1
    except Exception as e:
        print(f"Warning: Error counting files for {path}: {e}")
    return count


def parse_checkpoint_metadata(checkpoint_path: Path) -> Optional[Dict]:
    """Parse checkpoint metadata if it exists."""
    metadata_file = checkpoint_path / "_CHECKPOINT_METADATA"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse metadata for {checkpoint_path}: {e}")
    return None


def get_checkpoint_info(checkpoint_path: Path, step: int) -> CheckpointInfo:
    """Extract information about a checkpoint."""
    size_bytes = get_directory_size(checkpoint_path)
    size_mb = size_bytes / (1024 * 1024)
    num_files = count_files(checkpoint_path)
    metadata = parse_checkpoint_metadata(checkpoint_path)

    # Try to get creation time
    created_time = None
    try:
        stat = checkpoint_path.stat()
        created_time = datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        pass

    return CheckpointInfo(
        step=step,
        path=str(checkpoint_path),
        size_mb=round(size_mb, 2),
        num_files=num_files,
        created_time=created_time,
        has_metadata=metadata is not None,
        metadata=metadata
    )


@app.function(volumes={"/mnt": volume}, image=image)
def list_all_runs() -> List[str]:
    """List all training runs in the output directory."""
    base_path = Path(OUTPUT_BASE_PATH)

    if not base_path.exists():
        print(f"Output directory not found: {base_path}")
        return []

    runs = []
    for entry in base_path.iterdir():
        if entry.is_dir():
            # Check if it has a checkpoints subdirectory
            checkpoints_dir = entry / "checkpoints"
            if checkpoints_dir.exists():
                runs.append(entry.name)

    return sorted(runs)


@app.function(volumes={"/mnt": volume}, image=image)
def list_checkpoints(run_name: Optional[str] = None) -> Dict[str, List[CheckpointInfo]]:
    """
    List all checkpoints in the Modal volume.

    Args:
        run_name: If specified, only list checkpoints for this run.
                  If None, list checkpoints for all runs.

    Returns:
        Dictionary mapping run_name to list of CheckpointInfo
    """
    base_path = Path(OUTPUT_BASE_PATH)

    if not base_path.exists():
        print(f"Output directory not found: {base_path}")
        return {}

    all_checkpoints = {}

    # Get list of runs to process
    if run_name:
        run_dirs = [base_path / run_name]
    else:
        run_dirs = [entry for entry in base_path.iterdir() if entry.is_dir()]

    for run_dir in run_dirs:
        if not run_dir.exists():
            continue

        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue

        run = run_dir.name
        checkpoints = []

        # Orbax checkpoints are named by step number
        # Look for directories that are either:
        # 1. Numeric (step number) - e.g., "0", "1000", "2000"
        # 2. Numeric with .orbax-checkpoint-tmp suffix (incomplete checkpoint)
        for entry in sorted(checkpoints_dir.iterdir()):
            if entry.is_dir():
                name = entry.name

                # Try to parse step number
                step = None
                if name.isdigit():
                    step = int(name)
                elif name.endswith('.orbax-checkpoint-tmp'):
                    # Incomplete checkpoint
                    step_str = name.replace('.orbax-checkpoint-tmp', '')
                    if step_str.isdigit():
                        step = int(step_str)

                if step is not None:
                    info = get_checkpoint_info(entry, step)
                    checkpoints.append(info)

        if checkpoints:
            all_checkpoints[run] = checkpoints

    return all_checkpoints


@app.function(volumes={"/mnt": volume}, image=image)
def get_checkpoint_details(run_name: str, step: int) -> Optional[CheckpointInfo]:
    """Get detailed information about a specific checkpoint."""
    checkpoint_base = Path(OUTPUT_BASE_PATH) / run_name / "checkpoints"

    if not checkpoint_base.exists():
        print(f"Checkpoint directory not found for run: {run_name}")
        return None

    # Check for complete checkpoint
    checkpoint_path = checkpoint_base / str(step)
    if not checkpoint_path.exists():
        # Check for incomplete checkpoint
        checkpoint_path = checkpoint_base / f"{step}.orbax-checkpoint-tmp"
        if not checkpoint_path.exists():
            print(f"Checkpoint not found for run {run_name} at step {step}")
            return None

    return get_checkpoint_info(checkpoint_path, step)


@app.local_entrypoint()
def inspect_checkpoints(run_name: Optional[str] = None):
    """Main entry point: list all checkpoints."""
    print("=" * 80)
    print("MODAL CHECKPOINT INSPECTOR")
    print("=" * 80)
    print(f"Volume: {VOLUME_NAME}")
    print(f"Path: {OUTPUT_BASE_PATH}")
    if run_name:
        print(f"Run: {run_name}")
    print()

    all_checkpoints = list_checkpoints.remote(run_name)

    if not all_checkpoints:
        print("No checkpoints found.")
        print("\nAvailable runs:")
        runs = list_all_runs.remote()
        if runs:
            for run in runs:
                print(f"  - {run}")
            print(f"\nTo inspect a specific run: modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints --run-name <run_name>")
        else:
            print("  No runs found")
        return

    # Print summary for each run
    total_runs = len(all_checkpoints)
    total_checkpoints = sum(len(ckpts) for ckpts in all_checkpoints.values())
    total_size_mb = 0

    for run, checkpoints in sorted(all_checkpoints.items()):
        print(f"\n{'=' * 80}")
        print(f"Run: {run}")
        print(f"{'=' * 80}")
        print(f"Found {len(checkpoints)} checkpoint(s):\n")

        # Print header
        print(f"{'Step':<10} {'Size (MB)':<12} {'Files':<8} {'Status':<15} {'Created':<20}")
        print("-" * 80)

        # Sort by step
        checkpoints.sort(key=lambda x: x.step)

        run_size_mb = 0
        for ckpt in checkpoints:
            status = "Complete" if not ckpt.path.endswith('.orbax-checkpoint-tmp') else "Incomplete"
            created = ckpt.created_time.strftime("%Y-%m-%d %H:%M:%S") if ckpt.created_time else "Unknown"

            print(f"{ckpt.step:<10} {ckpt.size_mb:<12.2f} {ckpt.num_files:<8} {status:<15} {created:<20}")
            run_size_mb += ckpt.size_mb

        print("-" * 80)
        print(f"Run total: {len(checkpoints)} checkpoints, {run_size_mb:.2f} MB")
        total_size_mb += run_size_mb

        # Show latest checkpoint for this run
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.step)
            print(f"\nLatest checkpoint: Step {latest.step}")
            print(f"  Size: {latest.size_mb:.2f} MB")
            print(f"  Files: {latest.num_files}")

    print(f"\n{'=' * 80}")
    print(f"Grand Total: {total_runs} run(s), {total_checkpoints} checkpoint(s), {total_size_mb:.2f} MB")
    print(f"{'=' * 80}")


@app.local_entrypoint()
def inspect_checkpoint(run_name: str, step: int):
    """Inspect a specific checkpoint by run name and step number."""
    print("=" * 80)
    print(f"CHECKPOINT DETAILS - {run_name} - Step {step}")
    print("=" * 80)
    print(f"Volume: {VOLUME_NAME}")
    print()

    info = get_checkpoint_details.remote(run_name, step)

    if not info:
        return

    print(f"Run: {run_name}")
    print(f"Step: {info.step}")
    print(f"Path: {info.path}")
    print(f"Size: {info.size_mb:.2f} MB")
    print(f"Files: {info.num_files}")
    print(f"Status: {'Complete' if not info.path.endswith('.orbax-checkpoint-tmp') else 'Incomplete'}")

    if info.created_time:
        print(f"Created: {info.created_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if info.metadata:
        print("\nMetadata:")
        print(json.dumps(info.metadata, indent=2))
    else:
        print("\nNo metadata file found")


if __name__ == "__main__":
    # For local testing without Modal
    print("This script is designed to run with Modal.")
    print("Usage:")
    print("  # List all checkpoints across all runs")
    print("  modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints")
    print()
    print("  # List checkpoints for a specific run")
    print("  modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints --run-name my_run")
    print()
    print("  # Inspect a specific checkpoint")
    print("  modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoint --run-name my_run --step 1000")
