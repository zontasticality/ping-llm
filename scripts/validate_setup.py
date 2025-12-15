#!/usr/bin/env python3
"""
Pre-flight validation script for MaxText training setup.
Checks all prerequisites before submitting GPU SLURM job.

Usage:
    source .venv/bin/activate
    python scripts/validate_setup.py
"""

import sys
import os
from pathlib import Path

def check(name, condition, fix_hint=""):
    """Check a condition and print status."""
    status = "✓" if condition else "✗"
    print(f"  {status} {name}")
    if not condition and fix_hint:
        print(f"    → {fix_hint}")
    return condition

def main():
    print("=" * 60)
    print("MaxText Training Setup Validation")
    print("=" * 60)
    print()

    all_ok = True

    # Python version
    print("1. Python Environment")
    import sys
    py_version = sys.version_info
    all_ok &= check(
        f"Python 3.12.x (found: {py_version.major}.{py_version.minor}.{py_version.micro})",
        py_version.major == 3 and py_version.minor == 12,
        "Install Python 3.12: recommended by MaxText"
    )

    # Python packages
    print()
    print("2. Required Packages")

    packages = {
        "jax": "JAX framework",
        "flax": "Flax (neural networks)",
        "grain": "Grain (data loading)",
        "pyarrow": "PyArrow (parquet files)",
        "wandb": "Weights & Biases",
    }

    for pkg, desc in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            all_ok &= check(f"{desc} ({version})", True)
        except ImportError:
            all_ok &= check(f"{desc}", False, f"pip install {pkg}")

    # MaxText import
    try:
        sys.path.insert(0, "src")
        from MaxText import train
        all_ok &= check("MaxText (from src/)", True)
    except ImportError as e:
        all_ok &= check("MaxText", False, f"Install maxtext or set PYTHONPATH=src")

    # JAX backend
    print()
    print("3. JAX Configuration")
    try:
        import jax
        backend = jax.default_backend()
        devices = jax.devices()
        all_ok &= check(f"JAX backend: {backend}", True)
        all_ok &= check(f"JAX devices: {len(devices)} ({', '.join(str(d.device_kind) for d in devices)})", True)

        # Check for CUDA wheels
        try:
            import jax._src.lib
            cuda_available = hasattr(jax._src.lib, 'xla_extension_version')
            all_ok &= check("JAX CUDA wheels installed", cuda_available, "pip install 'jax[cuda12]'")
        except:
            pass
    except ImportError:
        all_ok &= check("JAX", False, "pip install jax")

    # Dataset files
    print()
    print("4. Dataset Validation")

    train_shards = list(Path("data/sharded/train").glob("*.parquet")) if Path("data/sharded/train").exists() else []
    test_shards = list(Path("data/sharded/test").glob("*.parquet")) if Path("data/sharded/test").exists() else []

    all_ok &= check(
        f"Training shards: {len(train_shards)}/180",
        len(train_shards) == 180,
        "Run: python scripts/shard_parquet.py"
    )
    all_ok &= check(
        f"Test shards: {len(test_shards)}/20",
        len(test_shards) == 20,
        "Run: python scripts/shard_parquet.py"
    )

    # Config file
    print()
    print("5. Configuration Files")

    config_path = Path("src/MaxText/configs/latency_network.yml")
    all_ok &= check(
        "Config file exists (latency_network.yml)",
        config_path.exists(),
        "Check: src/MaxText/configs/latency_network.yml"
    )

    # Output directories
    print()
    print("6. Output Directories")

    logs_dir = Path("logs")
    all_ok &= check(
        "Logs directory (logs/)",
        logs_dir.exists(),
        "Run: mkdir -p logs"
    )

    # Wandb setup
    print()
    print("7. Wandb Configuration")

    wandb_config = Path.home() / ".netrc"
    wandb_configured = wandb_config.exists() and "wandb" in wandb_config.read_text()
    all_ok &= check(
        "Wandb login configured (~/.netrc)",
        wandb_configured,
        "Run: wandb login"
    )

    # SLURM script
    print()
    print("8. SLURM Setup")

    slurm_script = Path("scripts/slurm_train_maxtext.sh")
    all_ok &= check(
        "SLURM script exists",
        slurm_script.exists(),
        "Check: scripts/slurm_train_maxtext.sh"
    )

    # Requirements lock
    print()
    print("9. Reproducibility")

    req_lock = Path("requirements.lock")
    all_ok &= check(
        "Frozen dependencies (requirements.lock)",
        req_lock.exists(),
        "Run: pip freeze > requirements.lock"
    )

    # Summary
    print()
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! Ready for GPU training.")
        print()
        print("Next steps:")
        print("  1. CPU smoke test: DECOUPLE_GCLOUD=TRUE python -m MaxText.train \\")
        print("       src/MaxText/configs/latency_network.yml hardware=cpu steps=10")
        print("  2. Wandb test: python test_wandb.py")
        print("  3. Submit GPU job: sbatch scripts/slurm_train_maxtext.sh")
        return 0
    else:
        print("✗ Some checks failed. Fix issues above before proceeding.")
        print()
        print("See VENV_SETUP_PLAN.md for detailed setup instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
