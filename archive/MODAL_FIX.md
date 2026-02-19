# Modal Script Naming Fix - 2024-12-19

## Problem

The Modal training script was named `modal.py`, causing an import conflict:

```
AttributeError: module 'modal' has no attribute 'Image'
```

**Root Cause**: When Modal CLI executes a script, it adds the script's directory to `sys.path[0]`. This caused Python to import `scripts/train/modal.py` instead of the actual `modal` package when the script tried to `import modal`.

## Solution

Renamed to avoid module shadowing:
```
scripts/train/modal.py → scripts/train/modal_wrapper.py
```

## Files Changed

1. **scripts/train/modal_wrapper.py** (renamed from modal.py)
   - Added docstring explaining naming choice

2. **run_modal_training.sh**
   - Updated script reference

3. **All documentation**
   - Updated command examples throughout docs/

## Verification

```bash
$ modal run scripts/train/modal_wrapper.py::run --help
Usage: modal run scripts/train/modal_wrapper.py::run [OPTIONS]

Options:
  --wandb-project TEXT
  --batch-size INTEGER
  --steps INTEGER
  --run-name TEXT
  -h, --help            Show this message and exit.

✅ Success - script loads and Modal builds image
```

## Usage

```bash
# Modal training
modal run scripts/train/modal_wrapper.py::run \
  --run-name my-run \
  --steps 5000 \
  --batch-size 128 \
  --wandb-project my-project

# Or use helper script
./run_modal_training.sh
```

This fix should be applied to **CHANGELOG.md** for the release notes.
