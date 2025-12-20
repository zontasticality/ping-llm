# Checkpointing in MaxText

## Overview

MaxText uses **Orbax** for checkpointing, which stores the complete training state including model parameters, optimizer state, and the current training step. This allows training to resume from any checkpoint seamlessly.

## How Checkpointing Works

### Checkpoint Structure

Each checkpoint is stored in a directory named by its **step number**:
```
outputs/latency_network/checkpoints/
├── 0/                      # Step 0 (initial checkpoint)
├── 1000/                   # Step 1000
├── 2000/                   # Step 2000
└── 5000/                   # Step 5000
```

### What's Stored in a Checkpoint

**TrainState** (from Flax):
```python
@dataclass
class TrainState:
    step: int | jax.Array          # Current training step (THE KEY)
    params: FrozenDict             # Model parameters
    opt_state: OptState            # Optimizer state (Adam moments, etc.)
    apply_fn: Callable             # Model apply function (not saved)
    tx: GradientTransformation     # Optimizer (not saved)
```

**Key Point**: The **`step`** field is part of the saved state. When you load a checkpoint, you resume at exactly that step.

### Checkpoint Lifecycle

1. **Saving** (every `checkpoint_period` steps):
   ```python
   # In train.py line ~446
   checkpointing.maybe_save_checkpoint(
       checkpoint_manager,
       state,           # Contains state.step
       config,
       data_iterator,
       step             # Current step number (becomes directory name)
   )
   ```

2. **Loading** (on training start):
   ```python
   # In checkpointing.py
   restored = checkpoint_manager.restore(
       step=-1,  # -1 means "latest checkpoint"
       args=checkpoint_args
   )
   state = restored["items"]  # State with step field intact
   ```

3. **Resuming**:
   ```python
   # In train.py line ~414
   start_step = get_first_step(state)  # Returns int(state.step)

   # Training loop starts from start_step
   for step in np.arange(start_step, config.steps):
       # Training continues from where it left off
   ```

### Directory Structure on Modal

When training on Modal with volume `/mnt`:
```
/mnt/
└── outputs/
    └── latency_network/
        ├── run_name_1/                     # First training run
        │   ├── checkpoints/                # Orbax checkpoints
        │   │   ├── 0/                      # Step 0
        │   │   │   ├── _CHECKPOINT_METADATA    # Orbax metadata (JSON)
        │   │   │   └── items.orbax-checkpoint-tmp/
        │   │   │       ├── _sharding       # Sharding info
        │   │   │       ├── array_metadatas/    # Array metadata
        │   │   │       └── ocdbt.process_0/    # Actual parameter data
        │   │   ├── 1000/                   # Step 1000
        │   │   └── 2000/                   # Step 2000
        │   └── tensorboard/                # TensorBoard logs
        │       └── run_name_1/
        │           └── events.out.tfevents.*
        └── run_name_2/                     # Second training run
            ├── checkpoints/
            └── tensorboard/
```

**Key Structure**: `base_output_directory/<run_name>/checkpoints/<step>/`

Example: `/mnt/outputs/latency_network/my_run/checkpoints/1000/`

### Incomplete Checkpoints

During saving, Orbax uses temporary directories:
```
checkpoints/
└── 1000.orbax-checkpoint-tmp/  # Still being written
```

Once complete, it's renamed to:
```
checkpoints/
└── 1000/  # Ready to use
```

## Step Tracking and Resume Logic

### How Steps are Tracked

1. **Initial Training** (no checkpoint):
   ```python
   state = TrainState.create(
       apply_fn=model.apply,
       params=params,
       tx=optimizer,
       step=0  # Start from step 0
   )
   ```

2. **Resuming from Checkpoint**:
   ```python
   # Load checkpoint
   state = checkpoint_manager.restore(step=-1)  # Latest

   # Extract step
   start_step = int(state.step)  # e.g., 1000

   # Continue training
   for step in range(start_step, config.steps):
       state, metrics = train_step(state, batch)
       # state.step is now 1001, 1002, 1003...
   ```

3. **Step Increment** (automatic):
   ```python
   # Inside TrainState.apply_gradients()
   return self.replace(
       step=self.step + 1,  # Auto-increment
       params=new_params,
       opt_state=new_opt_state
   )
   ```

### Warmup Steps Interaction

**warmup_steps** is relative to the **current training epoch**, NOT the checkpoint step.

From `configs/latency_network.yml`:
```yaml
warmup_steps: 2000
steps: 200000
```

**Scenario 1: Training from scratch**
- Step 0-2000: Learning rate warmup (cosine)
- Step 2000-200000: Regular cosine decay

**Scenario 2: Resuming from step 50000**
```python
# state.step = 50000
start_step = 50000

# Warmup already completed (50000 > 2000)
# Learning rate continues cosine decay from step 50000
```

**Key Point**: Warmup steps are absolute, not relative to resume point.

The learning rate schedule is computed as:
```python
# From maxtext_utils.py
def create_learning_rate_schedule(config):
    if step < warmup_steps:
        # Warmup: linear ramp from 0 to peak_lr
        lr = peak_lr * (step / warmup_steps)
    else:
        # Cosine decay from warmup_steps to total_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = peak_lr * 0.5 * (1 + cos(pi * progress))
```

So if you resume at step 50000 with warmup_steps=2000:
- Warmup is skipped (already past)
- Cosine decay continues from step 50000

## Checkpoint Management

### Configuration

In `latency_network.yml`:
```yaml
# Checkpointing
enable_checkpointing: true
checkpoint_period: 1000         # Save every 1000 steps
save_checkpoint_on_completion: true  # Save when training finishes
```

### Checkpoint on Interruption

The training script saves on:
- **KeyboardInterrupt** (Ctrl+C)
- **SystemExit**
- **Modal timeout**

```python
# In train.py line ~494
except KeyboardInterrupt:
    max_logging.log(f"Training interrupted at step {step}. Saving checkpoint...")
    checkpointing.maybe_save_checkpoint(
        checkpoint_manager, state, config, data_iterator, step
    )
```

### Inspecting Checkpoints

Use the provided inspection script:

```bash
# List all checkpoints on Modal volume
modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints

# Inspect a specific checkpoint
modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoint --step 1000
```

Example output:
```
================================================================================
MODAL CHECKPOINT INSPECTOR
================================================================================
Volume: ping-llm
Path: /mnt/outputs/latency_network/checkpoints

Found 3 checkpoint(s):

Step       Size (MB)    Files    Status          Created
--------------------------------------------------------------------------------
0          245.67       127      Complete        2024-12-19 10:30:15
1000       245.89       127      Complete        2024-12-19 11:15:42
2000       246.12       127      Complete        2024-12-19 12:01:08
--------------------------------------------------------------------------------
Total: 3 checkpoints, 737.68 MB

Latest checkpoint: Step 2000
  Size: 246.12 MB
  Files: 127
```

## Metadata Stored

Each checkpoint stores:

1. **_CHECKPOINT_METADATA** (JSON):
   ```json
   {
     "item_handlers": null,
     "metrics": {},
     "performance_metrics": {},
     "init_timestamp_nsecs": 1765666494089463221,
     "commit_timestamp_nsecs": null,
     "custom_metadata": {}
   }
   ```

2. **TrainState** (Orbax format):
   - `state.step` - Current step number
   - `state.params` - Model parameters
   - `state.opt_state` - Adam optimizer moments (m, v)

3. **Data Iterator State** (if using Grain):
   - Saved in `iter/` subdirectory
   - Allows resuming from exact same data position
   - Format: JSON with iterator state

### Calculating Training Progress

```python
# From checkpoint
checkpoint_step = 2000

# From config
total_steps = 200000
checkpoint_period = 1000

# Remaining steps
remaining = total_steps - checkpoint_step  # 198000

# Progress
progress = checkpoint_step / total_steps  # 1%

# Estimated checkpoints to save
remaining_checkpoints = remaining / checkpoint_period  # 198
```

## Best Practices

### Checkpoint Frequency

**Balance**:
- Too frequent (e.g., every 10 steps): High I/O overhead, slow training
- Too infrequent (e.g., every 10000 steps): Risk losing progress

**Recommended**:
```yaml
checkpoint_period: 1000  # For 200K total steps (every 0.5%)
```

For longer training:
```yaml
checkpoint_period: 5000  # For 1M+ total steps
```

### Storage Management

Each checkpoint is ~250 MB (for 95M parameter model).

**Example**:
- Training: 200,000 steps
- Checkpoint period: 1000
- Total checkpoints: 200
- Total storage: ~50 GB

**Cleanup old checkpoints**:
```python
# In config (automatic cleanup)
max_num_checkpoints_to_keep: 5  # Keep only latest 5
```

### Manual Checkpoint Inspection

```bash
# On Modal volume
modal volume ls ping-llm outputs/latency_network/checkpoints

# Download a checkpoint
modal volume get ping-llm outputs/latency_network/checkpoints/1000 ./local_checkpoint/
```

## Resuming Training

### Automatic Resume

Training automatically resumes from the latest checkpoint:

```bash
# First run
modal run scripts/train/modal_wrapper.py::run \
  --run-name my_run --steps 200000 --batch-size 128

# Container killed at step 5000
# Re-run with same command - automatically resumes from step 5000
modal run scripts/train/modal_wrapper.py::run \
  --run-name my_run --steps 200000 --batch-size 128
```

### Resume from Specific Step

```yaml
# In config
load_full_state_path: "outputs/latency_network/checkpoints/1000"
```

Or via CLI:
```bash
python scripts/train.py \
  --config src/MaxText/configs/latency_network.yml \
  load_full_state_path=outputs/latency_network/checkpoints/1000
```

## Troubleshooting

### Checkpoint Loading Fails

**Error**: `File not found: checkpoints/1000`

**Solution**: Check checkpoint exists:
```bash
modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints
```

### Step Count Mismatch

**Symptom**: Training starts from wrong step

**Cause**: `state.step` is the source of truth

**Debug**:
```python
# Add logging in train.py
max_logging.log(f"Loaded state.step = {state.step}")
max_logging.log(f"start_step = {start_step}")
```

### Checkpoint Size Growing

**Symptom**: Later checkpoints are larger

**Cause**: Optimizer state accumulates statistics

**Note**: This is normal for Adam (maintains running moments)

## Advanced: Emergency Checkpointing

MaxText supports multi-tier checkpointing for fault tolerance:

```yaml
enable_emergency_checkpoint: true
local_checkpoint_period: 100    # Local SSD (fast)
persistent_checkpoint_period: 1000  # Remote storage (durable)
```

This is primarily for TPU training with preemption. GPU training typically uses regular checkpoints.

## Summary

**Key Takeaways**:
1. ✅ Checkpoints store the complete TrainState including `step`
2. ✅ Resume is automatic - training continues from `state.step`
3. ✅ Warmup steps are absolute, not relative to resume point
4. ✅ Use `scripts/inspect_modal_checkpoints.py` to inspect checkpoints
5. ✅ Checkpoint every 1000 steps for 200K total (balance I/O vs safety)
6. ✅ Training saves checkpoint on interruption (Ctrl+C, timeout, etc.)

**Checkpoint = Complete snapshot of training state at a specific step**

## Inspecting Checkpoints on Modal

Use the provided inspection script to view checkpoints across all runs:

```bash
# List all checkpoints across all runs
modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints

# List checkpoints for a specific run
modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoints --run-name my_run

# Inspect detailed info for a specific checkpoint
modal run scripts/inspect_modal_checkpoints.py::inspect_checkpoint \
  --run-name my_run \
  --step 1000
```

Example output:
```
================================================================================
MODAL CHECKPOINT INSPECTOR
================================================================================
Volume: ping-llm
Path: /mnt/outputs/latency_network

================================================================================
Run: experiment_1
================================================================================
Found 3 checkpoint(s):

Step       Size (MB)    Files    Status          Created
--------------------------------------------------------------------------------
0          245.67       127      Complete        2024-12-19 10:30:15
1000       245.89       127      Complete        2024-12-19 11:15:42
2000       246.12       127      Complete        2024-12-19 12:01:08
--------------------------------------------------------------------------------
Run total: 3 checkpoints, 737.68 MB

Latest checkpoint: Step 2000
  Size: 246.12 MB
  Files: 127

================================================================================
Grand Total: 1 run(s), 3 checkpoint(s), 737.68 MB
================================================================================
```
