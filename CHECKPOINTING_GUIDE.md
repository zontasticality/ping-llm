# Checkpointing Guide

## Current Checkpoint Settings

From `src/MaxText/configs/latency_network.yml`:

```yaml
enable_checkpointing: true
checkpoint_period: 5000        # Save every 5000 steps
save_checkpoint_on_completion: true  # Save when training finishes normally
```

## When Checkpoints Are Saved

### ✅ Automatic Saves
1. **Every 5,000 steps** - Periodic checkpoint
2. **On normal completion** - Final checkpoint at step 200,000

### ❌ NOT Saved
- **On interruption (Ctrl+C)** - You only keep the last periodic checkpoint
- **On crash/error** - You only keep the last periodic checkpoint

## What This Means

### Example Scenario:
```
Step 0     → No checkpoint yet
Step 5000  → ✅ Checkpoint saved
Step 10000 → ✅ Checkpoint saved
Step 12500 → Press Ctrl+C → ❌ Progress from 10001-12500 LOST
```

You can resume from step 10,000 but lose 2,500 steps of work.

## Checkpoint Location

Checkpoints are saved to:
```
outputs/latency_network/checkpoints/
```

Each checkpoint includes:
- Model weights
- Optimizer state (Adam momentum)
- Training step number
- RNG state

## For 200k Step Training

With `checkpoint_period: 5000`:
- **40 checkpoints** will be created (every 5k steps)
- **Checkpoint size**: ~1-2GB each (model + optimizer state)
- **Total disk usage**: ~40-80GB for all checkpoints

## Improving Checkpoint Frequency

If you want more frequent checkpoints (to lose less progress on interruption):

**Option 1: More Frequent Checkpoints**
```yaml
checkpoint_period: 1000  # Save every 1000 steps instead of 5000
```
- ✅ Lose max 1000 steps on interrupt
- ❌ Creates 200 checkpoints (more disk space)
- ❌ Slightly slower training (more I/O)

**Option 2: Two-Tier Checkpointing**
```yaml
enable_multi_tier_checkpointing: true
local_checkpoint_period: 1000          # Frequent local checkpoints
checkpoint_period: 5000                # Less frequent main checkpoints
```
- ✅ Best of both worlds
- ✅ Local checkpoints are faster to save
- ✅ Main checkpoints are for long-term storage

## Resuming Training

To resume from a checkpoint:

```bash
python scripts/train_with_wandb_sync.py \
    --config src/MaxText/configs/latency_network.yml \
    --steps 200000 \
    --batch-size 512
    # MaxText automatically detects and loads latest checkpoint
```

MaxText will:
1. Look in `outputs/latency_network/checkpoints/`
2. Find the latest checkpoint
3. Resume from that step number
4. Continue training to step 200,000

## Monitoring Checkpoints

Watch checkpoint saves in the logs:
```
Saved checkpoint at step 5000
Saved checkpoint at step 10000
...
```

Check disk usage:
```bash
du -sh outputs/latency_network/checkpoints/
```

## Recommendations for Your 200k Run

Given your setup:
1. **Keep `checkpoint_period: 5000`** - Good balance
2. **Max loss on interrupt**: 4,999 steps (~4-8 hours at current speed)
3. **If batch_size=512 speeds things up**: Consider reducing to `checkpoint_period: 2500`
