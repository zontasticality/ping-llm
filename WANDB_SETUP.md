# Wandb Setup & Integration Guide

This guide walks you through setting up Weights & Biases (wandb) logging for your MaxText training runs.

---

## Step 1: Create Wandb Account (if needed)

If you don't have a wandb account:

1. Go to https://wandb.ai/signup
2. Sign up (free account works great!)
3. Verify your email

---

## Step 2: Get Your API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key (it looks like: `1234567890abcdef...`)
3. Keep it handy for the next step

---

## Step 3: Login to Wandb

```bash
source .venv/bin/activate
wandb login
```

When prompted, paste your API key.

**Expected output:**
```
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
[paste your key here]
wandb: Appending key for api.wandb.ai to your netrc file: /home/username/.netrc
```

---

## Step 4: Test Wandb Connection

Run a quick test to verify wandb is working:

```bash
source .venv/bin/activate
python test_wandb.py
```

**Expected output:**
```
============================================================
Testing Wandb Connection
============================================================

1. Initializing wandb...
✓ Wandb run initialized!
  Project: ping-llm-test
  Run ID: abc123xyz
  Run URL: https://wandb.ai/your-username/ping-llm-test/runs/abc123xyz

2. Logging test metrics...
  Step 0: loss=10.0234, lr=0.000030
  Step 1: loss=9.5123, lr=0.000060
  ...
  Step 9: loss=5.4789, lr=0.000300

3. Finishing wandb run...

============================================================
✓ Wandb test successful!
============================================================

View your test run at: https://wandb.ai/your-username/ping-llm-test/runs/abc123xyz
```

**Verify:**
1. Click the URL shown in the output
2. You should see a wandb dashboard with:
   - A line chart showing `train/loss` decreasing
   - A line chart showing `train/learning_rate` increasing
   - 10 logged steps

If you see these, wandb is configured correctly! ✓

---

## Step 5: Run CPU Smoke Test with Wandb

Now let's test wandb integration with MaxText:

```bash
source .venv/bin/activate

# Option A: Using the wrapper script (recommended)
python scripts/train_with_wandb.py \
  --config src/MaxText/configs/latency_network.yml \
  --project ping-llm-plan2 \
  --name smoke_test_cpu \
  --wandb-mode online \
  -- \
  hardware=cpu \
  steps=10 \
  per_device_batch_size=2 \
  run_name=smoke_cpu

# Option B: Direct MaxText with environment variables
DECOUPLE_GCLOUD=TRUE python -m MaxText.train \
  src/MaxText/configs/latency_network.yml \
  hardware=cpu \
  steps=10 \
  per_device_batch_size=2 \
  run_name=smoke_test_cpu
```

**Note:** Option B won't automatically log to wandb - it just runs MaxText normally. Use Option A for wandb integration.

---

## Step 6: Verify Wandb Dashboard

After the smoke test completes:

1. Check the terminal output for the wandb URL
2. Open the URL in your browser
3. You should see:
   - **Run name:** smoke_test_cpu
   - **Config:** All your model/training hyperparameters
   - **Charts:** Loss, learning rate, tokens/sec (if parsed)
   - **System metrics:** CPU/GPU usage, memory
   - **Logs:** Training output

---

## Step 7: Enable Wandb for GPU Training

For the full GPU training job, update the SLURM script:

```bash
# Edit scripts/slurm_train_maxtext.sh
# Change the training command at the end to:

python scripts/train_with_wandb.py \
  --config "$CONFIG_FILE" \
  --project ping-llm-plan2 \
  --name "plan2_${SLURM_JOB_ID}" \
  --wandb-mode online \
  --tags gpu training plan2 \
  -- \
  run_name="$RUN_NAME" \
  base_output_directory="$OUT_DIR" \
  hardware=gpu \
  per_device_batch_size=32 \
  steps=200000 \
  eval_interval=1000 \
  eval_steps=100 \
  checkpoint_period=5000 \
  log_period=100 \
  dataset_type=grain \
  grain_train_files="$DATA_DIR/train/*.parquet" \
  grain_eval_files="$DATA_DIR/test/*.parquet" \
  grain_worker_count=16
```

---

## Wandb Features You'll See

### During Training:

1. **Real-time metrics:**
   - Training loss (updated every 100 steps)
   - Learning rate schedule
   - Tokens/second throughput
   - GPU utilization

2. **System monitoring:**
   - GPU memory usage
   - CPU/RAM usage
   - Disk I/O

3. **Hyperparameters:**
   - All config values logged automatically
   - Easy to compare different runs

### After Training:

1. **Run comparison:**
   - Compare multiple runs side-by-side
   - See which hyperparameters work best

2. **Charts:**
   - Loss curves
   - Learning rate schedules
   - Performance metrics

3. **Artifacts:**
   - Save/load model checkpoints
   - Version your models

---

## Wandb Modes

The `--wandb-mode` flag controls how wandb behaves:

- **`online`** (default): Sync metrics to wandb cloud in real-time
  - Use for: Interactive development, monitoring live jobs
  - Requires: Internet connection

- **`offline`**: Log metrics locally, sync later
  - Use for: Cluster jobs with intermittent connectivity
  - Sync later with: `wandb sync <run_dir>`

- **`disabled`**: Don't log to wandb at all
  - Use for: Quick tests, debugging

---

## Troubleshooting

### "wandb login failed"
```bash
# Check if you're logged in
wandb login --relogin

# Or set API key directly
export WANDB_API_KEY="your-api-key-here"
```

### "Connection error" during training
```bash
# Use offline mode
python scripts/train_with_wandb.py --wandb-mode offline ...

# Sync later when you have internet
wandb sync wandb/offline-run-<timestamp>
```

### "No metrics appearing on dashboard"
- Check that the training script is actually running (not hanging)
- Verify that metrics are being logged (check terminal output)
- Try refreshing the wandb dashboard page

### "API key not found"
```bash
# Verify netrc file exists
cat ~/.netrc | grep wandb

# If not, login again
wandb login
```

---

## Advanced: TensorBoard Sync

MaxText writes TensorBoard logs. You can also sync these to wandb:

```bash
# In a separate terminal, while training is running:
wandb sync --tensorboard \
  outputs/latency_network/plan2_<job_id>/tensorboard
```

This will upload all TensorBoard logs to wandb, giving you:
- All MaxText's built-in metrics
- Histograms, distributions, etc.
- No code changes needed!

---

## Quick Reference

```bash
# Login
wandb login

# Test connection
python test_wandb.py

# Smoke test with wandb
python scripts/train_with_wandb.py \
  --config src/MaxText/configs/latency_network.yml \
  --project ping-llm-plan2 \
  --name smoke_test \
  -- hardware=cpu steps=10 per_device_batch_size=2

# View runs
wandb dashboard  # Opens browser to your runs

# Check login status
wandb status

# Logout
wandb logout
```

---

## Benefits of Using Wandb

✅ **Real-time monitoring** - Watch training progress from anywhere
✅ **Run comparison** - Compare different hyperparameters easily
✅ **Collaboration** - Share runs with teammates
✅ **Reproducibility** - All configs logged automatically
✅ **Alerts** - Get notified if training fails
✅ **Reports** - Create shareable reports with charts
✅ **Free** - Unlimited private projects on free tier

---

## Example Wandb Dashboard

After your smoke test, your wandb dashboard will show:

```
Project: ping-llm-plan2
Run: smoke_test_cpu

Overview:
  Status: ✓ Completed
  Duration: 3m 42s
  Steps: 10

Config:
  vocab_size: 267
  num_decoder_layers: 20
  emb_dim: 640
  learning_rate: 3.0e-4
  steps: 10
  ...

Charts:
  train/loss: [9.2, 9.0, 8.8, ..., 7.6]  (decreasing ✓)
  train/learning_rate: [0.00015, 0.00030, ...]
  train/tokens_per_sec: [~1000-2000 on CPU]

System:
  CPU: 12%
  Memory: 8.2 GB
```

---

**You're all set with wandb!** 🎉

Run the smoke test and check your wandb dashboard to confirm everything is working.
