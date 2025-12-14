# Smoke Test & GPU Job Setup Guide

**System:** 16 cores, 64GB RAM
**Dataset:** 200M rows (verified ✓)
**Goal:** Run CPU smoke test, then submit GPU training job

---

## Quick Status Check

```bash
# Check what's ready
source .venv/bin/activate
python -c "import jax, flax, grain, pyarrow; print('✓ Dependencies installed')"
ls data/sharded/train/*.parquet | wc -l  # Should show 180
ls data/sharded/test/*.parquet | wc -l   # Should show 20
```

---

## Step-by-Step Walkthrough

### 1. Shard the Dataset (if not done)

```bash
source .venv/bin/activate

python scripts/shard_parquet.py \
  --input data/training_data.parquet \
  --output data/sharded \
  --train-shards 180 \
  --test-shards 20
```

**Expected time:** 5-15 minutes
**Output:** 180 train shards + 20 test shards (~1M rows each)

---

### 2. (Optional) Set Up Wandb

If you want to log to Weights & Biases:

```bash
source .venv/bin/activate
wandb login
# Enter your API key from https://wandb.ai/authorize

# Test wandb
python -c "import wandb; wandb.init(project='test', mode='disabled'); print('✓ Wandb ready')"
```

To enable in SLURM job, edit `scripts/slurm_train_maxtext.sh`:
```bash
# Uncomment these lines:
export ENABLE_WANDB=true
export WANDB_PROJECT="ping-llm-plan2"
export WANDB_ENTITY="your-username"  # Your wandb username
```

---

### 3. Run CPU Smoke Test (10 Steps)

Test that everything works before submitting the GPU job:

```bash
source .venv/bin/activate

DECOUPLE_GCLOUD=TRUE python -m MaxText.train \
  src/MaxText/configs/latency_network.yml \
  hardware=cpu \
  steps=10 \
  per_device_batch_size=2 \
  run_name=smoke_test_cpu
```

**Expected time:** 2-5 minutes
**What to look for:**
- ✓ Config loads successfully
- ✓ Grain pipeline loads data from sharded files
- ✓ Model initializes (95M params)
- ✓ Training starts and completes 10 steps
- ✓ Loss values print (expect ~8-10 initially for random init)
- ✓ No errors or warnings

**Common issues:**
- "No training shards found" → Run step 1 (sharding)
- "Module not found" → Run `source .venv/bin/activate`
- Out of memory → Reduce `per_device_batch_size=1`

---

### 4. Create Logs Directory

```bash
mkdir -p logs
```

---

### 5. Submit GPU Training Job

Once smoke test passes:

```bash
sbatch scripts/slurm_train_maxtext.sh
```

**Expected output:**
```
Submitted batch job 12345
```

**Monitor the job:**
```bash
# Check job status
squeue -u $USER

# Watch training log (live)
tail -f logs/ping-llm-plan2-*.out

# Check for errors
tail -f logs/ping-llm-plan2-*.err

# Cancel job if needed
scancel <job_id>
```

---

## What Happens During GPU Training

### Timeline (200k steps on A100, ~37 hours)
```
Hour 0:   Job starts, data loading, model init
Hour 1:   Training begins, first checkpoint at step 5000
Hour 8:   ~50k steps (25% complete)
Hour 16:  ~100k steps (50% complete)
Hour 24:  ~150k steps (75% complete)
Hour 32:  ~190k steps (95% complete)
Hour 37:  200k steps (100% complete) ✅
```

### Expected Metrics
```
Loss trajectory:
  Initial: ~8-10 (random initialization)
  10k steps: ~4-6
  50k steps: ~2-4
  100k steps: ~1.5-3
  200k steps: ~1-2 (convergence)

Throughput:
  Target: 40-50k tokens/sec
  GPU utilization: 80-95%
```

### Checkpoints
Saved every 5,000 steps to:
```
outputs/latency_network/plan2_<job_id>/checkpoint_5000/
outputs/latency_network/plan2_<job_id>/checkpoint_10000/
...
outputs/latency_network/plan2_<job_id>/checkpoint_200000/
```

**Size:** ~200-300MB per checkpoint
**Total storage:** ~5-10GB for all checkpoints

---

## Troubleshooting

### Job doesn't start
```bash
squeue -u $USER  # Check if pending
scontrol show job <job_id>  # Check reason
```
- **Pending (Resources):** Waiting for GPU
- **Pending (Priority):** Queued behind other jobs

### Out of memory on GPU
Edit `scripts/slurm_train_maxtext.sh`:
```bash
per_device_batch_size=16  # Was 32
```

### Data not found
```bash
# Verify shards exist
ls data/sharded/train/*.parquet | wc -l  # Should be 180
ls data/sharded/test/*.parquet | wc -l   # Should be 20

# Re-shard if needed
python scripts/shard_parquet.py
```

### Training not progressing (loss not decreasing)
- Check logs for NaN/Inf values
- Verify learning rate schedule is active
- Check that real data is loading (not synthetic)

---

## Success Criteria

Training is successful if:

✅ Job completes without errors
✅ Loss decreases from ~10 to <2
✅ Checkpoints save every 5k steps
✅ Eval loss tracks train loss
✅ No NaN/Inf in metrics
✅ Throughput ~40-50k tokens/sec

---

## Quick Reference Commands

```bash
# Activate environment
source .venv/bin/activate

# Check dataset
python -c "import pyarrow.parquet as pq; t = pq.read_table('data/training_data.parquet'); print(f'{len(t):,} rows')"

# Verify sharding
ls data/sharded/train/*.parquet | wc -l && ls data/sharded/test/*.parquet | wc -l

# CPU smoke test
DECOUPLE_GCLOUD=TRUE python -m MaxText.train src/MaxText/configs/latency_network.yml hardware=cpu steps=10 per_device_batch_size=2

# Submit GPU job
sbatch scripts/slurm_train_maxtext.sh

# Monitor job
squeue -u $USER
tail -f logs/ping-llm-plan2-*.out

# Cancel job
scancel <job_id>
```

---

## Dataset Details

- **Total rows:** 200,000,842
- **Train split:** 180,000,758 rows (90%)
- **Test split:** 20,000,084 rows (10%)
- **Train shards:** 180 files (~1M rows each)
- **Test shards:** 20 files (~1M rows each)
- **Storage:** ~985MB sharded data

---

## Model Architecture (PLAN_2)

```yaml
Parameters: 95.54M (85M non-embedding)
Layers: 20 (deep for reasoning)
Embedding: 640
MLP: 2048 (3.2x ratio)
Attention: 10 heads × 64 dim
Vocab: 267 tokens (11 role + 256 byte)
Context: 1024 tokens
Position: RoPE (rotary embeddings)
```

---

## Training Hyperparameters

```yaml
Batch size: 32 (per device)
Steps: 200,000
Learning rate: 3.0e-4 (cosine schedule)
Warmup: 2,000 steps
Optimizer: Adam (β1=0.9, β2=0.999)
Weight decay: 0.01
Dropout: 0.1
```

---

## Next Steps After Training

1. **Evaluate final checkpoint:**
   ```bash
   python -m MaxText.train \
     src/MaxText/configs/latency_network.yml \
     run_name=eval_final \
     load_parameters_path=outputs/latency_network/plan2_<job_id>/checkpoint_200000 \
     eval_only=true
   ```

2. **Analyze metrics:**
   - TensorBoard: `tensorboard --logdir outputs/latency_network/plan2_<job_id>/tensorboard`
   - Wandb dashboard (if enabled)

3. **Test inference:**
   - Use trained model for predictions
   - Evaluate RTT prediction accuracy
   - Test inverse search (IP address inference)

---

**Good luck! 🚀**
