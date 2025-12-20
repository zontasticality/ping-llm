# Running Evaluations on Modal

Quick reference for running evaluation scripts on Modal with GPU acceleration.

## Prerequisites

1. **Modal account and CLI setup:**
   ```bash
   pip install modal
   modal setup
   ```

2. **Data uploaded to Modal volume:**
   - Your training/eval data should be in the `ping-llm` Modal volume
   - Default location: `/mnt/data/probe_rows/test.arrayrecord`
   - Upload if needed: `modal volume put ping-llm local_data/ /data/`

3. **Checkpoint available in Modal volume:**
   - Training creates checkpoints in `/mnt/outputs/latency_network/`
   - Or use the param_only_checkpoint

## Evaluation Scripts

### 1. Next-Token Predictions (NEW!)

Shows predicted vs actual tokens with color-coded output.

```bash
# Basic usage (uses default checkpoint and data)
modal run scripts/eval_next_token_predictions.py::eval_on_modal

# With custom parameters
modal run scripts/eval_next_token_predictions.py::eval_on_modal \
    --checkpoint-path /mnt/outputs/latency_network/full_run/full_run/checkpoints/2000/items \
    --data-file probe_rows/test.arrayrecord \
    --num-sequences 10 \
    --max-length 150 \
    --seed 42
```

**Parameters:**
- `--checkpoint-path`: Path to checkpoint items directory (default: param_only_checkpoint)
- `--data-file`: Data file within `/mnt/data/` (default: `probe_rows/test.arrayrecord`)
- `--num-sequences`: Number of sequences to evaluate (default: 5)
- `--max-length`: Max sequence length to evaluate (default: 100)
- `--seed`: Random seed (default: 42)

**Output:**
```
SEQUENCE 1/10
✓ Pos   0: Actual=SRC_IPV4        | Predicted=SRC_IPV4
✓ Pos   1: Actual=Byte(0xC0/192)  | Predicted=Byte(0xC0/192)
✗ Pos   2: Actual=Byte(0xA8/168)  | Predicted=Byte(0xA9/169)
...
Accuracy: 87.5% (77/88 correct)
```

### 2. Field Ordering Likelihood

Tests how different field orderings affect model predictions.

```bash
modal run scripts/eval_ordering_likelihood.py::eval_on_modal \
    --num-samples 100 \
    --max-new-tokens 20
```

### 3. Live Ping Evaluation (KL Divergence)

Compares model predictions with real ping measurements.

```bash
modal run scripts/eval_live_ping.py::eval_on_modal \
    --num-ips 10 \
    --pings-per-ip 20
```

## Tips for Modal Usage

### View Running Jobs
```bash
modal app list
```

### Stream Logs
Modal automatically streams logs when you use `modal run`.

### Use Different Checkpoints
```bash
# List available checkpoints
modal volume ls ping-llm outputs/latency_network/full_run/full_run/checkpoints/

# Use a specific checkpoint
modal run scripts/eval_next_token_predictions.py::eval_on_modal \
    --checkpoint-path /mnt/outputs/latency_network/full_run/full_run/checkpoints/5000/items
```

### Compare Checkpoints Over Time

Run evaluations on multiple checkpoints to track training progress:

```bash
# Early training
modal run scripts/eval_next_token_predictions.py::eval_on_modal \
    --checkpoint-path /mnt/outputs/latency_network/full_run/full_run/checkpoints/1000/items \
    --num-sequences 20 > eval_step1000.txt

# Mid training
modal run scripts/eval_next_token_predictions.py::eval_on_modal \
    --checkpoint-path /mnt/outputs/latency_network/full_run/full_run/checkpoints/3000/items \
    --num-sequences 20 > eval_step3000.txt

# Late training
modal run scripts/eval_next_token_predictions.py::eval_on_modal \
    --checkpoint-path /mnt/outputs/latency_network/full_run/full_run/checkpoints/5000/items \
    --num-sequences 20 > eval_step5000.txt

# Compare accuracy improvements
grep "Average accuracy" eval_step*.txt
```

## Resource Usage

All evaluation scripts use:
- **GPU:** A100 (can be changed to A10G or T4 for cheaper runs)
- **CPU:** 4 cores
- **Memory:** Auto-scaled based on model size
- **Timeout:** 2 hours (can be extended if needed)

To use a cheaper GPU, edit the script:
```python
@app.function(
    gpu="A10G",  # Changed from "A100"
    ...
)
```

## Cost Estimation

Approximate costs per evaluation run:
- Next-token predictions (10 sequences): ~$0.10 - 0.20
- Field ordering (100 samples): ~$0.20 - 0.40
- Live ping (10 IPs): ~$0.15 - 0.30

Using A10G instead of A100 reduces costs by ~60%.

## Troubleshooting

**"Volume not found":**
```bash
# List volumes
modal volume list

# Create if needed
modal volume create ping-llm
```

**"Checkpoint not found":**
```bash
# Check checkpoint exists
modal volume ls ping-llm outputs/latency_network/

# Or use param_only_checkpoint
modal run scripts/eval_next_token_predictions.py::eval_on_modal \
    --checkpoint-path /mnt/outputs/latency_network/param_only_checkpoint/checkpoints/0/items
```

**"Data file not found":**
```bash
# Check data exists
modal volume ls ping-llm data/

# Upload data if needed
modal volume put ping-llm local_data/probe_rows/ /data/probe_rows/
```

**Out of memory:**
- Reduce `--num-sequences` or `--max-length`
- Use smaller checkpoint if available
- Try A100-80GB instead of A100-40GB: `gpu="A100-80GB"`
