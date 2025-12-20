# Local Model Evaluation Guide

This guide explains how to download your trained checkpoint and run local evaluations.

## Step 1: Download Checkpoint from Modal

```bash
# Download checkpoint step 2000 (adjust as needed)
bash scripts/download_checkpoint.sh 2000 checkpoints/full_run

# Or with custom Modal volume name:
MODAL_VOLUME=my-volume bash scripts/download_checkpoint.sh 2000 checkpoints/full_run
```

This downloads the checkpoint to: `checkpoints/full_run/checkpoints/2000/`

## Step 2A: Evaluate Field Ordering Likelihood

This script tests how well the model learned different slices of the joint distribution by trying all possible orderings of fields.

```bash
python scripts/eval_ordering_likelihood.py \
    --checkpoint checkpoints/full_run/checkpoints/2000 \
    --data data/training_data.parquet \
    --num-samples 100
```

**What it does:**
- Samples 100 random measurements from training data
- For each measurement, creates all possible field orderings:
  - Without timestamp: 6 orderings (src→dst→rtt, src→rtt→dst, etc.)
  - With timestamp: 24 orderings (4! = 24 permutations)
- Measures model's log-likelihood for predicting the final tokens in each ordering
- Reports which orderings the model prefers

**Example output:**
```
WITHOUT TIMESTAMP:
  src → dst → rtt                Mean: -2.3456  Std: 1.2345  (n=100)
  src → rtt → dst                Mean: -3.4567  Std: 1.3456  (n=100)
  ...

WITH TIMESTAMP:
  src → dst → rtt → timestamp    Mean: -2.1234  Std: 1.1234  (n=100)
  ...

INTERPRETATION:
  - 'src → dst → rtt' tests P(rtt | src, dst) - predicting latency from endpoints
  - 'src → rtt → dst' tests P(dst | src, rtt) - predicting destination from source+latency
  - Higher likelihood = model learned that conditional better
```

## Step 2B: Evaluate with Live Pings (KL Divergence)

This script pings random IPs and compares the model's predicted latency distribution with real measurements.

**⚠️ Note:** This requires network access and will send ICMP pings. Start with small numbers for testing.

```bash
python scripts/eval_live_ping.py \
    --checkpoint checkpoints/full_run/checkpoints/2000 \
    --num-ips 10 \
    --pings-per-ip 20 \
    --model-samples 100 \
    --temperature 1.0
```

**What it does:**
- Generates 10 random public IPv4 addresses
- Pings each address 20 times to get empirical latency distribution
- For each address:
  - Conditions model on (src_ip, dst_ip)
  - Samples 100 latency predictions from model's 2-gram distribution
  - Computes KL divergence between real and predicted distributions
- Tests both WITH and WITHOUT timestamp conditioning

**Example output:**
```
[1/10] Testing 8.8.8.8...
  Pinging 20 times...
  Real pings: 18/20 successful
  Sampling model (no timestamp)...
  Sampling model (with timestamp)...
  KL divergence (no timestamp):   0.3456
  KL divergence (with timestamp):  0.2345

SUMMARY RESULTS:
KL Divergence (lower is better):
  WITHOUT timestamp:
    Mean: 0.4567
    Std:  0.1234

  WITH timestamp:
    Mean: 0.3456
    Std:  0.0987

✓ Model performs better WITH timestamp (lower KL)
```

## Tips

### Fast Testing
For quick tests, use smaller numbers:
```bash
# Ordering evaluation (quick test)
python scripts/eval_ordering_likelihood.py \
    --checkpoint checkpoints/full_run/checkpoints/2000 \
    --num-samples 10

# Live ping evaluation (quick test)
python scripts/eval_live_ping.py \
    --checkpoint checkpoints/full_run/checkpoints/2000 \
    --num-ips 5 \
    --pings-per-ip 10
```

### CPU vs GPU
These scripts default to CPU inference (for laptop use). If you have a GPU:
- Edit the scripts to change `hardware=cpu` to `hardware=gpu` in `load_model_and_params()`

### Understanding Results

**Ordering Likelihood:**
- Higher log-likelihood = better
- Compare orderings to see which conditionals the model learned best
- E.g., if "src → dst → rtt" has highest likelihood, model is best at P(RTT | src, dst)

**KL Divergence:**
- Lower KL = better match to reality
- KL = 0 means perfect match (impossible in practice)
- KL < 0.5 is generally good
- Compare with/without timestamp to see if temporal info helps

## Troubleshooting

**"Permission denied" for pings:**
- The script uses `ping` command (no root needed)
- If it fails, check network connectivity

**"Checkpoint not found":**
- Verify the checkpoint path exists
- Check Modal volume contents: `modal volume ls ping-llm outputs/latency_network/`

**Out of memory:**
- Reduce `--num-samples` or `--model-samples`
- The scripts are designed for CPU inference on a laptop

**Model predictions are all the same:**
- Check `--temperature` (try 1.0 for stochastic sampling)
- Verify checkpoint loaded correctly

## Next Steps

After running these evaluations, you can:
1. Compare different checkpoint steps to see training progress
2. Test on specific IP ranges (edit `generate_random_ipv4()`)
3. Analyze which field orderings work best for your use case
4. Export results for visualization (scripts print to stdout, pipe to file)
