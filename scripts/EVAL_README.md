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

## Step 2B: Evaluate Next-Token Predictions (Pretty-Printed)

This script shows exactly what the model predicts vs actual tokens at each position, helping you understand where the model succeeds and fails.

**Local CPU:**
```bash
python scripts/eval_next_token_predictions.py \
    --checkpoint checkpoints/full_run/checkpoints/2000 \
    --data data/probe_rows/test.arrayrecord \
    --num-sequences 5 \
    --max-length 100
```

**Modal GPU (faster):**
```bash
modal run scripts/eval_next_token_predictions.py::eval_on_modal \
    --num-sequences 10 \
    --max-length 150
```

**What it does:**
- Loads evaluation sequences from arrayrecord data
- For each position in the sequence, predicts the next token
- Shows predicted vs actual tokens in color-coded format
- Reports accuracy metrics

**Example output:**
```
SEQUENCE 1/5
Length: 89 tokens

Sequence: MEASUREMENT_START SRC_IPV4 Byte(0xC0/192) Byte(0xA8/168) ...

Evaluating next-token predictions...

Accuracy: 87.5% (77/88 correct)

First 20 predictions:
  ✓ Pos   0: Actual=SRC_IPV4             | Predicted=SRC_IPV4
  ✓ Pos   1: Actual=Byte(0xC0/192)       | Predicted=Byte(0xC0/192)
  ✗ Pos   2: Actual=Byte(0xA8/168)       | Predicted=Byte(0xA9/169)
  ✓ Pos   3: Actual=Byte(0x01/  1)       | Predicted=Byte(0x01/  1)
  ...

SUMMARY
Sequences evaluated: 5
Average accuracy: 85.3%
Min accuracy: 78.9%
Max accuracy: 91.2%
```

**Benefits:**
- **Track training progress:** Run on different checkpoints to see improvement
- **Identify failure modes:** See which token types (IPs, RTTs, timestamps) are hard to predict
- **Debug model behavior:** Understand exactly where predictions go wrong
- **Validate field shuffling:** Confirm model isn't just memorizing positions

**Quick demo (no model required):**
```bash
python scripts/test_eval_format.py
```

## Step 2C: Evaluate with Live Pings (KL Divergence)

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
