# ping-llm

Decoder-only Transformer trained on RIPE Atlas network latency measurements.

## Quick Start

```bash
# Install
pip install -e .

# Train on Modal (95M model, ~$7, ~3h)
modal run scripts/train/modal_wrapper.py::run \
  --run-name my-run --steps 14000 --batch-size 32
```

## Architecture

- **Model**: GPT-style Transformer with RoPE, ReLU², logit softcap
- **Optimizer**: Muon (weight matrices) + AdamW (embeddings)
- **Data**: Probe-centric ArrayRecord format with grain pipeline
- **Tokenization**: Custom byte-level scheme for IP addresses, RTTs, timestamps

See [PLAN.md](PLAN.md) for full details and [NEXT_STEPS.md](NEXT_STEPS.md) for roadmap.

## Project Structure

```
src/ping_llm/          # Core package
  model.py             # GPT model
  train.py             # Training loop
  config.py            # Configuration
  muon.py              # Muon optimizer
  data/                # Data pipeline (grain + ArrayRecord)
scripts/
  train/modal_wrapper.py  # Modal deployment
  eval_*.py               # Evaluation scripts
  data/                   # Data preparation tools
```
