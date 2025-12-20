# Troubleshooting

Common issues and solutions for the network training pipeline.

## Documentation

- **[Performance](performance.md)** - Performance optimization and profiling
- **[OOM Issues](oom-issues.md)** - Out of memory troubleshooting
- **[Inspection](inspection.md)** - Data inspection and debugging

## Quick Diagnostics

### Check Data Pipeline Performance

Enable Grain debug mode in config:
```yaml
grain_debug_mode: true
```

This will log execution metrics every 60 seconds showing bottlenecks.

### Inspect ArrayRecord Data

```bash
python scripts/data/inspect_probe_rows.py data/probe_rows/train.arrayrecord
```

### Check Padding Distribution

```bash
python scripts/data/analyze_padding.py --data-path data/probe_rows/train.arrayrecord
```

### Monitor Training

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check training logs
tail -f outputs/*/logs/train.log
```

## Common Issues

### Low Throughput
- Check grain_worker_count (increase to 8-16)
- Enable multiprocessing in pipeline
- Verify data is on local SSD (not network mount)

### OOM Errors
- Reduce batch_size
- Reduce grain_worker_count
- Reduce grain_ram_budget_mb

### High Step Variance
- Check padding percentage
- Verify tokenization is deterministic
- Look for data skew in probe sizes
