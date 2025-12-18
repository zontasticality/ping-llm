# Modal Preprocessing Cheat Sheet

## One-Line Command

```bash
modal run scripts/data/modal_run_streaming.py
```

**That's it!** Takes 20-25 minutes.

---

## Common Commands

### Deploy and Run
```bash
cd /home/zyansheep/Projects/ping-llm
modal run scripts/data/modal_run_streaming.py
```

### Check Volume Contents
```bash
modal volume ls ping-llm-data data/probe_rows
```

### View Logs
```bash
modal app logs probe-rows-streaming
```

### Download Results
```bash
modal volume get ping-llm-data \
  data/probe_rows/train.arrayrecord \
  ./train.arrayrecord
```

---

## Quick Configs

### Default (Recommended)
- 8 CPU cores
- 32GB RAM
- 28GB DuckDB limit
- 8 workers
- **Cost: ~$0.80**
- **Time: 20-25 min**

### Fast (Double Speed)
Edit script:
```python
cpu=16.0, memory=65536
workers=16, memory_limit_gb=60
```
- **Cost: ~$1.60**
- **Time: 15-18 min**

### Cheap (Half Cost)
Edit script:
```python
cpu=4.0, memory=16384
workers=4, memory_limit_gb=12
```
- **Cost: ~$0.40**
- **Time: 35-45 min**

---

## Expected Output

```
Train: ~6-7M rows, 180M measurements
Test: ~600-700K rows, 20M measurements
Files:
  /mnt/data/probe_rows/train.arrayrecord
  /mnt/data/probe_rows/test.arrayrecord
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| OOM | Lower `memory_limit_gb` |
| Timeout | Increase `timeout` |
| File not found | Check `/mnt/data/` path |
| Slow | Increase `cpu` and `workers` |

---

## Full Guide

See `MODAL_DEPLOYMENT_GUIDE.md` for detailed instructions.
