# Probe Row Inspection Guide

## Overview

After preprocessing your data into probe row format, you should **always** inspect the output to verify correctness. This guide explains what to check and how to interpret the results.

---

## Quick Start

### On Modal (After Preprocessing)

```bash
# Run comprehensive inspection on Modal
modal run scripts/data/modal_inspect_probe_rows.py
```

### Locally

```bash
# Quick check of train set
python scripts/data/quick_inspect_probe_rows.py data/probe_rows/train.arrayrecord

# Quick check of test set
python scripts/data/quick_inspect_probe_rows.py data/probe_rows/test.arrayrecord

# More samples
python scripts/data/quick_inspect_probe_rows.py data/probe_rows/train.arrayrecord --samples 10

# Quick mode (only scan first 1000 rows)
python scripts/data/quick_inspect_probe_rows.py data/probe_rows/train.arrayrecord --quick
```

---

## What Gets Checked

### 1. Basic Statistics ✅

The inspector reports:
- **Total rows**: Number of probe rows created
- **Row sizes**: Min/max/mean in MB (should be < 8MB)
- **Measurements per row**: Distribution of how many measurements in each row
- **Time spans**: How much time each row covers
- **Unique src_ids**: Number of unique probes

**What to look for:**
- ✓ Total rows should match expected probe count (or more if splitting occurred)
- ✓ Max row size should be close to but not exceed 8MB
- ✓ Measurement counts should look reasonable for your data

### 2. Split Probes ✂️

Large probes (>8MB) are split into multiple rows. The inspector shows:
- Number of split probes
- Maximum number of splits for any probe
- Examples of highly-split probes

**What to look for:**
- ✓ Split count should be reasonable (typically < 5% of probes)
- ✓ Max splits should be reasonable (typically < 10 rows per probe)
- ❌ If many probes are split, you may need to increase `max_row_size_mb`

**Example output:**
```
  ✂️  Split probes:     1,234 (2.5%)
      Max splits:      5

      Top 5 most-split probes:
        src_id 12345: 5 rows (indices: [0, 1, 2, 3, 4])
        src_id 67890: 4 rows (indices: [100, 101, 102, 103])
```

### 3. Data Quality Checks 🔍

The inspector verifies:

#### A. Timestamp Ordering
- Checks if measurements within each row are ordered by `event_time`
- Samples random rows and reports violations

**What to look for:**
- ✓ Should say: "Timestamp ordering: PASS (0 violations)"
- ❌ If violations found: Your input data may not be pre-ordered, rerun with `--no-assume-ordered`

#### B. src_addr Consistency
- Checks if all measurements in a row have the same `src_addr`
- This is critical for correctness

**What to look for:**
- ✓ Should say: "src_addr consistency: PASS"
- ❌ If violations found: **SERIOUS BUG** - preprocessing failed

### 4. Sample Rows 📋

Shows detailed information for sample rows including:
- First, middle, last, and largest rows
- Plus random samples

**For each sample, you'll see:**
```
Sample 1 - Row Index 0
────────────────────────────────────────────────────────────────────────────────
  src_id:            1234567890123456
  n_measurements:    1,543
  time_span:         86400.0 seconds (24.00 hours)
  first_timestamp:   2024-01-01 00:00:00
  last_timestamp:    2024-01-02 00:00:00
  row_size:          3.45 MB

  First 3 measurements:
    [0] 2024-01-01 00:00:00.123 | 1.2.3.4 -> 5.6.7.8 | RTT: 45.234ms | IPv4
    [1] 2024-01-01 00:01:32.456 | 1.2.3.4 -> 9.10.11.12 | RTT: 23.456ms | IPv4
    [2] 2024-01-01 00:03:45.789 | 1.2.3.4 -> 13.14.15.16 | RTT: 67.890ms | IPv4

  Last 3 measurements:
    [1540] 2024-01-01 23:55:12.345 | 1.2.3.4 -> 17.18.19.20 | RTT: 34.567ms | IPv4
    [1541] 2024-01-01 23:57:34.567 | 1.2.3.4 -> 21.22.23.24 | RTT: 78.901ms | IPv4
    [1542] 2024-01-02 00:00:00.000 | 1.2.3.4 -> 25.26.27.28 | RTT: 12.345ms | IPv4

  Quality checks:
    ✓ Timestamps are ordered
    ✓ All measurements have same src_addr
```

**What to look for:**
- ✓ Timestamps should increase monotonically
- ✓ All measurements should have the same src_addr (first column after timestamp)
- ✓ RTT values should be reasonable (< 1000ms typically)
- ✓ Destination addresses should vary
- ✓ Time spans should make sense for your data

---

## Common Issues and Solutions

### Issue 1: Timestamp Ordering Violations ❌

**Symptom:**
```
❌ Timestamp ordering: FAIL (15/100 samples have violations)
```

**Cause:** Input data is not pre-sorted by `(src_addr, event_time)`

**Solution:**
Rerun preprocessing with `--no-assume-ordered`:
```bash
python scripts/data/create_probe_rows_parallel_streaming.py \
  --input "data/*.parquet" \
  --output data/probe_rows \
  --no-assume-ordered
```

**Note:** This will use more memory but ensures correct ordering.

---

### Issue 2: src_addr Inconsistency ❌

**Symptom:**
```
❌ src_addr consistency: FAIL (5/100 samples have inconsistent src_addr)
```

**Cause:** **SERIOUS BUG** in preprocessing - this should never happen

**Solution:**
1. Report as a bug
2. Check your input data for corruption
3. Try rerunning preprocessing from scratch

---

### Issue 3: Many Split Probes ✂️

**Symptom:**
```
✂️  Split probes: 15,234 (25%)
    Max splits: 45
```

**Cause:** Many probes have > 8MB of measurements

**Solutions:**
1. **Increase max row size:**
   ```bash
   --max-row-size-mb 16
   ```
   (Note: This may affect training performance)

2. **Filter out extremely large probes** in preprocessing
3. **Accept the splits** - they're handled correctly

---

### Issue 4: Unrealistic Data Values

**Symptom:**
- RTT values are negative or > 10,000ms
- Timestamps are in the future or before epoch
- IP addresses look invalid

**Cause:** Data corruption or wrong input format

**Solution:**
1. Check your input parquet files
2. Verify the data extraction process
3. Add validation in preprocessing

---

## Interpreting Statistics

### Row Size Distribution

**Good:**
```
Min row size:    0.05 MB
Max row size:    7.95 MB
Mean row size:   2.34 MB
Median row size: 1.89 MB
```
- Max is close to 8MB limit
- Most rows are smaller (efficient)

**Bad:**
```
Min row size:    0.05 MB
Max row size:    15.23 MB  ← EXCEEDS 8MB LIMIT!
Mean row size:   8.12 MB
```
- Rows exceed 8MB (will cause issues)
- Need to reduce `max_row_size_mb` or fix splitting logic

### Measurement Distribution

**Good:**
```
Total measurements:     203,456,789
Min measurements/row:   1
Max measurements/row:   15,234
Mean measurements/row:  1,234.5
Median measurements/row: 892

Percentiles:
  P10:      123
  P50:      892
  P90:    3,456
  P99:   12,345
```
- Wide distribution is normal
- P99 shows most rows are reasonable

**Bad:**
```
Total measurements:     203,456,789
Min measurements/row:   1
Max measurements/row:   1,234,567  ← EXTREMELY LARGE
Mean measurements/row:  89,234.5
```
- Some probes are huge (will cause splits)
- Consider filtering or increasing max row size

### Time Span Distribution

**Good:**
```
Min time span:    60.0 seconds (0.02 hours)
Max time span:    604800.0 seconds (168.00 hours)  ← 1 week
Mean time span:   86400.0 seconds (24.00 hours)
```
- Reasonable time ranges
- Max of 1 week is typical for network data

**Bad:**
```
Min time span:    0.0 seconds  ← ALL MEASUREMENTS AT SAME TIME?
Max time span:    31536000.0 seconds (8760.00 hours)  ← 1 YEAR!
```
- Suspicious patterns
- Check data quality

---

## Validation Checklist

Use this checklist after preprocessing:

### Critical (Must Pass) ✅
- [ ] Timestamp ordering: PASS (0 violations)
- [ ] src_addr consistency: PASS
- [ ] Max row size < 8.5 MB
- [ ] No corrupted/null data in samples
- [ ] Measurements look realistic (valid IPs, RTTs, timestamps)

### Important (Should Check) ⚠️
- [ ] Split probe count is reasonable (< 10% typically)
- [ ] Time spans make sense for your data
- [ ] Measurement counts look reasonable
- [ ] Train/test split is correct (~90/10)

### Nice to Have 💡
- [ ] Uniform distribution of measurements across rows
- [ ] No extreme outliers in RTT values
- [ ] Coverage of expected IP ranges

---

## Example Good Output

```
==================================================================================
PROBE ROW INSPECTION REPORT
==================================================================================

TRAIN SET ANALYSIS
==================================================================================

📊 BASIC STATISTICS
────────────────────────────────────────────────────────────────────────────────
Total rows: 6,789,123

📈 SIZE DISTRIBUTION
────────────────────────────────────────────────────────────────────────────────
  Min row size:         52,340 bytes  (  0.05 MB)
  Max row size:      8,388,608 bytes  (  8.00 MB)
  Mean row size:     2,456,789 bytes  (  2.34 MB)
  Median row size:   1,987,654 bytes  (  1.89 MB)

📊 MEASUREMENT DISTRIBUTION
────────────────────────────────────────────────────────────────────────────────
  Total measurements:     183,456,789
  Min measurements/row:                 1
  Max measurements/row:            15,234
  Mean measurements/row:            1,234.5
  Median measurements/row:            892

🔑 PROBE (src_id) STATISTICS
────────────────────────────────────────────────────────────────────────────────
  Unique src_ids:                6,123,456
  Rows per unique src_id:              1.11
  ✂️  Split probes (>1 row):          234,567 (3.83%)
      Max rows per probe:                  8

🔍 DATA QUALITY CHECKS
────────────────────────────────────────────────────────────────────────────────
  ✓ Timestamp ordering: PASS (0 violations in 100 samples)
  ✓ src_addr consistency: PASS

✅ INSPECTION COMPLETE - ALL CHECKS PASSED!
```

---

## When to Be Concerned

### 🚨 Critical Issues (Must Fix)
- ❌ Timestamp ordering failures
- ❌ src_addr inconsistency
- ❌ Rows > 8.5 MB
- ❌ Null/corrupted data

### ⚠️ Warning Signs (Should Investigate)
- Split probe count > 20%
- Max splits > 15 rows per probe
- Time spans of 0 or > 1 year
- RTT values < 0 or > 5000ms
- Unexpected IP addresses

### 💡 Minor Issues (Acceptable)
- Some split probes (< 10%)
- Variation in measurement counts
- Small percentage of outlier RTTs

---

## Summary

**Run inspection after every preprocessing job!**

**Quick check:**
```bash
# On Modal
modal run scripts/data/modal_inspect_probe_rows.py

# Locally
python scripts/data/quick_inspect_probe_rows.py data/probe_rows/train.arrayrecord
```

**Look for:**
1. ✓ All quality checks PASS
2. ✓ Samples look realistic
3. ✓ Row sizes < 8MB
4. ✓ Reasonable split counts

If anything looks wrong, investigate before training! 🔍
