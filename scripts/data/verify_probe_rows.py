#!/usr/bin/env python3
"""Spot-check verification of ArrayRecord probe row output."""

import random
import hashlib
import pyarrow as pa
import pyarrow.ipc as ipc
import array_record.python.array_record_module as ar

DATA_DIR = "/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-llm/data/probe_rows"

EXPECTED_SCHEMA_FIELDS = {'event_time', 'src_addr', 'dst_addr', 'ip_version', 'rtt'}


def hash_src_addr(src_addr: str) -> int:
    hash_hex = hashlib.md5(src_addr.encode()).hexdigest()[:15]
    return int(hash_hex, 16)


def decode_entry(entry_bytes):
    """Decode an ArrayRecord entry back to its metadata + measurement table."""
    reader = ipc.open_stream(entry_bytes)
    batch = reader.read_all()
    src_id = batch.column('src_id')[0].as_py()
    n_measurements = batch.column('n_measurements')[0].as_py()
    first_ts = batch.column('first_timestamp')[0].as_py()
    last_ts = batch.column('last_timestamp')[0].as_py()
    time_span = batch.column('time_span_seconds')[0].as_py()
    meas_bytes = batch.column('measurements')[0].as_py()

    meas_reader = ipc.open_stream(meas_bytes)
    meas_table = meas_reader.read_all()
    return {
        'src_id': src_id,
        'n_measurements': n_measurements,
        'first_ts': first_ts,
        'last_ts': last_ts,
        'time_span_s': time_span,
        'meas_table': meas_table,
    }


def check_entry(entry_bytes, split_name, idx):
    """Run all checks on a single entry. Returns list of issues."""
    issues = []
    prefix = f"  [{split_name} #{idx}]"

    try:
        d = decode_entry(entry_bytes)
    except Exception as e:
        return [f"{prefix} FAILED to decode: {e}"]

    tbl = d['meas_table']

    # 1. Schema check
    actual_fields = set(tbl.schema.names)
    if actual_fields != EXPECTED_SCHEMA_FIELDS:
        issues.append(f"{prefix} Schema mismatch: {actual_fields}")

    # 2. n_measurements matches actual row count
    if len(tbl) != d['n_measurements']:
        issues.append(f"{prefix} n_measurements={d['n_measurements']} but table has {len(tbl)} rows")

    # 3. event_time sorted ascending
    times = tbl.column('event_time')
    for i in range(1, len(times)):
        if times[i].as_py() < times[i - 1].as_py():
            issues.append(f"{prefix} event_time NOT sorted at row {i}")
            break

    # 4. first/last timestamp match
    if len(tbl) > 0:
        actual_first = times[0].as_py()
        actual_last = times[len(times) - 1].as_py()
        if actual_first != d['first_ts']:
            issues.append(f"{prefix} first_timestamp mismatch: meta={d['first_ts']} actual={actual_first}")
        if actual_last != d['last_ts']:
            issues.append(f"{prefix} last_timestamp mismatch: meta={d['last_ts']} actual={actual_last}")

    # 5. time_span consistency
    if len(tbl) > 1:
        expected_span = (d['last_ts'] - d['first_ts']).total_seconds()
        if abs(expected_span - d['time_span_s']) > 0.01:
            issues.append(f"{prefix} time_span mismatch: meta={d['time_span_s']} computed={expected_span}")

    # 6. No nulls in critical columns
    for col_name in ['event_time', 'src_addr', 'rtt']:
        col = tbl.column(col_name)
        if col.null_count > 0:
            issues.append(f"{prefix} {col_name} has {col.null_count} nulls")

    # 7. Train/test split correctness
    src_addrs = tbl.column('src_addr')
    first_src = src_addrs[0].as_py()
    expected_train = hash_src_addr(first_src) % 10 < 9
    if split_name == 'train' and not expected_train:
        issues.append(f"{prefix} src_addr={first_src} should be in TEST (hash % 10 = {hash_src_addr(first_src) % 10})")
    if split_name == 'test' and expected_train:
        issues.append(f"{prefix} src_addr={first_src} should be in TRAIN (hash % 10 = {hash_src_addr(first_src) % 10})")

    # 8. All src_addrs in this entry have the same src_id
    unique_srcs = set(s.as_py() for s in src_addrs)
    if len(unique_srcs) > 1:
        issues.append(f"{prefix} Multiple src_addrs in one entry: {unique_srcs}")

    return issues


def main():
    N_SAMPLES = 20  # per split
    random.seed(42)

    all_issues = []

    for split_name in ['train', 'test']:
        path = f"{DATA_DIR}/{split_name}.arrayrecord"
        reader = ar.ArrayRecordReader(path)
        n = reader.num_records()
        print(f"\n{'='*60}")
        print(f"{split_name.upper()}: {n:,} records in {path}")
        print(f"{'='*60}")

        indices = sorted(random.sample(range(n), min(N_SAMPLES, n)))

        # Read sampled entries
        entries = reader.read(indices)

        time_spans = []
        measurement_counts = []

        for sample_idx, (record_idx, entry) in enumerate(zip(indices, entries)):
            issues = check_entry(entry, split_name, record_idx)
            all_issues.extend(issues)

            d = decode_entry(entry)
            time_spans.append(d['time_span_s'])
            measurement_counts.append(d['n_measurements'])

            tbl = d['meas_table']
            src = tbl.column('src_addr')[0].as_py()
            print(f"  Record {record_idx:>7,}: src={src:<16s} "
                  f"n_meas={d['n_measurements']:>9,}  "
                  f"span={d['time_span_s']/86400:>6.1f}d  "
                  f"first={d['first_ts']}  "
                  f"{'OK' if not issues else 'ISSUES: ' + '; '.join(issues)}")

        reader.close()

        print(f"\n  Summary ({split_name}):")
        print(f"    Samples checked: {len(indices)}")
        print(f"    Measurement counts: min={min(measurement_counts):,} max={max(measurement_counts):,} "
              f"avg={sum(measurement_counts)/len(measurement_counts):,.0f}")
        print(f"    Time spans (days): min={min(time_spans)/86400:.1f} max={max(time_spans)/86400:.1f} "
              f"avg={sum(time_spans)/len(time_spans)/86400:.1f}")

    print(f"\n{'='*60}")
    if all_issues:
        print(f"ISSUES FOUND: {len(all_issues)}")
        for issue in all_issues:
            print(issue)
    else:
        print("ALL CHECKS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
