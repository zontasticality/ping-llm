"""Modal diagnostic for timestamp violations - minimal output."""
import modal

app = modal.App("probe-rows-diagnostic")
image = modal.Image.debian_slim(python_version="3.12").pip_install("pyarrow", "array_record", "numpy")
volume = modal.Volume.from_name("ping-llm", create_if_missing=True)

@app.function(image=image, volumes={"/mnt": volume}, timeout=600, cpu=2, memory=8*1024)
def diagnose(path: str = "/mnt/data/probe_rows/train.arrayrecord", n: int = 100):
    import array_record.python.array_record_module as ar
    import pyarrow.ipc as ipc
    import numpy as np
    from collections import Counter

    def get_meta(rec):
        r = ipc.open_stream(rec)
        b = r.read_next_batch()
        r.close()
        return {
            'meas_bytes': b.column('measurements')[0].as_py(),
            'n': b.column('n_measurements')[0].as_py(),
            'span': b.column('time_span_seconds')[0].as_py(),
        }

    def get_meas(mb):
        r = ipc.open_stream(mb)
        t = r.read_all()
        r.close()
        return t.to_pylist()

    reader = ar.ArrayRecordReader(path)
    total = reader.num_records()
    indices = np.random.choice(total, min(n, total), replace=False) if total > n else range(total)

    v_counts = Counter()
    neg_spans = 0

    for idx in indices:
        meta = get_meta(reader.read([int(idx)])[0])
        meas = get_meas(meta['meas_bytes'])

        violations = sum(1 for i in range(len(meas)-1) if meas[i]['event_time'] > meas[i+1]['event_time'])
        v_counts[violations] += 1

        if meta['span'] < 0:
            neg_spans += 1

    reader.close()

    print(f"Samples: {len(indices)}/{total}")
    print(f"Violation distribution: {dict(sorted(v_counts.items()))}")
    print(f"Negative timespans: {neg_spans}")
    print(f"Most common violations: {v_counts.most_common(3)}")

    if v_counts.most_common(1)[0][0] > 0:
        pct = 100 * v_counts.most_common(1)[0][1] / len(indices)
        print(f"❌ {pct:.0f}% have {v_counts.most_common(1)[0][0]} violations - DATA NOT SORTED")
    else:
        print("✅ All samples ordered correctly")

@app.local_entrypoint()
def main():
    diagnose.remote()
