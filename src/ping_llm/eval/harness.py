"""
Evaluation harness: production-realistic baseline evaluation on raw parquet data.

Uses DuckDB for fast analytical queries over parquet shards.

Subcommands:
    scan      - Query all shards, build bidir IP set + per-pair mean RTT
    neighbors - Select 100 proximity-weighted neighbors per node
    extract   - Query shards again, split into train/test parquet
    train     - Train all baselines on training data
    predict   - Generate predictions on test data → observations.parquet
    run       - All of the above in sequence

Usage:
    python -m ping_llm.eval.harness scan --data-dir data/parquet_ping --eval-dir data/eval
    python -m ping_llm.eval.harness run  --data-dir data/parquet_ping --eval-dir data/eval
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd


def _parquet_glob(data_dir):
    return str(Path(data_dir) / "*.parquet")


def _detect_rtt_col(data_dir):
    con = duckdb.connect()
    cols = con.execute(
        f"SELECT name FROM parquet_schema('{_parquet_glob(data_dir)}') "
        "WHERE name IN ('rtt', 'rtt_avg') LIMIT 1"
    ).fetchone()
    if cols is None:
        raise ValueError(f"No rtt/rtt_avg column in {data_dir}")
    return cols[0]


# ── Step 1: scan ─────────────────────────────────────────────────────────────

def cmd_scan(args):
    t0 = time.time()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    glob = _parquet_glob(args.data_dir)
    rtt_col = _detect_rtt_col(args.data_dir)
    con = duckdb.connect()

    print(f"Scanning {glob} (rtt column: {rtt_col})...", flush=True)

    # Step 1a: find bidirectional IPs
    print("  Finding bidirectional IPs...", flush=True)
    bidir_df = con.execute(f"""
        WITH srcs AS (SELECT DISTINCT src_addr AS ip FROM '{glob}' WHERE {rtt_col} > 0),
             dsts AS (SELECT DISTINCT dst_addr AS ip FROM '{glob}' WHERE {rtt_col} > 0)
        SELECT ip FROM srcs INTERSECT SELECT ip FROM dsts
    """).fetchdf()
    bidir_ips = set(bidir_df["ip"].tolist())
    print(f"  Bidirectional IPs: {len(bidir_ips)} ({time.time()-t0:.0f}s)", flush=True)

    # Step 1b: per-pair mean RTT for bidir pairs + time range
    print("  Computing per-pair mean RTT...", flush=True)
    bidir_list = sorted(bidir_ips)
    con.execute("CREATE TEMPORARY TABLE bidir_ips (ip VARCHAR)")
    con.executemany("INSERT INTO bidir_ips VALUES (?)", [(ip,) for ip in bidir_list])

    pair_stats = con.execute(f"""
        SELECT src_addr, dst_addr,
               AVG({rtt_col}) AS mean_rtt,
               COUNT(*) AS cnt
        FROM '{glob}'
        WHERE {rtt_col} > 0
          AND src_addr IN (SELECT ip FROM bidir_ips)
          AND dst_addr IN (SELECT ip FROM bidir_ips)
        GROUP BY src_addr, dst_addr
    """).fetchdf()
    print(f"  Directed pairs: {len(pair_stats):,} ({time.time()-t0:.0f}s)", flush=True)

    # Time range
    time_range = con.execute(f"""
        SELECT MIN(event_time) AS t_min, MAX(event_time) AS t_max
        FROM '{glob}'
        WHERE {rtt_col} > 0
    """).fetchone()
    min_time, max_time = time_range
    midpoint = min_time + (max_time - min_time) / 2

    # Save
    pair_keys = (pair_stats["src_addr"] + "|" + pair_stats["dst_addr"]).values
    np.savez(
        eval_dir / "pass1_stats.npz",
        bidir_ips=np.array(bidir_list),
        pair_keys=pair_keys,
        pair_mean_rtt=pair_stats["mean_rtt"].values,
        midpoint=np.array(str(midpoint)),
        min_time=np.array(str(min_time)),
        max_time=np.array(str(max_time)),
        rtt_col=np.array(rtt_col),
    )

    print(f"\nScan complete ({time.time()-t0:.0f}s)")
    print(f"  Bidir IPs: {len(bidir_ips)}")
    print(f"  Directed pairs: {len(pair_stats):,}")
    print(f"  Time range: {min_time} → {max_time}")
    print(f"  Midpoint: {midpoint}")
    print(f"  → {eval_dir / 'pass1_stats.npz'}", flush=True)


# ── Step 2: neighbors ────────────────────────────────────────────────────────

def cmd_neighbors(args):
    eval_dir = Path(args.eval_dir)
    stats = np.load(eval_dir / "pass1_stats.npz", allow_pickle=True)

    bidir_ips = set(stats["bidir_ips"].tolist())
    pair_keys = stats["pair_keys"].tolist()
    pair_mean_rtt = stats["pair_mean_rtt"].tolist()

    node_peers = defaultdict(dict)
    for key, mean_rtt in zip(pair_keys, pair_mean_rtt):
        src, dst = key.split("|")
        if src in bidir_ips and dst in bidir_ips:
            node_peers[src][dst] = mean_rtt

    rng = np.random.RandomState(42)
    n_neighbors = args.n_neighbors
    neighbors = {}

    for node, peers in node_peers.items():
        peer_list = list(peers.keys())
        if len(peer_list) <= n_neighbors:
            neighbors[node] = peer_list
            continue
        weights = np.array([1.0 / max(peers[p], 0.001) for p in peer_list])
        weights /= weights.sum()
        chosen = rng.choice(len(peer_list), size=n_neighbors, replace=False, p=weights)
        neighbors[node] = [peer_list[i] for i in chosen]

    neighbor_pairs = set()
    for node, nbrs in neighbors.items():
        for nbr in nbrs:
            neighbor_pairs.add(frozenset({node, nbr}))

    neighbor_pairs_list = [sorted(list(p)) for p in neighbor_pairs if len(p) == 2]

    meta = {
        "n_neighbors": n_neighbors,
        "num_nodes": len(neighbors),
        "num_neighbor_pairs": len(neighbor_pairs_list),
        "avg_neighbors": float(np.mean([len(v) for v in neighbors.values()])),
        "neighbors": neighbors,
        "neighbor_pairs": neighbor_pairs_list,
    }

    with open(eval_dir / "neighbor_graph.json", "w") as f:
        json.dump(meta, f)

    print(f"Neighbor selection: {len(neighbors)} nodes, {len(neighbor_pairs_list)} unordered pairs")
    print(f"  Avg neighbors/node: {meta['avg_neighbors']:.1f}")
    print(f"  → {eval_dir / 'neighbor_graph.json'}", flush=True)


# ── Step 3: extract ──────────────────────────────────────────────────────────

def cmd_extract(args):
    t0 = time.time()
    eval_dir = Path(args.eval_dir)

    stats = np.load(eval_dir / "pass1_stats.npz", allow_pickle=True)
    bidir_list = stats["bidir_ips"].tolist()
    midpoint_str = str(stats["midpoint"])
    rtt_col = str(stats["rtt_col"])

    with open(eval_dir / "neighbor_graph.json") as f:
        ng = json.load(f)

    glob = _parquet_glob(args.data_dir)
    max_per_pair = args.max_per_pair
    con = duckdb.connect()

    # Load bidir IPs and neighbor pairs into DuckDB temp tables
    con.execute("CREATE TEMPORARY TABLE bidir_ips (ip VARCHAR)")
    con.executemany("INSERT INTO bidir_ips VALUES (?)", [(ip,) for ip in bidir_list])

    con.execute("CREATE TEMPORARY TABLE neighbor_pairs (lo VARCHAR, hi VARCHAR)")
    con.executemany("INSERT INTO neighbor_pairs VALUES (?, ?)",
                    [(p[0], p[1]) for p in ng["neighbor_pairs"]])

    print(f"Extracting train/test from {glob}...", flush=True)

    # Train: first half, neighbor pairs, capped per directed pair
    print("  Extracting train (first half, neighbor pairs)...", flush=True)
    con.execute(f"""
        COPY (
            SELECT src_addr, dst_addr, {rtt_col} AS rtt, event_time
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY src_addr, dst_addr ORDER BY event_time
                ) AS rn
                FROM '{glob}'
                WHERE {rtt_col} > 0
                  AND event_time <= TIMESTAMP '{midpoint_str}'
                  AND src_addr IN (SELECT ip FROM bidir_ips)
                  AND dst_addr IN (SELECT ip FROM bidir_ips)
                  AND LEAST(src_addr, dst_addr) || '|' || GREATEST(src_addr, dst_addr)
                      IN (SELECT lo || '|' || hi FROM neighbor_pairs)
            )
            WHERE rn <= {max_per_pair}
        ) TO '{eval_dir}/train_measurements.parquet' (FORMAT PARQUET)
    """)
    train_count = con.execute(
        f"SELECT COUNT(*) FROM '{eval_dir}/train_measurements.parquet'"
    ).fetchone()[0]
    print(f"  Train: {train_count:,} measurements ({time.time()-t0:.0f}s)", flush=True)

    # Test: second half, non-neighbor pairs
    print("  Extracting test (second half, non-neighbor pairs)...", flush=True)
    con.execute(f"""
        COPY (
            SELECT src_addr, dst_addr, {rtt_col} AS rtt, event_time
            FROM '{glob}'
            WHERE {rtt_col} > 0
              AND event_time > TIMESTAMP '{midpoint_str}'
              AND src_addr IN (SELECT ip FROM bidir_ips)
              AND dst_addr IN (SELECT ip FROM bidir_ips)
              AND LEAST(src_addr, dst_addr) || '|' || GREATEST(src_addr, dst_addr)
                  NOT IN (SELECT lo || '|' || hi FROM neighbor_pairs)
        ) TO '{eval_dir}/test_measurements.parquet' (FORMAT PARQUET)
    """)
    test_count = con.execute(
        f"SELECT COUNT(*) FROM '{eval_dir}/test_measurements.parquet'"
    ).fetchone()[0]
    print(f"  Test: {test_count:,} measurements ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nExtract complete ({time.time()-t0:.0f}s)")
    print(f"  → {eval_dir / 'train_measurements.parquet'}")
    print(f"  → {eval_dir / 'test_measurements.parquet'}", flush=True)


# ── Step 4: train ────────────────────────────────────────────────────────────

def cmd_train(args):
    t0 = time.time()
    eval_dir = Path(args.eval_dir)
    models_dir = eval_dir / "models"
    models_dir.mkdir(exist_ok=True)

    train_df = pd.read_parquet(eval_dir / "train_measurements.parquet")
    meas = list(zip(train_df["src_addr"], train_df["dst_addr"], train_df["rtt"]))
    print(f"Training on {len(meas):,} measurements...")

    global_median = float(train_df["rtt"].median())
    print(f"  Global median: {global_median:.2f} ms")

    from ping_llm.eval.mf_baseline import DMFSGD
    print("\n  Training DMFSGD (r=10, 25 epochs, L1)...")
    dmfsgd = DMFSGD(embed_dim=10, lr=0.01, reg=1.0)
    dmfsgd.train(meas, epochs=25, verbose=True)
    np.savez(models_dir / "dmfsgd.npz",
             X=dmfsgd.X, Y=dmfsgd.Y, scale=dmfsgd._scale,
             ip_to_idx=json.dumps(dmfsgd.ip_to_idx))

    from ping_llm.eval.mf_baseline import BiasedMF
    print("\n  Training BiasedMF (r=16, 10 epochs, L2 log-space)...")
    biased_mf = BiasedMF(embed_dim=16, lr=0.01, reg=0.1)
    biased_mf.train(meas, epochs=10, verbose=True)
    np.savez(models_dir / "biased_mf.npz",
             X=biased_mf.X, Y=biased_mf.Y,
             bias_src=biased_mf.bias_src, bias_dst=biased_mf.bias_dst,
             global_bias=biased_mf.global_bias,
             ip_to_idx=json.dumps(biased_mf.ip_to_idx))

    from ping_llm.eval.vivaldi import fit_vivaldi
    print("\n  Training Vivaldi (dim=4, 5 epochs)...")
    viv = fit_vivaldi(meas, dim=4, n_epochs=5)
    viv_ips = sorted(viv.keys())
    viv_coords = np.array([viv[ip][0] for ip in viv_ips])
    viv_heights = np.array([viv[ip][1] for ip in viv_ips])
    np.savez(models_dir / "vivaldi.npz",
             ips=np.array(viv_ips), coords=viv_coords, heights=viv_heights)

    meta = {
        "num_train_measurements": len(meas),
        "num_unique_ips": len(set(train_df["src_addr"]) | set(train_df["dst_addr"])),
        "global_median_rtt_ms": global_median,
        "train_time_sec": round(time.time() - t0, 1),
    }
    meta_path = eval_dir / "baselines_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = {**json.load(f), **meta}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete ({time.time()-t0:.0f}s)", flush=True)


# ── Step 5: predict ──────────────────────────────────────────────────────────

def cmd_predict(args):
    t0 = time.time()
    eval_dir = Path(args.eval_dir)

    test_df = pd.read_parquet(eval_dir / "test_measurements.parquet")
    print(f"Predicting on {len(test_df):,} test measurements...")

    with open(eval_dir / "baselines_meta.json") as f:
        meta = json.load(f)
    global_median = meta["global_median_rtt_ms"]

    from ping_llm.eval.mf_baseline import DMFSGD, BiasedMF
    from ping_llm.eval.vivaldi import predict_vivaldi
    models_dir = eval_dir / "models"

    dmfsgd_data = np.load(models_dir / "dmfsgd.npz", allow_pickle=True)
    dmfsgd = DMFSGD(embed_dim=dmfsgd_data["X"].shape[1])
    dmfsgd.X = dmfsgd_data["X"]
    dmfsgd.Y = dmfsgd_data["Y"]
    dmfsgd._scale = float(dmfsgd_data["scale"])
    dmfsgd.ip_to_idx = json.loads(str(dmfsgd_data["ip_to_idx"]))

    bmf_data = np.load(models_dir / "biased_mf.npz", allow_pickle=True)
    biased_mf = BiasedMF(embed_dim=bmf_data["X"].shape[1])
    biased_mf.X = bmf_data["X"]
    biased_mf.Y = bmf_data["Y"]
    biased_mf.bias_src = bmf_data["bias_src"]
    biased_mf.bias_dst = bmf_data["bias_dst"]
    biased_mf.global_bias = float(bmf_data["global_bias"])
    biased_mf.ip_to_idx = json.loads(str(bmf_data["ip_to_idx"]))

    viv_data = np.load(models_dir / "vivaldi.npz", allow_pickle=True)
    viv_ips = viv_data["ips"].tolist()
    viv_coords = viv_data["coords"]
    viv_heights = viv_data["heights"]
    viv_dict = {ip: (viv_coords[i], viv_heights[i], 0.0) for i, ip in enumerate(viv_ips)}

    dmfsgd_preds = []
    bmf_preds = []
    viv_preds = []
    for _, row in test_df.iterrows():
        s, d = row["src_addr"], row["dst_addr"]
        dmfsgd_preds.append(dmfsgd.predict_rtt(s, d) or global_median)
        bmf_preds.append(biased_mf.predict_rtt(s, d) or global_median)
        viv_preds.append(predict_vivaldi(viv_dict, s, d) or global_median)

    test_df = test_df.copy()
    test_df["actual_rtt_ms"] = test_df["rtt"]
    test_df["global_median_pred"] = global_median
    test_df["dmfsgd_pred"] = dmfsgd_preds
    test_df["biased_mf_pred"] = bmf_preds
    test_df["vivaldi_pred"] = viv_preds

    test_df = test_df.sort_values(["src_addr", "event_time"]).reset_index(drop=True)

    last_seen_preds = []
    ema_preds = []
    window_mean_preds = []
    prior_rtts_list = []

    for _, group in test_df.groupby("src_addr", sort=False):
        ema = None
        window = []
        for i, (_, row) in enumerate(group.iterrows()):
            prior_rtts_list.append(i)
            if i == 0:
                last_seen_preds.append(global_median)
                ema_preds.append(global_median)
                window_mean_preds.append(global_median)
                ema = row["rtt"]
            else:
                last_seen_preds.append(float(window[-1]))
                ema_preds.append(float(ema))
                window_mean_preds.append(float(np.mean(window[-3:])))
                ema = 0.3 * row["rtt"] + 0.7 * ema
            window.append(row["rtt"])

    test_df["last_seen_pred"] = last_seen_preds
    test_df["ema_pred"] = ema_preds
    test_df["window_mean_pred"] = window_mean_preds
    test_df["prior_rtts"] = prior_rtts_list

    obs_cols = [
        "src_addr", "dst_addr", "actual_rtt_ms", "event_time", "prior_rtts",
        "global_median_pred", "dmfsgd_pred", "biased_mf_pred", "vivaldi_pred",
        "last_seen_pred", "ema_pred", "window_mean_pred",
    ]
    obs = test_df[obs_cols]
    obs.to_parquet(eval_dir / "observations.parquet", index=False)

    meta["num_test_measurements"] = len(obs)
    meta["num_test_pairs"] = int(obs[["src_addr", "dst_addr"]].drop_duplicates().shape[0])
    with open(eval_dir / "baselines_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPredict complete ({time.time()-t0:.0f}s)")
    print(f"  {len(obs):,} observations → {eval_dir / 'observations.parquet'}", flush=True)


# ── run (all steps) ──────────────────────────────────────────────────────────

def cmd_run(args):
    cmd_scan(args)
    cmd_neighbors(args)
    cmd_extract(args)
    cmd_train(args)
    cmd_predict(args)
    print("\n=== Harness complete. Run analysis.py next. ===")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Eval harness with production-realistic train/test split")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in [("scan", cmd_scan), ("neighbors", cmd_neighbors),
                     ("extract", cmd_extract), ("train", cmd_train),
                     ("predict", cmd_predict), ("run", cmd_run)]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
        sp.add_argument("--eval-dir", default="data/eval")
        if name in ("scan", "extract", "run"):
            sp.add_argument("--data-dir", required=True)
        if name in ("neighbors", "run"):
            sp.add_argument("--n-neighbors", type=int, default=100)
        if name in ("extract", "run"):
            sp.add_argument("--max-per-pair", type=int, default=100)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
