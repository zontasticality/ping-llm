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

import duckdb
import numpy as np
import pandas as pd


def _parquet_glob(data_dir):
    return str(Path(data_dir) / "*.parquet")


def _sql_literal(value: str | Path) -> str:
    """Return a single-quoted SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_connect(args, eval_dir: Path):
    con = duckdb.connect()
    temp_dir = Path(args.duckdb_temp_dir) if args.duckdb_temp_dir else eval_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit={_sql_literal(args.duckdb_memory_limit)}")
    con.execute(f"SET temp_directory={_sql_literal(temp_dir)}")
    con.execute("SET preserve_insertion_order=false")
    if args.duckdb_threads:
        con.execute(f"SET threads={int(args.duckdb_threads)}")
    print(
        f"  DuckDB: memory_limit={args.duckdb_memory_limit}, "
        f"temp_directory={temp_dir}, threads={args.duckdb_threads or 'default'}",
        flush=True,
    )
    return con


def _detect_rtt_col(data_dir):
    con = duckdb.connect()
    cols = con.execute(
        f"SELECT name FROM parquet_schema('{_parquet_glob(data_dir)}') "
        "WHERE name IN ('rtt', 'rtt_avg') LIMIT 1"
    ).fetchone()
    if cols is None:
        raise ValueError(f"No rtt/rtt_avg column in {data_dir}")
    return cols[0]


def _load_scan_meta(eval_dir: Path) -> dict:
    with open(eval_dir / "scan_meta.json") as f:
        return json.load(f)


def _replace_output(path: Path):
    if path.exists():
        path.unlink()


def _parquet_count(con, path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({_sql_literal(path)})"
    ).fetchone()[0])


def _model_path(models_dir: Path, name: str) -> Path:
    return models_dir / f"{name}.npz"


def _save_mf_model(path: Path, model, **extra):
    np.savez(
        path,
        X=model.X,
        Y=model.Y,
        scale=model._scale,
        ip_to_idx=json.dumps(model.ip_to_idx),
        **extra,
    )


def _load_dmfsgd_model(path: Path, cls):
    data = np.load(path, allow_pickle=True)
    model = cls(embed_dim=data["X"].shape[1])
    model.X = data["X"]
    model.Y = data["Y"]
    model._scale = float(data["scale"])
    model.ip_to_idx = json.loads(str(data["ip_to_idx"]))
    return model


# ── Step 1: scan ─────────────────────────────────────────────────────────────

def cmd_scan(args):
    t0 = time.time()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    glob = _parquet_glob(args.data_dir)
    rtt_col = _detect_rtt_col(args.data_dir)
    con = _duckdb_connect(args, eval_dir)
    bidir_path = eval_dir / "bidir_ips.parquet"
    bidir_ids_path = eval_dir / "bidir_ip_ids.parquet"
    pair_stats_path = eval_dir / "pair_stats.parquet"
    _replace_output(bidir_path)
    _replace_output(bidir_ids_path)
    _replace_output(pair_stats_path)

    print(f"Scanning {glob} (rtt column: {rtt_col})...", flush=True)

    time_range = con.execute(f"""
        SELECT MIN(event_time) AS t_min, MAX(event_time) AS t_max
        FROM read_parquet({_sql_literal(glob)})
        WHERE {rtt_col} > 0
    """).fetchone()
    min_time, max_time = time_range
    midpoint = min_time + (max_time - min_time) / 2

    graph_window = args.graph_window
    graph_time_clause = (
        f"AND event_time <= TIMESTAMP '{midpoint}'"
        if graph_window == "train" else
        ""
    )
    graph_time_clause_p = (
        f"AND p.event_time <= TIMESTAMP '{midpoint}'"
        if graph_window == "train" else
        ""
    )
    print(
        f"  Time range: {min_time} → {max_time}; midpoint: {midpoint}; "
        f"graph_window={graph_window}",
        flush=True,
    )

    # Step 1a: find bidirectional IPs
    print("  Finding bidirectional IPs...", flush=True)
    con.execute(f"""
        COPY (
            WITH srcs AS (
                SELECT DISTINCT src_addr AS ip
                FROM read_parquet({_sql_literal(glob)})
                WHERE {rtt_col} > 0
                  {graph_time_clause}
            ),
            dsts AS (
                SELECT DISTINCT dst_addr AS ip
                FROM read_parquet({_sql_literal(glob)})
                WHERE {rtt_col} > 0
                  {graph_time_clause}
            )
            SELECT ip FROM srcs INTERSECT SELECT ip FROM dsts
        ) TO {_sql_literal(bidir_path)} (FORMAT PARQUET, CODEC 'ZSTD')
    """)
    bidir_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({_sql_literal(bidir_path)})"
    ).fetchone()[0]
    print(f"  Bidirectional IPs: {bidir_count:,} ({time.time()-t0:.0f}s)", flush=True)

    print("  Assigning compact IP IDs...", flush=True)
    con.execute(f"""
        COPY (
            SELECT ip,
                   ROW_NUMBER() OVER (ORDER BY ip) - 1 AS ip_id
            FROM read_parquet({_sql_literal(bidir_path)})
        ) TO {_sql_literal(bidir_ids_path)}
          (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 100000)
    """)

    # Step 1b: per-pair mean RTT for bidir pairs + time range
    print("  Computing per-pair mean RTT to parquet using compact IP IDs...", flush=True)
    con.execute(f"""
        COPY (
            WITH ip_ids AS (
                SELECT ip, ip_id
                FROM read_parquet({_sql_literal(bidir_ids_path)})
            ),
            agg AS (
                SELECT s.ip_id AS src_id,
                       d.ip_id AS dst_id,
                       SUM(p.{rtt_col}) AS sum_rtt,
                       COUNT(*) AS cnt
                FROM read_parquet({_sql_literal(glob)}) p
                JOIN ip_ids s ON p.src_addr = s.ip
                JOIN ip_ids d ON p.dst_addr = d.ip
                WHERE p.{rtt_col} > 0
                  {graph_time_clause_p}
                GROUP BY s.ip_id, d.ip_id
            )
            SELECT s.ip AS src_addr,
                   d.ip AS dst_addr,
                   agg.sum_rtt / agg.cnt AS mean_rtt,
                   agg.cnt
            FROM agg
            JOIN ip_ids s ON agg.src_id = s.ip_id
            JOIN ip_ids d ON agg.dst_id = d.ip_id
        ) TO {_sql_literal(pair_stats_path)}
          (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 100000)
    """)
    directed_pairs = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({_sql_literal(pair_stats_path)})"
    ).fetchone()[0]
    print(f"  Directed pairs: {directed_pairs:,} ({time.time()-t0:.0f}s)", flush=True)

    meta = {
        "data_dir": str(args.data_dir),
        "rtt_col": rtt_col,
        "graph_window": graph_window,
        "bidir_ips_path": str(bidir_path),
        "bidir_ip_ids_path": str(bidir_ids_path),
        "pair_stats_path": str(pair_stats_path),
        "num_bidir_ips": int(bidir_count),
        "num_directed_pairs": int(directed_pairs),
        "min_time": str(min_time),
        "max_time": str(max_time),
        "midpoint": str(midpoint),
        "scan_time_sec": round(time.time() - t0, 1),
    }
    with open(eval_dir / "scan_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nScan complete ({time.time()-t0:.0f}s)")
    print(f"  Bidir IPs: {bidir_count:,}")
    print(f"  Directed pairs: {directed_pairs:,}")
    print(f"  Time range: {min_time} → {max_time}")
    print(f"  Midpoint: {midpoint}")
    print(f"  → {eval_dir / 'scan_meta.json'}")
    print(f"  → {pair_stats_path}", flush=True)


# ── Step 2: neighbors ────────────────────────────────────────────────────────

def cmd_neighbors(args):
    eval_dir = Path(args.eval_dir)
    meta = _load_scan_meta(eval_dir)
    con = _duckdb_connect(args, eval_dir)
    pair_stats_path = Path(meta["pair_stats_path"])
    selected_path = eval_dir / "selected_neighbors.parquet"
    neighbor_pairs_path = eval_dir / "neighbor_pairs.parquet"
    _replace_output(selected_path)
    _replace_output(neighbor_pairs_path)

    print(f"Selecting {args.n_neighbors} RTT-weighted neighbors per node...", flush=True)
    con.execute(f"""
        COPY (
            SELECT src_addr, dst_addr, mean_rtt
            FROM (
                SELECT src_addr, dst_addr, mean_rtt,
                       ROW_NUMBER() OVER (
                           PARTITION BY src_addr
                           ORDER BY -LN(GREATEST(random(), 1e-12))
                                    * GREATEST(mean_rtt, 0.001)
                       ) AS rn
                FROM read_parquet({_sql_literal(pair_stats_path)})
                WHERE src_addr <> dst_addr
            )
            WHERE rn <= {int(args.n_neighbors)}
        ) TO {_sql_literal(selected_path)}
          (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 100000)
    """)

    con.execute(f"""
        COPY (
            SELECT DISTINCT
                   LEAST(src_addr, dst_addr) AS lo,
                   GREATEST(src_addr, dst_addr) AS hi
            FROM read_parquet({_sql_literal(selected_path)})
            WHERE src_addr <> dst_addr
        ) TO {_sql_literal(neighbor_pairs_path)}
          (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 100000)
    """)

    num_nodes, avg_neighbors = con.execute(f"""
        SELECT COUNT(*), AVG(n)
        FROM (
            SELECT src_addr, COUNT(*) AS n
            FROM read_parquet({_sql_literal(selected_path)})
            GROUP BY src_addr
        )
    """).fetchone()
    num_directed_neighbors = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({_sql_literal(selected_path)})"
    ).fetchone()[0]
    num_neighbor_pairs = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({_sql_literal(neighbor_pairs_path)})"
    ).fetchone()[0]

    neighbor_meta = {
        "n_neighbors": args.n_neighbors,
        "num_nodes": int(num_nodes),
        "num_directed_neighbors": int(num_directed_neighbors),
        "num_neighbor_pairs": int(num_neighbor_pairs),
        "avg_neighbors": float(avg_neighbors or 0.0),
        "selected_neighbors_path": str(selected_path),
        "neighbor_pairs_path": str(neighbor_pairs_path),
    }

    with open(eval_dir / "neighbor_graph.json", "w") as f:
        json.dump(neighbor_meta, f, indent=2)

    print(f"Neighbor selection: {num_nodes:,} nodes, {num_neighbor_pairs:,} unordered pairs")
    print(f"  Avg neighbors/node: {float(avg_neighbors or 0.0):.1f}")
    print(f"  → {neighbor_pairs_path}", flush=True)


# ── Step 3: extract ──────────────────────────────────────────────────────────

def cmd_extract(args):
    t0 = time.time()
    eval_dir = Path(args.eval_dir)

    meta = _load_scan_meta(eval_dir)
    bidir_path = Path(meta["bidir_ips_path"])
    midpoint_str = meta["midpoint"]
    rtt_col = meta["rtt_col"]

    with open(eval_dir / "neighbor_graph.json") as f:
        ng = json.load(f)

    glob = _parquet_glob(args.data_dir)
    max_train_per_pair = args.max_per_pair
    max_test_per_pair = args.max_test_per_pair
    con = _duckdb_connect(args, eval_dir)
    neighbor_pairs_path = Path(ng["neighbor_pairs_path"])
    train_path = eval_dir / "train_measurements.parquet"
    test_path = eval_dir / "test_measurements.parquet"
    test_candidates_path = eval_dir / "test_candidates.parquet"
    extract_split = args.extract_split
    if extract_split in ("both", "train"):
        _replace_output(train_path)
    if extract_split in ("both", "test"):
        _replace_output(test_path)
        _replace_output(test_candidates_path)

    print(f"Extracting train/test from {glob}...", flush=True)

    # Train: first half, neighbor pairs, capped per directed pair
    if extract_split in ("both", "train"):
        print("  Extracting train (first half, neighbor pairs)...", flush=True)
        train_limit = (
            f"ORDER BY hash(src_addr, dst_addr, event_time) LIMIT {int(args.max_train_observations)}"
            if args.max_train_observations > 0 else
            "ORDER BY src_addr, dst_addr, event_time"
        )
        con.execute(f"""
            COPY (
                SELECT src_addr, dst_addr, {rtt_col} AS rtt, event_time
                FROM (
                    SELECT p.src_addr, p.dst_addr, p.{rtt_col}, p.event_time,
                           ROW_NUMBER() OVER (
                        PARTITION BY src_addr, dst_addr ORDER BY event_time
                    ) AS rn
                    FROM read_parquet({_sql_literal(glob)}) p
                    SEMI JOIN read_parquet({_sql_literal(bidir_path)}) s
                        ON p.src_addr = s.ip
                    SEMI JOIN read_parquet({_sql_literal(bidir_path)}) d
                        ON p.dst_addr = d.ip
                    SEMI JOIN read_parquet({_sql_literal(neighbor_pairs_path)}) n
                        ON LEAST(p.src_addr, p.dst_addr) = n.lo
                       AND GREATEST(p.src_addr, p.dst_addr) = n.hi
                    WHERE p.{rtt_col} > 0
                      AND p.event_time <= TIMESTAMP '{midpoint_str}'
                )
                WHERE rn <= {int(max_train_per_pair)}
                {train_limit}
            ) TO {_sql_literal(train_path)}
              (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 100000)
        """)
        train_count = _parquet_count(con, train_path)
        print(f"  Train: {train_count:,} measurements ({time.time()-t0:.0f}s)", flush=True)
    else:
        train_count = _parquet_count(con, train_path)
        print(f"  Train: reusing {train_count:,} existing measurements", flush=True)

    # Test: second half, non-neighbor pairs, capped per directed pair and globally.
    test_candidates_count = None
    if extract_split in ("both", "test"):
        sample_rate = max(0.0, min(1.0, float(args.test_sample_rate)))
        sample_modulus = 1_000_000
        sample_threshold = int(sample_rate * sample_modulus)
        if sample_threshold <= 0:
            raise ValueError("--test-sample-rate must be > 0")
        sample_clause = (
            ""
            if sample_threshold >= sample_modulus else
            f"AND hash(p.src_addr, p.dst_addr, p.event_time) % {sample_modulus} < {sample_threshold}"
        )

        print(
            f"  Extracting sampled test candidates "
            f"(second half, non-neighbor pairs, sample_rate={sample_rate:g})...",
            flush=True,
        )
        con.execute(f"""
            COPY (
                SELECT p.src_addr, p.dst_addr, p.{rtt_col} AS rtt, p.event_time
                FROM read_parquet({_sql_literal(glob)}) p
                SEMI JOIN read_parquet({_sql_literal(bidir_path)}) s
                    ON p.src_addr = s.ip
                SEMI JOIN read_parquet({_sql_literal(bidir_path)}) d
                    ON p.dst_addr = d.ip
                ANTI JOIN read_parquet({_sql_literal(neighbor_pairs_path)}) n
                    ON LEAST(p.src_addr, p.dst_addr) = n.lo
                   AND GREATEST(p.src_addr, p.dst_addr) = n.hi
                WHERE p.{rtt_col} > 0
                  AND p.event_time > TIMESTAMP '{midpoint_str}'
                  {sample_clause}
            ) TO {_sql_literal(test_candidates_path)}
              (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 100000)
        """)
        test_candidates_count = _parquet_count(con, test_candidates_path)
        print(
            f"  Test candidates: {test_candidates_count:,} measurements "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

        print("  Applying test per-pair and global caps...", flush=True)
        test_limit = (
            f"ORDER BY hash(src_addr, dst_addr, event_time) LIMIT {int(args.max_test_observations)}"
            if args.max_test_observations > 0 else
            "ORDER BY src_addr, event_time"
        )
        con.execute(f"""
            COPY (
                SELECT src_addr, dst_addr, rtt, event_time
                FROM (
                    SELECT src_addr, dst_addr, rtt, event_time,
                           ROW_NUMBER() OVER (
                               PARTITION BY src_addr, dst_addr ORDER BY event_time
                           ) AS rn
                    FROM read_parquet({_sql_literal(test_candidates_path)})
                )
                WHERE rn <= {int(max_test_per_pair)}
                {test_limit}
            ) TO {_sql_literal(test_path)}
              (FORMAT PARQUET, CODEC 'ZSTD', ROW_GROUP_SIZE 100000)
        """)
        test_count = _parquet_count(con, test_path)
        print(f"  Test: {test_count:,} measurements ({time.time()-t0:.0f}s)", flush=True)
    else:
        test_count = _parquet_count(con, test_path)
        print(f"  Test: reusing {test_count:,} existing measurements", flush=True)

    extract_meta = {
        "num_train_measurements": int(train_count),
        "num_test_measurements": int(test_count),
        "num_test_candidates": (
            int(test_candidates_count) if test_candidates_count is not None else None
        ),
        "extract_split": extract_split,
        "max_train_per_pair": int(max_train_per_pair),
        "max_test_per_pair": int(max_test_per_pair),
        "max_train_observations": int(args.max_train_observations),
        "max_test_observations": int(args.max_test_observations),
        "test_sample_rate": float(args.test_sample_rate),
        "extract_time_sec": round(time.time() - t0, 1),
    }
    with open(eval_dir / "extract_meta.json", "w") as f:
        json.dump(extract_meta, f, indent=2)

    print(f"\nExtract complete ({time.time()-t0:.0f}s)")
    print(f"  → {train_path}")
    print(f"  → {test_path}", flush=True)


# ── Step 4: train ────────────────────────────────────────────────────────────

def cmd_train(args):
    t0 = time.time()
    eval_dir = Path(args.eval_dir)
    models_dir = eval_dir / "models"
    models_dir.mkdir(exist_ok=True)

    train_df = pd.read_parquet(eval_dir / "train_measurements.parquet")
    source_train_count = len(train_df)
    max_baseline_train = getattr(args, "max_baseline_train_observations", 0)
    if max_baseline_train > 0 and source_train_count > max_baseline_train:
        print(
            f"  Sampling {max_baseline_train:,}/{source_train_count:,} "
            "train measurements for baseline fitting...",
            flush=True,
        )
        train_df = train_df.sample(n=max_baseline_train, random_state=42)

    meas = list(zip(train_df["src_addr"], train_df["dst_addr"], train_df["rtt"]))
    meas_with_time = list(zip(
        train_df["src_addr"],
        train_df["dst_addr"],
        train_df["rtt"],
        train_df["event_time"],
    ))
    print(f"Training on {len(meas):,} measurements...", flush=True)

    global_median = float(train_df["rtt"].median())
    print(f"  Global median: {global_median:.2f} ms", flush=True)

    from ping_llm.eval.mf_baseline import DMFSGD
    print(
        f"\n  Training DMFSGD (r={args.dmfsgd_dim}, {args.dmfsgd_epochs} epochs, L1)...",
        flush=True,
    )
    dmfsgd = DMFSGD(
        embed_dim=args.dmfsgd_dim,
        lr=args.dmfsgd_lr,
        reg=args.dmfsgd_reg,
        scale_quantile=args.dmfsgd_scale_quantile,
    )
    dmfsgd.train(meas, epochs=args.dmfsgd_epochs, verbose=True)
    _save_mf_model(
        _model_path(models_dir, "dmfsgd"),
        dmfsgd,
        scale_quantile=dmfsgd.scale_quantile,
    )

    if args.time_baselines:
        print(
            f"\n  Training time-ordered DMFSGD "
            f"(r={args.dmfsgd_dim}, {args.dmfsgd_epochs} epochs, L1)...",
            flush=True,
        )
        dmfsgd_time = DMFSGD(
            embed_dim=args.dmfsgd_dim,
            lr=args.dmfsgd_lr,
            reg=args.dmfsgd_reg,
            scale_quantile=args.dmfsgd_scale_quantile,
        )
        dmfsgd_time.train(
            meas_with_time,
            epochs=args.dmfsgd_epochs,
            verbose=True,
            shuffle=False,
        )
        _save_mf_model(
            _model_path(models_dir, "dmfsgd_time"),
            dmfsgd_time,
            scale_quantile=dmfsgd_time.scale_quantile,
        )

    if args.paper_dmfsgd:
        from ping_llm.eval.mf_baseline import PaperDMFSGD
        print(
            f"\n  Training paper-style DMFSGD "
            f"(r={args.paper_dmfsgd_dim}, {args.paper_dmfsgd_epochs} epochs, "
            "L1 minibatch + line search)...",
            flush=True,
        )
        paper_dmfsgd = PaperDMFSGD(
            embed_dim=args.paper_dmfsgd_dim,
            reg=args.paper_dmfsgd_reg,
            eta_init=args.paper_dmfsgd_eta_init,
            line_search_steps=args.paper_dmfsgd_line_search_steps,
            line_search_delta=args.paper_dmfsgd_line_search_delta,
            neighbor_cap=args.paper_dmfsgd_neighbor_cap,
            scale_quantile=args.paper_dmfsgd_scale_quantile,
            use_decay=args.paper_dmfsgd_decay,
        )
        paper_dmfsgd.train(meas, epochs=args.paper_dmfsgd_epochs, verbose=True)
        _save_mf_model(
            _model_path(models_dir, "dmfsgd_paper"),
            paper_dmfsgd,
            scale_quantile=paper_dmfsgd.scale_quantile,
            reg=paper_dmfsgd.reg,
            eta_init=paper_dmfsgd.eta_init,
            line_search_steps=paper_dmfsgd.line_search_steps,
            line_search_delta=paper_dmfsgd.line_search_delta,
            neighbor_cap=paper_dmfsgd.neighbor_cap,
            use_decay=paper_dmfsgd.use_decay,
        )

        if args.time_baselines:
            print(
                f"\n  Training time-ordered paper-style DMFSGD "
                f"(r={args.paper_dmfsgd_dim}, {args.paper_dmfsgd_epochs} epochs, "
                "L1 minibatch + line search)...",
                flush=True,
            )
            paper_dmfsgd_time = PaperDMFSGD(
                embed_dim=args.paper_dmfsgd_dim,
                reg=args.paper_dmfsgd_reg,
                eta_init=args.paper_dmfsgd_eta_init,
                line_search_steps=args.paper_dmfsgd_line_search_steps,
                line_search_delta=args.paper_dmfsgd_line_search_delta,
                neighbor_cap=args.paper_dmfsgd_neighbor_cap,
                scale_quantile=args.paper_dmfsgd_scale_quantile,
                use_decay=args.paper_dmfsgd_decay,
            )
            paper_dmfsgd_time.train(
                meas_with_time,
                epochs=args.paper_dmfsgd_epochs,
                verbose=True,
            )
            _save_mf_model(
                _model_path(models_dir, "dmfsgd_paper_time"),
                paper_dmfsgd_time,
                scale_quantile=paper_dmfsgd_time.scale_quantile,
                reg=paper_dmfsgd_time.reg,
                eta_init=paper_dmfsgd_time.eta_init,
                line_search_steps=paper_dmfsgd_time.line_search_steps,
                line_search_delta=paper_dmfsgd_time.line_search_delta,
                neighbor_cap=paper_dmfsgd_time.neighbor_cap,
                use_decay=paper_dmfsgd_time.use_decay,
            )

    from ping_llm.eval.mf_baseline import BiasedMF
    print(
        f"\n  Training BiasedMF (r={args.biased_mf_dim}, "
        f"{args.biased_mf_epochs} epochs, L2 log-space)...",
        flush=True,
    )
    biased_mf = BiasedMF(
        embed_dim=args.biased_mf_dim,
        lr=args.biased_mf_lr,
        reg=args.biased_mf_reg,
    )
    biased_mf.train(meas, epochs=args.biased_mf_epochs, verbose=True)
    np.savez(models_dir / "biased_mf.npz",
             X=biased_mf.X, Y=biased_mf.Y,
             bias_src=biased_mf.bias_src, bias_dst=biased_mf.bias_dst,
             global_bias=biased_mf.global_bias,
             ip_to_idx=json.dumps(biased_mf.ip_to_idx))

    from ping_llm.eval.vivaldi import fit_vivaldi
    print(f"\n  Training Vivaldi (dim=4, {args.vivaldi_epochs} epochs)...", flush=True)
    viv = fit_vivaldi(meas, dim=4, n_epochs=args.vivaldi_epochs)
    viv_ips = sorted(viv.keys())
    viv_coords = np.array([viv[ip][0] for ip in viv_ips])
    viv_heights = np.array([viv[ip][1] for ip in viv_ips])
    np.savez(models_dir / "vivaldi.npz",
             ips=np.array(viv_ips), coords=viv_coords, heights=viv_heights)

    if args.time_baselines:
        print(
            f"\n  Training time-ordered Vivaldi "
            f"(dim=4, {args.vivaldi_epochs} epochs)...",
            flush=True,
        )
        viv_time = fit_vivaldi(
            meas_with_time,
            dim=4,
            n_epochs=args.vivaldi_epochs,
            shuffle=False,
        )
        viv_time_ips = sorted(viv_time.keys())
        viv_time_coords = np.array([viv_time[ip][0] for ip in viv_time_ips])
        viv_time_heights = np.array([viv_time[ip][1] for ip in viv_time_ips])
        np.savez(
            models_dir / "vivaldi_time.npz",
            ips=np.array(viv_time_ips),
            coords=viv_time_coords,
            heights=viv_time_heights,
        )

    meta = {
        "num_train_measurements": len(meas),
        "num_source_train_measurements": int(source_train_count),
        "num_unique_ips": len(set(train_df["src_addr"]) | set(train_df["dst_addr"])),
        "global_median_rtt_ms": global_median,
        "dmfsgd_epochs": int(args.dmfsgd_epochs),
        "dmfsgd_dim": int(args.dmfsgd_dim),
        "dmfsgd_lr": float(args.dmfsgd_lr),
        "dmfsgd_reg": float(args.dmfsgd_reg),
        "dmfsgd_scale_quantile": float(args.dmfsgd_scale_quantile),
        "time_baselines": bool(args.time_baselines),
        "paper_dmfsgd": bool(args.paper_dmfsgd),
        "paper_dmfsgd_epochs": int(args.paper_dmfsgd_epochs),
        "paper_dmfsgd_dim": int(args.paper_dmfsgd_dim),
        "paper_dmfsgd_reg": float(args.paper_dmfsgd_reg),
        "paper_dmfsgd_eta_init": float(args.paper_dmfsgd_eta_init),
        "paper_dmfsgd_line_search_steps": int(args.paper_dmfsgd_line_search_steps),
        "paper_dmfsgd_line_search_delta": float(args.paper_dmfsgd_line_search_delta),
        "paper_dmfsgd_neighbor_cap": int(args.paper_dmfsgd_neighbor_cap),
        "paper_dmfsgd_scale_quantile": float(args.paper_dmfsgd_scale_quantile),
        "paper_dmfsgd_decay": bool(args.paper_dmfsgd_decay),
        "biased_mf_epochs": int(args.biased_mf_epochs),
        "biased_mf_dim": int(args.biased_mf_dim),
        "biased_mf_lr": float(args.biased_mf_lr),
        "biased_mf_reg": float(args.biased_mf_reg),
        "vivaldi_epochs": int(args.vivaldi_epochs),
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

    from ping_llm.eval.mf_baseline import DMFSGD, PaperDMFSGD, BiasedMF
    from ping_llm.eval.vivaldi import predict_vivaldi
    models_dir = eval_dir / "models"

    dmfsgd = _load_dmfsgd_model(_model_path(models_dir, "dmfsgd"), DMFSGD)

    dmfsgd_time = None
    dmfsgd_time_path = _model_path(models_dir, "dmfsgd_time")
    if dmfsgd_time_path.exists():
        dmfsgd_time = _load_dmfsgd_model(dmfsgd_time_path, DMFSGD)

    paper_dmfsgd = None
    paper_path = _model_path(models_dir, "dmfsgd_paper")
    if paper_path.exists():
        paper_dmfsgd = _load_dmfsgd_model(paper_path, PaperDMFSGD)

    paper_dmfsgd_time = None
    paper_time_path = _model_path(models_dir, "dmfsgd_paper_time")
    if paper_time_path.exists():
        paper_dmfsgd_time = _load_dmfsgd_model(paper_time_path, PaperDMFSGD)

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

    viv_time_dict = None
    viv_time_path = _model_path(models_dir, "vivaldi_time")
    if viv_time_path.exists():
        viv_time_data = np.load(viv_time_path, allow_pickle=True)
        viv_time_ips = viv_time_data["ips"].tolist()
        viv_time_coords = viv_time_data["coords"]
        viv_time_heights = viv_time_data["heights"]
        viv_time_dict = {
            ip: (viv_time_coords[i], viv_time_heights[i], 0.0)
            for i, ip in enumerate(viv_time_ips)
        }

    dmfsgd_preds = []
    dmfsgd_time_preds = []
    paper_dmfsgd_preds = []
    paper_dmfsgd_time_preds = []
    bmf_preds = []
    viv_preds = []
    viv_time_preds = []
    for _, row in test_df.iterrows():
        s, d = row["src_addr"], row["dst_addr"]
        dmfsgd_preds.append(dmfsgd.predict_rtt(s, d) or global_median)
        if dmfsgd_time is not None:
            dmfsgd_time_preds.append(dmfsgd_time.predict_rtt(s, d) or global_median)
        if paper_dmfsgd is not None:
            paper_dmfsgd_preds.append(paper_dmfsgd.predict_rtt(s, d) or global_median)
        if paper_dmfsgd_time is not None:
            paper_dmfsgd_time_preds.append(
                paper_dmfsgd_time.predict_rtt(s, d) or global_median
            )
        bmf_preds.append(biased_mf.predict_rtt(s, d) or global_median)
        viv_preds.append(predict_vivaldi(viv_dict, s, d) or global_median)
        if viv_time_dict is not None:
            viv_time_preds.append(predict_vivaldi(viv_time_dict, s, d) or global_median)

    test_df = test_df.copy()
    test_df["actual_rtt_ms"] = test_df["rtt"]
    test_df["global_median_pred"] = global_median
    test_df["dmfsgd_pred"] = dmfsgd_preds
    if dmfsgd_time is not None:
        test_df["dmfsgd_time_pred"] = dmfsgd_time_preds
    if paper_dmfsgd is not None:
        test_df["dmfsgd_paper_pred"] = paper_dmfsgd_preds
    if paper_dmfsgd_time is not None:
        test_df["dmfsgd_paper_time_pred"] = paper_dmfsgd_time_preds
    test_df["biased_mf_pred"] = bmf_preds
    test_df["vivaldi_pred"] = viv_preds
    if viv_time_dict is not None:
        test_df["vivaldi_time_pred"] = viv_time_preds

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
    test_df["obs_id"] = np.arange(len(test_df), dtype=np.int64)

    obs_cols = [
        "obs_id", "src_addr", "dst_addr", "actual_rtt_ms", "event_time", "prior_rtts",
        "global_median_pred", "dmfsgd_pred", "biased_mf_pred", "vivaldi_pred",
        "last_seen_pred", "ema_pred", "window_mean_pred",
    ]
    if "dmfsgd_time_pred" in test_df.columns:
        obs_cols.insert(obs_cols.index("biased_mf_pred"), "dmfsgd_time_pred")
    if "dmfsgd_paper_pred" in test_df.columns:
        obs_cols.insert(obs_cols.index("biased_mf_pred"), "dmfsgd_paper_pred")
    if "dmfsgd_paper_time_pred" in test_df.columns:
        obs_cols.insert(obs_cols.index("biased_mf_pred"), "dmfsgd_paper_time_pred")
    if "vivaldi_time_pred" in test_df.columns:
        obs_cols.insert(obs_cols.index("last_seen_pred"), "vivaldi_time_pred")
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
        sp.add_argument("--duckdb-memory-limit", default="16GB")
        sp.add_argument("--duckdb-temp-dir", default=None)
        sp.add_argument("--duckdb-threads", type=int, default=0)
        if name in ("scan", "extract", "run"):
            sp.add_argument("--data-dir", required=True)
        if name in ("scan", "run"):
            sp.add_argument("--graph-window", choices=("all", "train"), default="all",
                            help="Rows used to build bidir IPs and pair stats")
        if name in ("neighbors", "run"):
            sp.add_argument("--n-neighbors", type=int, default=100)
        if name in ("extract", "run"):
            sp.add_argument("--max-per-pair", type=int, default=100)
            sp.add_argument("--max-test-per-pair", type=int, default=10)
            sp.add_argument("--max-train-observations", type=int, default=0,
                            help="Optional global cap after per-pair train cap; 0 disables")
            sp.add_argument("--max-test-observations", type=int, default=1_000_000,
                            help="Optional global cap after per-pair test cap; 0 disables")
            sp.add_argument("--test-sample-rate", type=float, default=0.002,
                            help="Deterministic row sample rate before test per-pair window")
            sp.add_argument("--extract-split", choices=("both", "train", "test"), default="both",
                            help="Extract both splits or rerun only one split")
        if name in ("train", "run"):
            sp.add_argument("--max-baseline-train-observations", type=int, default=500_000,
                            help="Deterministic train subsample for pure-Python baselines; 0 disables")
            sp.add_argument("--dmfsgd-epochs", type=int, default=8)
            sp.add_argument("--dmfsgd-dim", type=int, default=10)
            sp.add_argument("--dmfsgd-lr", type=float, default=0.02)
            sp.add_argument("--dmfsgd-reg", type=float, default=0.001)
            sp.add_argument("--dmfsgd-scale-quantile", type=float, default=0.99)
            sp.add_argument("--time-baselines", action=argparse.BooleanOptionalAction,
                            default=True)
            sp.add_argument("--paper-dmfsgd", action=argparse.BooleanOptionalAction,
                            default=True)
            sp.add_argument("--paper-dmfsgd-epochs", type=int, default=3)
            sp.add_argument("--paper-dmfsgd-dim", type=int, default=10)
            sp.add_argument("--paper-dmfsgd-reg", type=float, default=1.0)
            sp.add_argument("--paper-dmfsgd-eta-init", type=float, default=0.01)
            sp.add_argument("--paper-dmfsgd-line-search-steps", type=int, default=8)
            sp.add_argument("--paper-dmfsgd-line-search-delta", type=float, default=1e-6)
            sp.add_argument("--paper-dmfsgd-neighbor-cap", type=int, default=32)
            sp.add_argument("--paper-dmfsgd-scale-quantile", type=float, default=1.0)
            sp.add_argument("--paper-dmfsgd-decay", action=argparse.BooleanOptionalAction,
                            default=True)
            sp.add_argument("--biased-mf-epochs", type=int, default=10)
            sp.add_argument("--biased-mf-dim", type=int, default=64)
            sp.add_argument("--biased-mf-lr", type=float, default=0.02)
            sp.add_argument("--biased-mf-reg", type=float, default=0.001)
            sp.add_argument("--vivaldi-epochs", type=int, default=3)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
