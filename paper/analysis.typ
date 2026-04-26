#set page(margin: 2.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 16pt, weight: "bold")[RTT Prediction Evaluation Report]
  #v(0.5em)
  #text(size: 11pt)[Ping-LLM Baseline Comparison and Methodology]
  #v(1em)
]

= Evaluation Pipeline

The evaluation follows a three-stage architecture designed so that expensive baseline training (Stage 1) runs once per test dataset, while model evaluation (Stage 2) runs cheaply per checkpoint, and analysis (Stage 3) runs locally without a GPU.

/ Stage 1 --- Harness: Loads test sequences from ArrayRecord, extracts ground-truth RTT values at every prediction position, trains all baseline methods, and caches per-position predictions to `observations.parquet`.

/ Stage 2 --- Model Eval: Loads a model checkpoint, runs a single forward pass per test sequence, and extracts the model's top-1 RTT prediction at each position. Saves to `model_preds/{run_name}.parquet`.

/ Stage 3 --- Analysis: Joins harness observations with one or more model prediction files on `(seq_idx, meas_idx)`, computes error metrics, and generates CDF figures and percentile tables.

== Test Data

200 sequences sampled from the RIPE Atlas test split (`test.arrayrecord`, seed\=42). Each sequence is a 1024-token crop from a probe-centric row: all measurements within a sequence share the same source--destination IP pair and are ordered temporally.

#table(
  columns: 2,
  [Observations], [3,793],
  [With timestamps], [2,323 (61%)],
  [Unique (src, dst) pairs], [1,831],
  [Global median RTT], [80.64 ms],
)

= Baseline Methods

All baselines are trained _in-sample_ on the test sequences. The transformer models are trained on a separate training set and evaluated _out-of-sample_. This gives baselines a structural advantage; if the transformer still outperforms them, it is a conservative result.

== Simple Baselines

*Global median.* A single constant prediction for all positions:

$ hat(r) = op("median")({r_i : i in D}) $

*Last-seen.* The most recent RTT observation for the same pair within the sequence:

$ hat(r)^((t)) = r^((t-1)) $

*Window mean.* Average of the last $w=3$ observations:

$ hat(r)^((t)) = 1/w sum_(s=t-w)^(t-1) r^((s)) $

*Exponential moving average (EMA).* With smoothing factor $alpha = 0.3$:

$ hat(r)^((t)) = alpha r^((t-1)) + (1 - alpha) hat(r)^((t-1)) $

For the first observation in each sequence, all simple baselines fall back to the global median.

== Biased Matrix Factorization

Each IP address receives separate source and destination embedding vectors. Prediction is in log-space:

$ log hat(r)_(i,j) = bold(x)_i^top bold(y)_j + b_i^s + b_j^d + mu $

where $bold(x)_i, bold(y)_j in RR^r$ ($r=16$), $b_i^s, b_j^d$ are per-IP biases, and $mu$ is the global mean log-RTT. The model is trained via SGD on the squared error in log-space with $ell_2$ regularization ($lambda = 0.1$) and non-negativity constraints on embeddings. Based on DMFSGD (Liao et al., IEEE/ACM ToN 2013).

== Vivaldi Network Coordinate System

Each IP is assigned a coordinate vector $bold(c)_i in RR^d$ ($d=4$) and a height scalar $h_i >= 0$. The predicted RTT is:

$ hat(r)_(i,j) = ||bold(c)_i - bold(c)_j||_2 + h_i + h_j $

On each observation $(i, j, r)$, define the prediction error $e = r - hat(r)$ and relative error $epsilon = |e| slash r$. The error estimates are updated via EMA:

$ epsilon_i <- c_e dot epsilon + (1 - c_e) dot epsilon_i $

Coordinates are updated with an adaptive step size:

$ delta_i = c_c dot epsilon_i / (epsilon_i + epsilon_j), quad c_c = 0.25 $

$ bold(c)_i <- bold(c)_i + delta_i dot e dot bold(u)(bold(c)_i - bold(c)_j) $

$ h_i <- max(0, h_i + delta_i dot e) $

Both endpoints are updated symmetrically per observation. Parameters: $d=4$, $c_c=0.25$, $c_e=0.5$, 5 epochs.

== Temporal Regularized Matrix Factorization (TRMF)

TRMF factorizes a time-indexed RTT matrix with autoregressive temporal regularization. Measurements $(i, j, r, t)$ are binned into 15-minute intervals to form a sparse matrix $bold(Y) in RR^(n times T)$ (in log-RTT space, z-score normalized per pair). The observation mask $bold(M)$ is 1 where data exists and 0 otherwise.

The factorization is $bold(Y) approx bold(F) bold(X)$ where $bold(F) in RR^(n times K)$ captures spatial structure and $bold(X) in RR^(K times T)$ captures temporal dynamics. The temporal factors are regularized by an autoregressive model:

$ bold(x)_t approx sum_(l in cal(L)) bold(W)_l circle.tiny bold(x)_(t-l) $

where $cal(L) = {1, 2, 4, 96, 672}$ corresponds to lags of 15m, 30m, 1h, 1 day, and 1 week. Each $bold(W)_l in RR^K$ is a diagonal lag coefficient vector. The full objective minimized via gradient descent is:

$ cal(L) = underbrace(||bold(M) circle.tiny (bold(Y) - bold(F) bold(X))||_F^2, "reconstruction") + lambda_f ||bold(F)||_F^2 + lambda_x underbrace(sum_t ||bold(x)_t - sum_l bold(W)_l circle.tiny bold(x)_(t-l)||^2, "AR penalty") + eta ||bold(X)||_F^2 + lambda_w ||bold(W)||^2 + alpha sum_k (sum_l W_(l,k) - 1)^2 $ <trmf-objective>

The sum-to-one penalty on $bold(W)$ encourages the lag coefficients to form a convex combination. Gradients are accumulated across all lags for $bold(X)$ (a known pitfall in reference implementations that update lag-by-lag). Parameters: $K=20$, $lambda_f=1$, $lambda_x=100$, $eta=0.5$, $alpha=500$, $lambda_w=1$, $"lr"=10^(-4)$, 10k iterations.

Prediction: $hat(r)_(i,j)(t) = exp(sigma_i dot (bold(f)_i^top bold(x)_t) + mu_i)$ where $mu_i, sigma_i$ are the per-pair z-score parameters. Unseen pairs fall back to the biased MF prediction.

= Transformer Models

Two decoder-only GPT models trained on the RIPE Atlas training split with cross-entropy loss on next-token prediction. Both use RoPE, RMSNorm, $"ReLU"^2$ activations, and logit softcapping at 15.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: center,
  table.header[Name][Layers][Emb dim][Heads][Head dim][Params][Steps],
  [deep60-60k], [60], [384], [6], [64], [106M], [60k],
  [680m-200k], [24], [1536], [12], [128], [680M], [200k],
)

The deep60 architecture is a narrow-deep variant at roughly the same parameter count as the default 95M model (20L/640E), testing whether depth is more important than width for this task.

= Metrics

== Relative Error

For each prediction position with actual RTT $r$ and predicted RTT $hat(r)$:

$ epsilon_"rel" = (|hat(r) - r|) / r $

This metric is scale-invariant: a 10ms error on a 100ms RTT and a 1ms error on a 10ms RTT both give $epsilon_"rel" = 0.1$.

== Empirical CDF

The CDF at threshold $x$ is the fraction of predictions with relative error at most $x$:

$ F(x) = 1/N |{i : epsilon_"rel"^((i)) <= x}| $

A curve further to the left (higher CDF at lower error) indicates a better method. The log-scale variant reveals behavior at very small errors.

== Percentile Table

Reports $p$-th percentile of the relative error distribution for each method, along with mean absolute error (MAE) and median absolute error.

= Results

#figure(
  image("../outputs/figures/cdf_rel_err_log.pdf", width: 95%),
  caption: [CDF of relative prediction error (log scale). Solid black lines: transformer models. Dashed colored lines: structured baselines. Dotted faint lines: simple baselines. TRMF (in-sample, red) dominates at low error; 680m-200k is the best out-of-sample method.],
) <fig-cdf>

#figure(
  image("../outputs/figures/cdf_abs_err_ms_log.pdf", width: 95%),
  caption: [CDF of absolute prediction error in milliseconds (log scale). 80% of transformer predictions are within 100ms of the true RTT. TRMF achieves 50% of predictions within 10ms.],
) <fig-cdf-abs>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    table.header[Method][Count][MAE (ms)][Med. AE][p50 rel][p75 rel][p90 rel][p95 rel],
    [*TRMF*#super[\*]], [3793], [41.90], [17.23], [0.327], [0.727], [1.985], [4.766],
    [*680m-200k*], [3793], [51.81], [23.81], [0.327], [0.848], [5.358], [14.47],
    [*680m-200k-nots*], [3793], [52.24], [25.02], [0.326], [0.852], [5.358], [14.54],
    [*deep60-60k*], [3793], [55.24], [32.06], [0.397], [0.894], [4.801], [13.28],
    [*deep60-60k-nots*], [3793], [55.49], [32.64], [0.395], [0.898], [4.962], [14.28],
    [Vivaldi], [3793], [62.17], [51.06], [0.611], [1.991], [6.424], [16.00],
    [EMA], [3793], [66.45], [53.87], [0.625], [1.961], [5.998], [10.99],
    [Window mean], [3793], [69.03], [55.18], [0.633], [1.865], [5.819], [11.03],
    [Biased MF#super[\*]], [3793], [69.43], [57.47], [0.638], [1.718], [5.685], [12.62],
    [Last-seen], [3793], [79.05], [61.06], [0.677], [1.245], [6.246], [12.56],
    [Global median], [3793], [78.13], [64.34], [0.687], [2.592], [7.744], [18.54],
  ),
  caption: [Percentile table of relative prediction error. Methods marked #super[\*] are trained in-sample. Bold methods are transformers (out-of-sample). Suffix "-nots" denotes evaluation with timestamps stripped from the input.],
) <tab-percentile>

#figure(
  image("../outputs/figures/context_curve.pdf", width: 95%),
  caption: [Median absolute error vs.\ number of prior RTT observations in context. Transformers (solid black) drop sharply from \~62ms at cold start to \~22ms after 2--3 prior measurements, matching TRMF. Simple baselines (dotted) remain flat. This is in-context learning: the model identifies the pair's RTT regime from a few examples.],
) <fig-context>

== Key Observations

+ *TRMF dominates at all percentiles*, but it is trained in-sample. Its near-zero errors at low percentiles (@fig-cdf) reflect partial memorization of the sparse observation matrix (1,702 of 3.45M entries filled).

+ *680m-200k matches TRMF at the median* ($p_50 = 0.327$ for both) despite being evaluated out-of-sample. It leads all out-of-sample methods by a clear margin (MAE 51.8 vs next-best deep60 at 55.2).

+ *Scale helps*: 680m-200k beats deep60-60k across the board (MAE, median AE, $p_50$ through $p_75$). The deep60 architecture (narrow-deep at 106M params) is competitive but does not outperform the wider 680M model.

+ *Timestamps do not improve RTT prediction.* Stripping timestamps from the input changes median absolute error by less than 1ms for both models (@tab-percentile, "-nots" rows). The RTT byte cross-entropy is also nearly identical (4.134 vs 4.133 for deep60, 4.064 vs 4.061 for 680m). The model's RTT predictions come from IP pair identity and within-sequence RTT history, not temporal conditioning.

+ *In-context learning is the key mechanism.* @fig-context shows that transformers improve sharply with 1--3 prior measurements, dropping from 62ms to 22ms median error. After 3 observations, they match TRMF. Simple baselines remain flat. This suggests the model rapidly identifies the pair's RTT regime from a few examples.

+ *Transformers have heavy right tails*: at $p_90$ and $p_95$, both models are _worse_ than simple baselines like EMA. This suggests the models make confidently wrong predictions on a subset of measurements --- likely rare pairs or highly volatile RTTs. The Wasserstein loss (pending evaluation) may improve this tail behavior by penalizing predictions that are far from the true RTT in ordinal space.

+ *Simple baselines cluster together*: EMA, window mean, last-seen, biased MF, and Vivaldi all perform similarly ($p_50$ relative error 0.61--0.69). Vivaldi coordinates offer minimal benefit in this setup because each test sequence covers a single src--dst pair, limiting cross-pair leverage.

= Methodological Notes

*In-sample vs.\ out-of-sample.* Biased MF, Vivaldi, and TRMF are trained on the _test_ sequences and then evaluated on the same data. The transformer models were trained on a separate training split and have never seen the test sequences. This design is intentionally conservative: if the transformer outperforms in-sample baselines, the result is more compelling. A fairer comparison would hold out a temporal split within the test data, training baselines on earlier measurements and predicting later ones.

*Probe-centric sequences.* Each test sequence comes from a single probe row (one src--dst pair over time). Within-sequence baselines (EMA, last-seen, window mean) are therefore predicting the _next RTT for the same pair_, which is their strongest use case. Cross-pair methods (MF, Vivaldi, TRMF) can leverage structure across sequences but operate on a relatively small test set (1,831 unique pairs).

*First-observation fallback.* The first RTT observation in each sequence has no history, so all history-based baselines predict the global median at that position. The transformer also has limited context at the first position (just IP fields, no prior RTTs). This position is included in all metrics.
