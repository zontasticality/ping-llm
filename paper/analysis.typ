#set page(margin: 2cm)
#set text(font: "New Computer Modern", size: 10.5pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Compile from the repository root with:
// typst compile --root . paper/analysis.typ paper/analysis.pdf

#align(center)[
  #text(size: 16pt, weight: "bold")[RTT Prediction Analysis]
  #v(0.35em)
  #text(size: 11pt)[Time-clean model-vs-baseline evaluation]
  #v(0.8em)
]

= Summary

This report collects the current analysis figures and percentile table in one place. The underlying split is the time-clean evaluation in `data/eval_timeclean`: the graph used by the structured baselines is selected only from first-half measurements, while the test rows are sampled from the second half.

#table(
  columns: 2,
  [Evaluation directory], [`data/eval_timeclean`],
  [Analysis output], [`outputs/eval_timeclean_models`],
  [Graph-selection window], [First half only (`graph_window=train`)],
  [Train measurements], [5,000,000],
  [Test observations], [1,000,000],
  [Model sample], [10,000 observations x 5 context sizes],
  [Context sizes], [`0, 1, 2, 5, 10`],
)

The structured baselines are trained transductively on the 5M first-half training rows. Model evaluations are out-of-sample checkpoint evaluations on a deterministic 10k sample of the 1M test observations. The model rows in the table therefore have count 10k, while baseline rows have count 1M.

For the token-position analysis, the 680M checkpoint was evaluated with greedily max-packed same-source context. With the 1024-token model window, the observed maximum was 62 context measurements, median capacity was 59, and the median packed sequence length was 1,014 tokens. Capacity is lower for IPv6-heavy contexts because each IPv6 address consumes 16 byte tokens instead of 4.

= Baseline Definitions

The baseline columns are computed as follows:

+ `global_median`: every test row receives the median RTT from the first-half training measurements. This is also the fallback when a structured baseline cannot score an unseen endpoint.

+ `last_seen`: for each source address, test observations are sorted by time. The prediction is the previous RTT seen for that source in the test stream; the first row for a source falls back to `global_median`.

+ `window_mean`: same causal per-source test stream as `last_seen`, but predicts the mean of the previous three source RTTs; the first row falls back to `global_median`.

+ `ema`: same causal per-source test stream, using an exponential moving average with update `ema = 0.3 * current_rtt + 0.7 * ema`; the first prediction for a source is `global_median`.

+ `vivaldi`: a 4D Vivaldi network-coordinate system. Each IP gets a coordinate vector and nonnegative height; prediction is coordinate distance plus source and destination heights. Training is an online iterative fit on first-half measurements.

+ `vivaldi_time`: the same Vivaldi model, but timestamped training rows are processed in chronological order rather than shuffled.

+ `dmfsgd`: asymmetric nonnegative matrix factorization. Each IP has a source vector `X` and destination vector `Y`; prediction is `dot(X_src, Y_dst) * scale`. The implementation uses L1 sign-SGD, robust 99th-percentile RTT scaling, nonnegative projection, and shuffled passes over first-half measurements.

+ `dmfsgd_time`: the same tuned DMFSGD model, but timestamped training rows are processed chronologically rather than shuffled.

+ `dmfsgd_paper`: a closer reproduction of the paper-style DMFSGD mechanics: max RTT scaling, uniform `[0, 1]` initialization, stronger default regularization, bounded neighbor caches, optional age decay, and minibatch-style line-search updates.

+ `dmfsgd_paper_time`: the paper-style DMFSGD variant with timestamped rows processed chronologically.

+ `biased_mf`: recommender-style matrix factorization in log-RTT space. It predicts `log(rtt)` as a global bias plus source bias, destination bias, and a source-destination dot product, then exponentiates back to milliseconds.

= Figure Gallery

#figure(
  image("../outputs/eval_timeclean_models/figures/cdf_rel_err.pdf", width: 96%),
  caption: [CDF of relative prediction error on a linear x-axis. Structured graph baselines dominate the low-error region; model curves improve substantially once any context is available.],
) <fig-rel-cdf>

#text(weight: "bold")[Analysis.] This figure is computed by sorting `abs(predicted_rtt - actual_rtt) / actual_rtt` for each method and plotting the cumulative fraction below each error. The tuned structured baselines are clearly ahead: `dmfsgd` has median relative error 0.0838 and `vivaldi` has 0.0928, while the best model median is `680m-200k ctx1` at 0.1563. The cold-start model curves are much worse, which means the transformer depends heavily on at least one relevant context measurement. The paper-style DMFSGD curves sit near relative error 1 because max scaling collapses normal RTTs in this outlier-heavy corpus.

#figure(
  image("../outputs/eval_timeclean_models/figures/cdf_rel_err_log.pdf", width: 96%),
  caption: [CDF of relative prediction error on a log x-axis. This view separates the low-error portion of the graph baselines from the transformer and simple-history baselines.],
) <fig-rel-cdf-log>

#text(weight: "bold")[Analysis.] This is the same relative-error CDF as @fig-rel-cdf, but the x-axis is logarithmic. The log view shows that the structured baselines win across the low-error range, not only at the median. Among models, `680m-200k` is strongest around the median but has a worse high-relative-error tail than `deep60-was-60k`: at p95 relative error, `deep60-was ctx10` is 1.2038 while `680m ctx10` is 4.2035. That suggests the 680M checkpoint often gets typical cases closer, but fails more catastrophically on some small-RTT or hard rows.

#figure(
  image("../outputs/eval_timeclean_models/figures/cdf_abs_err_ms.pdf", width: 96%),
  caption: [CDF of absolute error in milliseconds on a linear x-axis.],
) <fig-abs-cdf>

#text(weight: "bold")[Analysis.] This figure sorts `abs(predicted_rtt - actual_rtt)` in milliseconds. Absolute error tells the same broad story as relative error: `dmfsgd` and `vivaldi` are tightly concentrated under roughly 10--20ms, while the best transformer rows are closer to 18ms median absolute error. The simple history baselines are poor because same-source recent RTTs do not reliably identify the destination-specific latency. `biased_mf` is competitive but still behind `dmfsgd` and `vivaldi`, likely because log-space squared loss trades median accuracy for smoother behavior on large values.

#figure(
  image("../outputs/eval_timeclean_models/figures/cdf_abs_err_ms_log.pdf", width: 96%),
  caption: [CDF of absolute error in milliseconds on a log x-axis. This makes the 5--50ms range easier to inspect.],
) <fig-abs-cdf-log>

#text(weight: "bold")[Analysis.] This is the same absolute-error CDF as @fig-abs-cdf with a logarithmic x-axis. The log view makes the sub-50ms region easier to compare. The best model families improve sharply over simple baselines in this range, but still sit to the right of `dmfsgd` and `vivaldi`; in practical terms, the model is learning some local RTT structure, but not the persistent per-node latent state that the graph baselines explicitly fit.

#figure(
  image("../outputs/eval_timeclean_models/figures/context_curve.pdf", width: 96%),
  caption: [Median absolute error versus the number of previous test observations already seen for the same source, clipped at 10+. Model input context size is encoded by line thickness.],
) <fig-context>

#text(weight: "bold")[Analysis.] This curve groups rows by causal same-source test history count (`prior_rtts`, clipped at 10+) and plots bootstrap median absolute-error intervals for each prediction column. The line thickness legend is the fixed model context size used when producing that model prediction, not the x-axis value. The structured baselines stay low and relatively flat across bins because they already carry fitted per-node state from the training graph. The model curves are noisier, but the main signal is robust: cold start is bad, one context measurement gives most of the improvement, and more fixed context beyond one or two measurements has little consistent effect in this evaluation.

#figure(
  image("../outputs/eval_timeclean_models/figures/context_capacity.pdf", width: 92%),
  caption: [Maximum number of same-source context measurements that fit in the 1024-token model window when greedily packing recent train measurements before the query. Capacity varies mainly with IPv4/IPv6 length and timestamp delta encoding.],
) <fig-context-capacity>

#text(weight: "bold")[Analysis.] This histogram was computed from 2,000 deterministic sampled test queries. For each query, the analysis greedily prepended the most recent same-source training measurements until the encoded sequence would exceed the 1024-token model limit. The median capacity is 59 context measurements and the observed max is 62, but the distribution is bimodal: 854 rows fit only 23--24 measurements, while most of the rest fit about 58--62. The low-capacity mode is consistent with IPv6-heavy contexts, where each address consumes many more byte tokens.

#figure(
  image("../outputs/eval_timeclean_models/figures/token_position_accuracy_heatmap.pdf", width: 98%),
  caption: [Top-1 token accuracy by semantic token type and measurement offset from the query. Offset 0 is the query measurement; -1 is the most recent context measurement.],
) <fig-token-heatmap>

#text(weight: "bold")[Analysis.] This heatmap uses the max-packed 680M run. After a full forward pass over each packed sequence, every next-token prediction is classified by semantic token type and measurement offset from the query; the color is top-1 token accuracy. Source IP bytes are nearly trivial overall, with 98.7% average accuracy, because same-source context repeats the source address. Destination IP bytes are harder but still strong at 70.8% average. Timestamp bytes are much weaker at 35.1%, and RTT bytes are weakest at 14.0%. The query measurement at offset 0 has 17.6% RTT-byte top-1 accuracy, so the model's scalar RTT errors are not just an artifact of the decoding harness; the byte-level RTT distribution is genuinely uncertain.

#figure(
  image("../outputs/eval_timeclean_models/figures/token_position_accuracy_relative.pdf", width: 98%),
  caption: [Token accuracy by context position, centered relative to each token type's own mean accuracy. This separates whether a token type improves or degrades near the query from the baseline difficulty of that token type.],
) <fig-token-relative>

#text(weight: "bold")[Analysis.] This figure uses the same token-position aggregates as @fig-token-heatmap, but subtracts each token type's own weighted mean accuracy. It exposes position effects without letting easy source-IP tokens dominate the scale. The oldest offsets are less reliable because fewer examples reach those positions and because the long-context/short-context mixture changes. Destination accuracy improves closer to the query, source accuracy is flat except at the sparse oldest positions, and RTT-byte accuracy fluctuates around its own low mean with only a small query-position bump. That is evidence that the model is not extracting much additional RTT signal from deep context, even when about 60 measurements fit.

= Percentile Table

#figure(
  text(size: 7.4pt)[
    #table(
      columns: (2.9cm, 1.2cm, 1.25cm, 1.25cm, 1.05cm, 1.05cm, 1.05cm, 1.05cm),
      align: (left, right, right, right, right, right, right, right),
      table.header[Method][Count][MAE][Med. AE][p50 rel][p75 rel][p90 rel][p95 rel],
      [`dmfsgd`], [1M], [16.57], [8.58], [0.0838], [0.1711], [0.3258], [0.4658],
      [`vivaldi`], [1M], [17.63], [9.45], [0.0928], [0.1918], [0.3508], [0.5043],
      [`dmfsgd_time`], [1M], [17.56], [9.85], [0.0935], [0.1939], [0.3747], [0.5388],
      [`biased_mf`], [1M], [25.84], [12.60], [0.1232], [0.2313], [0.3772], [0.4996],
      [`vivaldi_time`], [1M], [21.36], [12.66], [0.1172], [0.2470], [0.4927], [0.7874],
      [`680m-200k ctx0`], [10k], [54.41], [30.91], [0.2288], [0.5515], [2.6157], [5.9618],
      [`680m-200k ctx1`], [10k], [44.08], [17.74], [0.1563], [0.4214], [0.9242], [4.2394],
      [`680m-200k ctx2`], [10k], [44.57], [17.95], [0.1591], [0.4251], [0.9270], [4.0797],
      [`680m-200k ctx5`], [10k], [44.00], [17.83], [0.1599], [0.4131], [0.9233], [4.0203],
      [`680m-200k ctx10`], [10k], [44.19], [17.94], [0.1601], [0.4146], [0.9307], [4.2035],
      [`deep60-was ctx0`], [10k], [56.33], [32.15], [0.3278], [0.6582], [1.1749], [2.4075],
      [`deep60-was ctx1`], [10k], [42.30], [18.48], [0.1867], [0.5387], [0.8247], [1.2437],
      [`deep60-was ctx2`], [10k], [42.50], [18.33], [0.1880], [0.5405], [0.8253], [1.1961],
      [`deep60-was ctx5`], [10k], [42.34], [18.24], [0.1868], [0.5397], [0.8263], [1.2098],
      [`deep60-was ctx10`], [10k], [41.97], [17.98], [0.1842], [0.5326], [0.8228], [1.2038],
      [`deep60 ctx0`], [10k], [64.33], [53.41], [0.3639], [0.7432], [2.5505], [5.0207],
      [`deep60 ctx1`], [10k], [57.73], [38.95], [0.3043], [0.6241], [1.5797], [3.8093],
      [`deep60 ctx2`], [10k], [56.96], [36.22], [0.2935], [0.6103], [1.2205], [3.4329],
      [`deep60 ctx5`], [10k], [54.85], [33.79], [0.2747], [0.5701], [1.1968], [3.6787],
      [`deep60 ctx10`], [10k], [53.98], [33.11], [0.2612], [0.5492], [1.1967], [3.9923],
      [`dmfsgd_paper`], [1M], [139.79], [138.11], [0.9707], [0.9799], [0.9851], [0.9876],
      [`dmfsgd_paper_time`], [1M], [139.83], [138.16], [0.9706], [0.9800], [0.9853], [0.9878],
      [`ema`], [1M], [65.83], [55.43], [0.4126], [0.9254], [3.3455], [5.5632],
      [`last_seen`], [1M], [81.72], [66.97], [0.4849], [0.8959], [3.2381], [6.4685],
      [`window_mean`], [1M], [69.35], [57.19], [0.4265], [0.8880], [3.3439], [5.7434],
      [`global_median`], [1M], [104.92], [92.82], [0.7112], [0.8069], [1.0931], [2.1305],
    )
  ],
  caption: [Percentile table for the time-clean evaluation. Lower is better. Model rows are evaluated on the 10k deterministic model sample; baseline rows are evaluated on the full 1M observations.],
) <tab-percentiles>

#text(weight: "bold")[Analysis.] This table is computed from the same error columns used by the CDFs. The best overall method is `dmfsgd` with 8.58ms median absolute error and 0.0838 median relative error. `vivaldi` is close behind, and `biased_mf` is a credible but weaker structured baseline. The best model row is `680m-200k ctx1` by median absolute error at 17.74ms, while `deep60-was ctx10` has slightly better mean absolute error and much better high-percentile relative error. The comparison is useful but not symmetric: baselines evaluate 1M rows with transductive fitted node state, while model rows evaluate 10k sampled rows with only local serialized context.

= Notes

+ The best structured baseline remains `dmfsgd` with 8.58ms median absolute error and 0.0838 median relative error.

+ The strongest model result is `680m-200k` with one context measurement: 17.74ms median absolute error and 0.1563 median relative error. Additional context does not materially improve it on this sample.

+ The latest WAS model, `deep60-was-60k`, improves sharply with context: 32.15ms median absolute error at cold start, about 18ms once context is available.

+ The previous deep run, `deep60-60k`, is clearly weaker than both `deep60-was-60k` and `680m-200k` across all context sizes.

+ The paper-style DMFSGD variants remain poor because max-scaling collapses the normalized target distribution on this outlier-heavy RTT corpus.

+ The structured baselines are transductive graph-completion methods with persistent per-node fitted state. The transformer model receives only a small local context at eval time, so this evaluation currently favors Vivaldi/DMFSGD-style baselines.

+ The token-position figures (@fig-context-capacity, @fig-token-heatmap, and @fig-token-relative) use the 680M checkpoint with max-packed same-source context. They measure general next-token accuracy across all tokens in the packed sequence, not just RTT-byte point prediction at the query.

= Accuracy Conclusions

+ The current models are learning useful RTT structure, but not enough persistent graph structure. One context measurement cuts model median error roughly in half relative to cold start, yet additional same-source context mostly plateaus.

+ The strongest baselines win because they fit durable per-IP latent state from the training graph. The model must rediscover this state from serialized local context on every query, and same-source context repeats the source while often providing weak destination-specific evidence.

+ RTT and timestamp tokenization are likely limiting factors. Address bytes are predicted well, but RTT-byte top-1 accuracy is only 14.0% overall and 17.6% at the query. A numeric RTT head, a better continuous/discretized RTT target, loss weighting toward RTT bytes, or auxiliary scalar regression could improve the part of the sequence that actually matters for point prediction.

+ Longer packed context is feasible, up to about 62 same-source measurements in this data, but the token-position analysis does not show strong RTT-byte gains from deep context by itself. A next step is to evaluate point prediction with dynamic max-packed context and compare it directly to fixed `ctx0/1/2/5/10`.

+ The most promising modeling direction is to give the transformer more graph-completion signal rather than only more chronological context: retrieve same-pair or same-destination neighbors, add learned node/coordinate features, distill from `dmfsgd` or `vivaldi`, or pretrain/fine-tune with an objective that emphasizes query RTT accuracy over generic next-token accuracy.
