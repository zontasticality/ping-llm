= Decentralized Internet Topology Learning Independent Study

== Motivation & Introduction

In order to create an internet more controlled by users than by corporations, we need to enable services to be able to run directly on end-user devices in a peer-to-peer network. Core to this problem is the issue of figuring out how to find nodes in _topologically ideal_ places, for example in anonymous routing where you want to route through an intermediary on the way to your destination, or for data storage, where you want your data to be stored close to the most likely users. These are fundamentally optimization problems that require being able to predict network properties between arbitrary nodes in a network in order to optimize node selection.

Much research has been done in this area over the years, but as of the author's knowledge, no one has trained a very large transformer model to learn a very general prediction algorithm that can not only predict latency between two nodes, but also predict a wide variety of other topological features for a variety of applications, including being able to condition on timestamp. This independent study investigates training a transformer directly on latency measurements in order to learn a variety of potentially useful prediction modes for node selection, in addition to reviewing the literature on how such a transformer could be trained in a decentralized fashion.

== Proposed Objectives

The objective of this independent study is work towards creating a system with _all_ of the following properties:

1. Ability to take into account time when predicting link properties.
2. Ability to predict link properties even without a connection to the network.
3. Ability to cheaply calibrate the predictive model with additional local measurements.
4. Ability to detect and resist significant node failures or malicious behavior (e.g. data poisoning).
5. Ability for nodes to limit training data discovery. (i.e. inspecting the model to figure out exactly who is communicating frequently with whom)
6. Ability to scale prediction accuracy based on end-user hardware capabilities.

A transformer model trained on heterogeneous measurement sequences leveraging in-context learning could potentially satisfy properties 1-3. It is unclear how properties 4-6 could be easily satisfied however. Thus the goal of this independent study is to train a transformer to solve goals 1-3, and do further, and then do a literature review to get a better idea on how properties 4-6 could be achieved.

== Design

=== Transformer

The transformer trained is a decoder-only transformer using Google's `maxtext` library. A decoder-only solution is used because this application only has one sequence type and is expected to be used autoregressively from the first measurement.

Architecturally, the target model is a medium-sized network (on the order of 80–100M parameters) with:

- An embedding dimension of approximately 640 and a vocabulary size of 267 (11 role tokens + 256 byte tokens) shared between input and output embeddings.
- 20 transformer decoder layers with multi-head self-attention (around 10 heads of width 64) and MLP blocks of width roughly 2048 (a ~3.2× expansion over the embedding size).
- A context window of 1024 tokens, which corresponds to roughly 60–70 IPv4 measurements or 40–50 mixed IPv4/IPv6 measurements per sequence under the tokenization scheme described below.

This design intentionally favors depth over width and uses a smaller-than-usual MLP ratio so that the model learns general routing and topology rules instead of memorizing individual (src, dst, time) triples.

=== Data

The data to be trained on is roughly 200M measurements directly downloaded and parsed from the RIPE Atlas project's 'daily dumps' page split 90/10 into train/test sets. @ripe_atlas_daily_dumps

The RIPE Atlas is a project that contains tens of thousands of 'anchor' nodes running on high-capacity networks that regularly ping each other as well as satisfy requests to ping other locations on the internet, acting as a public tool for internet testing and measurement.

For this project, the raw JSON data is preprocessed into a Parquet dataset `data/training_data.parquet`. And only the following fields are actually used:

```js
event_time: timestamp[ns]  # Event timestamp with timezone
src_addr: string           # Source IP address (IPv4 or IPv6)
dst_addr: string           # Destination IP address
ip_version: int64          # 4 or 6
rtt: double                # Round-trip time in milliseconds (-1.0 = failed probe)
```

The resulting dataset has the following characteristics (for the current `training_data.parquet` snapshot):

- Approximately 200M measurements (\~2.2 GB) sampled at random from over 28 days of measurement data (\~1TB).
- Date range on the order of late June to late July 2025, so the model sees both short-term and medium-term temporal variation.
- Mixed IPv4/IPv6 coverage with a roughly 60%/40% split between IPv4 and IPv6 rows.

=== Tokenization

In order to create a flexible prediction engine for internet measurements, we need to create a language that can encode sequences of measurements and allow for the transformer to learn various conditional slices of a given measurement. We have a base token language, and a way of composing them to encode sequences of measurements in a way that allows the transformer to learn how to predict from various slices.

The token language it defined as follows:
#import "@preview/simplebnf:0.1.1": *

#bnf(
  Prod(
    $"Token"$,
    {
      Or[$"<MeasurementStart>"$][Measurement boundary]
      Or[$"<SrcIPv4>" | "<SrcIPv6>"$][Source IP markers]
      Or[$"<DstIPv4>" | "<DstIPv6>"$][Destination IP markers]
      Or[$"<TimestampAbs>"$][Absolute timestamp marker]
      Or[$"<TimestampDelta1>" | "<TimestampDelta4>"$][Delta timestamp markers]
      Or[$"<RttStart>" | "<ThroughputStart>"$][Result value markers]
      Or[$"<Failed>"$][Connection failed marker]
      Or[$"<Byte0>", ..., "<Byte255>"$][Data bytes (11-266)]
    },
  ),
  Prod(
    $"U8"$,
    {
      Or[$"<Byte0>" | ... | "<Byte255>"$][Any single byte]
    },
  ),
  Prod(
    $"Float16"$,
    {
      Or[$"U8" "U8"$][Mantissa and Exponent]
    },
  ),
  Prod(
    $"SrcIp"$,
    {
      Or[$"<SrcIPv4>" "U8"^4$][IPv4 Address]
      Or[$"<SrcIPv6>" "U8"^16$][IPv6 Address]
    },
  ),
  Prod(
    $"DstIp"$,
    {
      Or[$"<DstIPv4>" "U8"^4$][IPv4 Address]
      Or[$"<DstIPv6>" "U8"^16$][IPv6 Address]
    },
  ),
  Prod(
    $"Timestamp"$,
    {
      Or[$"<TimestampAbs>" "U8"^8$][Absolute (First measurement)]
      Or[$"<TimestampDelta1>" "U8"$][Delta $< 256s$]
      Or[$"<TimestampDelta4>" "U8"^4$][Delta $>= 256s$]
    },
  ),
  Prod(
    $"Result"$,
    {
      Or[$"<RttStart>" "Float16"$][Successful probe (RTT)]
      Or[$"<ThroughputStart>" "Float16"$][Throughput (Future)]
      Or[$"<Failed>"$][Failed probe]
    },
  ),
  Prod(
    $"Field"$,
    {
      Or[$"SrcIp" | "DstIp" | "Timestamp" | "Result"$][Deterministically shuffled]
    },
  ),
  Prod(
    $"Meas"$,
    {
      Or[$"<MeasurementStart>" "Field"^(3..4)$][3 or 4 fields per record]
    },
  ),
  Prod(
    $"Context"$,
    {
      Or[$"Meas"^+$][Sequence of measurements]
    },
  ),
)

In the implementation, this grammar is realized as a compact 267-token vocabulary:

- 11 role tokens that mark measurement structure (`<MeasurementStart>`, IP family and direction, timestamp/RTT markers, and failure/throughput markers).
- 256 byte tokens (`<Byte0>`–`<Byte255>`) used to encode all numeric values as big-endian byte sequences.

Each measurement is serialized as `<MeasurementStart>` followed by three or four field blocks (source IP, destination IP, RTT or failure indicator, and optionally a timestamp). These field blocks are deterministically shuffled per measurement using a hash of `(src_addr, dst_addr, event_time)` so that the model is forced to learn the full joint distribution over fields rather than relying on a fixed field order.

Under this scheme, a typical IPv4 measurement with a timestamp uses about 16–23 tokens depending on whether the timestamp can be encoded as a 1-byte delta from the previous timestamp; IPv6 measurements are longer but follow the same pattern. Compared to the earlier, more verbose tokenization, this reduces sequence length by roughly a factor of 2–3, allowing a 1024-token context window to cover around three times as many measurements and enabling stronger in-context "network localization" from recent observations.

=== Rationale

The goal of training this transformer is to create a sort of 'foundation model' to predict various useful properties of internet links. Some useful properties might be:
- Given `<SrcIp>` and `<DestIp>`, and optionally `<Timestamp>` what is estimated distribution of `<Latency>` or `<Bandwidth>` for this potential connection?
- Given an `<SrcIP>` and a `<Latency>`, which `<DestIp>`s are most likely to be closest?
- Given an IP prefix, which IPs are most likely to be online?
- What is the IPv6 / IPv4 that a given `<SrcIp>` likely corresponds to?
- What is the distribution of IPs that `<SrcIP>` is most likely to talk to?
- Given a `<SrcIP>`, `<DestIP>` and `<Latency>` or `<Bandwidth>`, what is the most likely time this measurement was taken?
- What `<SrcIP>`s `<DestIP>`s `<Timestamps>` are most common?
- Given `<Latency>` and `<Timestamp>`, what connections (`<SrcIP>` + `<DestIP>` pairs) are most likely to fit these properties?

Multiple usages of these kinds of predictions could enable not only more efficient routing, but if further trained on the properties of full anonymous paths, this might allow for both path selection and estimation of path anonymity, by either sampling paths from the model, or given a path estimating how likely it is to have been generated.

All the above prediction modes can be achieved by simply randomizing the properties within each measurement when encoding measurements into the sequence.

==== Privacy Concerns

Because the model is trained on real-world network measurements, there is a risk of memorizing rare or sensitive traffic patterns. The architectural choices above (medium model size, smaller MLP ratio, and emphasis on learning aggregate routing structure) are partly motivated by a desire to generalize rather than memorize individual flows, but a more thorough privacy analysis is left to the second half of this paper discussing the space of decentralized training algorithms.

=== Size & Training

The model is in the 100M parameter range, which should ideally be large enough to capture rich IP and routing structure but small enough to be trainable on a single modern accelerator.

Training is formulated as standard next-token prediction over tokenized measurement sequences. With a context length of 1024. Tokenization is done at runetime using a loader built on google's `grain` library. It samples rows randomly from multiple sharded parquet files where each row contains measurements exclusively from a single source IP address. This row is then randomly sampled from given a random window (window size sampled from a log uniform distribution) and offset in order for the model to learn to be able to infer from arbitrary measurements at various times and distances from each other.

== Model Architecture

The model is a decoder-only transformer whose primary goal is to learn the joint distribution of a measurement `(src_ip, dst_ip, rtt, timestamp)` conditional on past measurements. Key architectural choices are:

- A relatively deep stack of decoder layers (≈20) to support multi-step reasoning tasks such as "RTT → IP search" and hierarchical IP structure learning (from coarse prefixes down to specific subnets).
- A moderate embedding size (\~640) with standard 64-dimensional attention heads to keep the model expressive without oversizing the representation for a 267-token vocabulary.
- A smaller MLP expansion ratio (\~3.2×) than is typical in general-purpose language models, which reduces pure memorization capacity and encourages the network to represent reusable routing and geography patterns.
- A context window of 1024 tokens so the model can condition on tens of recent measurements from the same vantage point and thereby infer location, connection type (residential vs datacenter), and other latent properties via in-context learning.

These are mostly arbitrary choices based on intuition since I don't have any data as to what degree parameters effect performance on this dataset.

== Runtime Data Processing

The goal is to train a transformer in a somewhat reminiscent manner as to how it might be trained in a federated environment. The dataset also needs to be preprocessed into a format that can efficiently feed a high-speed GPU like an A100/H100/B200.

To do this, we use `duckdb` to group by source ip address and then serialize each group of measurements to an arrayrecord row with a variable number of measurements per row. Context: Arrayrecord is a technology google uses to do efficient runtime deterministic random sampling of large datasets for training large models.

During traing-time sampling, training alternates between three timestamp modes:

- *Full timestamp:* all measurements in a window include timestamps and are ordered temporally, allowing the model to learn diurnal and long-range temporal patterns with efficient delta encoding.
- *No timestamp:* timestamps are omitted and measurements are randomly shuffled, forcing the model to learn atemporal network topology (e.g., geographic and ASN structure) that can be applied when timestamps are unavailable.
- *Mixed timestamps:* some measurements retain timestamps while others do not; timestamped measurements preserve temporal order while non-timestamped measurements are shuffled and then interleaved, teaching the model to use temporal information when available but remain robust to missing or partial timing data.


== Training Details

Training uses the AdamW optimizer with a cosine learning rate schedule and a brief warmup period. A representative configuration is:

- Learning rate on the order of `3e-4` with ~2000 warmup steps followed by cosine decay.
- Global batch size 256 with sequence length 1024, but currently likely bottlenecked by token feeding rate at only 50k tokens per second.
  - Dropout of about 0.1 and weight decay of about 0.01 for regularization.
  - bfloat16 for parameters and activations to match modern accelerator hardware.

The primary metric during training is token-level cross-entropy on held-out measurements, with downstream evaluation focusing on latency prediction accuracy and generalization to unseen IP pairs and network conditions.

==== Note: Training is still in Work In Progress.
#image("training_wip.png")

Some issues I ran into + am actively trying to solve when training this transformer:
 - Defining the token spec
 - Figuring out how optimize Maximum FLOP utilization
   - Data loading issues (moving from parquet to arrayrecord, implementation issues with array record)
   - Attempting to get more efficient attention implementation from nvidia's transformer-engine library (depended on a version of cudnn that was not preinstalled on Unity. Didn't want to deal with Apptainer so I moved to a 3rd party cloud provider.)

== Evaluation & Metrics

Note: Not yet completed.

The model is ultimately evaluated on how well it captures the joint distribution of `(src_ip, dst_ip, rtt, timestamp)` and how useful its predictions are for downstream tasks.

During training and basic validation:

- Token-level cross-entropy and perplexity are monitored on a held-out test split drawn from the same time period as the training data.

// TODO: Check if we are actually doing this!

- For forward latency prediction (given `src_ip`, `dst_ip`, and optionally a timestamp), we compute standard regression metrics such as mean absolute error (MAE), mean squared error (MSE), and calibration curves (`P(rtt < X)` vs empirical frequency), broken down by IPv4 vs IPv6, geographic/ASN distance, and inferred connection type (residential vs datacenter).

For downstream behaviors, we consider:

- *Forward modeling:* quality of RTT predictions for novel IP pairs, with a target of achieving MAE on the order of tens of milliseconds for typical internet paths (50–500ms RTT).

// This can be done by just sampling a bunch of connections (perhaps traceroute) and then comparing latency predeictions.

- *Inverse search (RTT → IP):* given a target RTT and context, sample candidate destination IPs and evaluate the plausibility and diversity of samples (e.g. fraction of samples whose RTT falls in a desired band, geographic spread, and consistency across repeated queries).

// TODO: Do we actually have enough time for this?
// This can be verified by choosing a source ip, picking an a range of rtts, sampling a selection of ips for each ip from the model, and then pinging all of them comparing how close they were to the target.
// TODO: This this actually work if we don't allow for representing the stdev of the latency, i.e. jitter?

- *IP completion:* given a partial IP (e.g. prefix) and context, predict plausible completions and measure byte-level accuracy and top‑K accuracy, stratified by prefix length and allocation type (datacenter vs residential).

// This can be done by choosing a prefix, sampling a bunch of connected and non-connected sub-addresses, and checking to make sure they are indeed connected and non-connected.

Generalization is probed by constructing held-out evaluation sets that remove specific structure from training, such as:

- Particular geographic regions or ASNs (to test whether the model has learned reusable routing rules rather than memorizing specific pairs).
- Later time windows beyond the training range (to test temporal robustness).
- IP pairs never observed during training (to test extrapolation based on learned topology and address-structure priors).

= Reflections on Decentralized Learning

Truly decentralized learning, where a model is updated locally and gradients are shared, often with differential privacy applied to the gradients or various other techniques to avoid leaking too much about private training data to the rest of the network, is a very active area of research. The design space is vast, with variations on how nodes connect, how updates are exchanged, how data is made private (differential privacy, multi-party computation) among many other dimensions.

Looking at existing research, no solution that I can see so far solves _all_ the following requirements that would be required for this to be practical in a ungoverned peer-to-peer network. Most of them don't fully solve just the categories they are in either.

1. Resilience to sybil attacks
   - None found

2. Capable of keeping information that needs to be private, or at least providing a reasonable tradeoff between privacy and performance.
   - P4: Towards private, personalized, and Peer-to-Peer learning @maheri2024p4
   - Exposing the Vulnerability of Decentralized Learning to Membership Inference Attacks Through the Lens of Graph Mixing @touat2024graphmixingmia
   - Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy @piran2025privatedfl
   - Low-Cost Privacy-Preserving Decentralized Learning (Zip-DL) @biswas2025zipdl
   - Robust peer-to-peer learning via secure multi-party computation @luo2023robustp2psmpc

3. Capable of dealing with adversarial actors who spread bad information for specific ends.
   - Backdoor Attacks in Peer-to-Peer Federated Learning @syros2023backdoorp2pfl
   - Robust Peer-to-Peer Machine Learning Against Poisoning Attacks @bouhaddi2025robustp2ppoisoning
   - Byzantine-Robust Decentralized Federated Learning (BALANCE) @fang2024balance
   - GRANITE: a Byzantine-Resilient Dynamic Gossip Learning Framework @belal2025granite

4. Can deal with free-loaders / is well-incentivized.
   - None found

5. Capable of scaling model size to node capabilities.
   - P4: Towards private, personalized, and Peer-to-Peer learning @maheri2024p4
   - Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy @piran2025privatedfl
   - Low-Cost Privacy-Preserving Decentralized Learning (Zip-DL) @biswas2025zipdl
   - A Tale of Two Learning Algorithms: Multiple Stream Random Walk and Asynchronous Gossip @gholami2025taleoftwo


How in such a dynamic system like the internet is a static protocol supposed to deal with all the ways adversaries can possibly disrupt or measure a network? Not only for the purposes of figuring out if your connection is sufficiently obfuscated, but even for modeling the internet itself, or incentivizing nodes to cooperate with each other, the sheer amount of complexity and things that could go wrong / be exploited seems insurmountable.

However, we already have a system of decentralized independently-learning and cooperating nodes that can incentivize each other and form protocols in various fashions: Human Society! The question then is, is it possible to program AIs to manage a given node's connections based on the owner's preferences in such a way as to take into account potential adversarial scenarios?

== A Probably Dangerous Theoretical Moon-Shot

One idea I've been thinking about for some time is recursively improving AI models. Specifically a model where given a type declaration in some (ideally dependently-typed) programming language, the AI would infer a term that satisfies the type. What could this be used for? Some obvious examples come to mind: Theorem proving, SAT solving, generating implementations that fit a specification. An example of such a specification might be the specification for such a self-improving AI itself:

```rust
// The type of a program that takes a predicate on programs (a.k.a. a 'type') and returns a program that satisfies the type
ProgramInferrer := { pred : Program -> bool } -> Program;
```

This can even be generalized to arbitrary valuation functions on programs.

```rust
// The type of a program that takes a predicate on programs (a.k.a. a 'type') and returns a program that satisfies the type
ProgramInferrer := { pred : Program -> Real } -> Program;
```

Lets say we also generalized our type system into one that graded programs (assigning numbers), as opposed to just checking them.

```rust
// The type of a program that takes a predicate on programs (a.k.a. a 'type') and returns a program that satisfies the type
ProgramInferrer := { pred : Program -> Real } -> Program;
```

Then you might imagine being able to run something like:
```
program_inferrer : ProgramInferrer := {/* initial impl */}
program_inferrer(ProgramInferrer)(ProgramInferrer)(ProgramInferrer)(...)
```

What could someone possible do with this kind of program that can find a close-to-optimal program satisfying a given specification and scoring function? I imagine it would be possible to do a wide variety of things with such a generalized search program. For one, you could create the fastest and most accurate simulations of any domain you would like, or more precisely navigate along the Pareto-frontier of such a trade-off. Such a simulation could be used to create intensive training environments for RL agents that are themselves produced by the general optimization program to maximize reward and minimize risk over the course of the simulation, similar to how self-driving cars or robots are trained (in highly adversarial environments where you have god-like view to provide dense feedback, for example, if a routing agent is effectively hiding traffic or not). The skies the limit!

Here are some potential outlines of what this might look like (WIP).

#bibliography("bibliography.bib")
