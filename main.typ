= Decentralized Internet Topology Learning Independent Study

== Motivation & Introduction

In order to create an internet more controlled by users than by corporations, we need to enable services to be able to run directly on end-user devices in a peer-to-peer network. Core to this problem is the issue of figuring out how to find nodes in _topologically ideal_ places, for example in anonymous routing where you want to route through an intermediary on the way to your destination, or for data storage, where you want your data to be stored close to the most likely users. These are fundamentally optimization problems that require being able to predict network properties between arbitrary nodes in a network in order to optimize node selection.

Much research has been done in this area over the years, but as of the author's knowledge, no-one has trained a very large transformer model to learn a very general prediction algorithm that can not only predict latency between two nodes, but also predict a wide variety of other topological features for a variety of applications, including being able to condition on timestamp. This independent study investigates training a transformer directly on latency measurements in order to learn a variety of potentially useful prediction modes for node selection, in addition to reviewing the literature on how such a transformer could be trained in a decentralized fashion.

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

It was trained with the following parameters:

// Figure out best parameters when we train

- Written in JAX + Equinox + Heliax so it is comprehensible. Highly modular across multiple files and well-commented.
- Transformer should learn various associations between ips and latency, either predicting latency or ip.

=== Data

The data to be trained on is roughly 100M measurements directly downloaded and parsed from the RIPE Atlas' project's daily dumps. @ripe_atlas_daily_dumps

The RIPE Atlas is a project that contains tens of thousands of 'anchor' nodes running on high-capacity networks that regularly ping each other as well as satisfy requests to ping other locations on the internet, acting as a public tool for internet testing and measurement. The `ping` measurements result in the following row structure:

// TODO need to figure out the schema that doesn't only include other anchors

```
prb_id: uint32          # Probe identifier
ts: int64               # Unix timestamp
sent: uint8             # Packets sent
rcvd: uint8             # Packets received
avg: float32            # Average latency (ms) - target variable
rtt_1,2,3: float32      # Individual round-trip times
dst_is_ipv6: bool       # Destination IP version
dst_ipv4_int: uint32    # Destination IPv4 as integer
dst_ipv6_bytes: binary  # Destination IPv6 as bytes
src_is_ipv6: bool       # Source IP version
src_ipv4_int: uint32    # Source IPv4 as integer
src_ipv6_bytes: binary  # Source IPv6 as bytes
dst_addr_display: str   # Human-readable destination IP
src_addr_display: str   # Human-readable source IP
```

=== Tokenization

In order to create a flexible prediction engine for internet measurements, we need to create a language that can encode sequences of measurements and allow for the transformer to learn various conditional slices of a given measurement. We have a base token language, and a way of composing them to encode sequences of measurements in a way that allows the transformer to learn how to predict from various slices.

The token language it defined as follows:
#import "@preview/simplebnf:0.1.1": *

#bnf(
  Prod(
    $"Token"$,
    annot: $sans("Token")$,
    {
      Or[$"<MeasurementStart>"$][Start of measurement]
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
    annot: $sans("U8")$,
    {
      Or[$"<Byte0>" | ... | "<Byte255>"$][Any single byte]
    },
  ),
  Prod(
    $"U16"$,
    annot: $sans("U16")$,
    {
      Or[$U8 U8$][Big-endian uint16]
    },
  ),
  Prod(
    $"U32"$,
    annot: $sans("U32")$,
    {
      Or[$U8 U8 U8 U8$][Big-endian uint32]
    },
  ),
  Prod(
    $"U64"$,
    annot: $sans("U64")$,
    {
      Or[$U8 ... U8$][Big-endian uint64 (8 bytes)]
    },
  ),
  Prod(
    $"Float16"$,
    annot: $sans("Float16")$,
    {
      Or[$U8 U8$][Mantissa and Exponent (microseconds)]
    },
  ),
  Prod(
    $"SrcIPv4"$,
    annot: $sans("SrcIPv4")$,
    {
      Or[$"<SrcIPv4>" U8 ... U8$][Marker + 4 bytes]
    },
  ),
  Prod(
    $"SrcIPv6"$,
    annot: $sans("SrcIPv6")$,
    {
      Or[$"<SrcIPv6>" U8 ... U8$][Marker + 16 bytes]
    },
  ),
  Prod(
    $DstIPv4$,
    annot: $sans("DstIPv4")$,
    {
      Or[$"<DstIPv4>" U8 ... U8$][Marker + 4 bytes]
    },
  ),
  Prod(
    $DstIPv6$,
    annot: $sans("DstIPv6")$,
    {
      Or[$"<DstIPv6>" U8 ... U8$][Marker + 16 bytes]
    },
  ),
  Prod(
    $SrcIp$,
    annot: $sans("SrcIp")$,
    {
      Or[$SrcIPv4 | SrcIPv6$][Source Address]
    },
  ),
  Prod(
    $DstIp$,
    annot: $sans("DstIp")$,
    {
      Or[$DstIPv4 | DstIPv6$][Destination Address]
    },
  ),
  Prod(
    $Timestamp$,
    annot: $sans("Timestamp")$,
    {
      Or[$"<TimestampAbs>" U64$][Absolute (First measurement)]
      Or[$"<TimestampDelta1>" U8$][Delta < 256s (Common)]
      Or[$"<TimestampDelta4>" U32$][Delta >= 256s (Rare)]
    },
  ),
  Prod(
    $Result$,
    annot: $sans("Result")$,
    {
      Or[$"<RttStart>" Float16$][Successful probe (RTT)]
      Or[$"<ThroughputStart>" Float16$][Throughput (Future)]
      Or[$"<Failed>"$][Failed probe]
    },
  ),
  Prod(
    $Field$,
    annot: $sans("Field")$,
    {
      Or[$SrcIp | DstIp$][IP Fields]
      Or[$Timestamp | Result$][Data Fields]
    },
  ),
  Prod(
    $Meas$,
    annot: $sans("Meas")$,
    {
      Or[$"<MeasurementStart>" Field ...$][3 or 4 fields (shuffled)]
    },
  ),
  Prod(
    $Context$,
    annot: $sans("Context")$,
    {
      Or[$Meas ... Meas$][Sequence of measurements]
    },
  ),
)

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

Given a

=== Size & Training

In order to maximize the


=== Rationale



On: RIPE Atlas Latency Data
- Duration: 1 month
- Content: Measurements between Anchor nodes
  - Source IP(ipv4/6)
  - Dest IP (ipv4/6)
  - Optional timestamp (unix w/ second precision)
  - 1 or more latency measurements (milliseconds w/ floating point precision)

How:
- Given a sequence of measurements (src/dest ip, timestamp, measurements), predict distribution over next measurement latency between given src and dst ip (at optionally given timestamp)

For What Purpose?:
- Enable Peer-to-Peer networks to estimate latency between arbitrary nodes on the internet.


== Model Architecture



== Data Processing


== Training Details


#bibliography("bibliography.bib")
