# Probe-Centric Big-Row Pipeline Plan (Plan 3)

Assumptions:
 - We have a database of 200M latency measurements in the form of a parquet file
   - This is sampled from a database of 35B latency measurements

Preprocess:
 - We want to take this database and shard it into a fixed amount of arrayrecords that can be individually read from and processed for the purpose of feeding the transformer model.
 - We will group the measurements in the parquet by src_addr, ordered by event_time, and serialize all the fields, taking care to make it be as similarly information-dense as the a series of tokens
   - i.e. we will translate the f32 latency into a 2-byte latency in the same format as the tokenizer accepts. Same with representing the timestamp as a single u64 representing unix epoch seconds
 - If the length of a single row exceeds 8MB while writing to a row, we will round off at the nearest measurement and start writing to a new row. This is to minimize I/O overhead when streaming from arrayrecord as with big rows there will likely be much more overhead / delay loading and unloading large blocks from memory. We will randomly choose rows (as opposed to random src_addr) because it is easier and prioritizes using all the data (as opposed to focusing more on rare probes). The 8MB is arbitrary but it seems like a good starting point.

Training:
 - During training, the grain loader will deterministically load from random rows
 - After having loaded a row, it will continually produce a number of samples based on the number of measurements in a row, ideally interleaved with the generations from other cores.
 - Given a row, produce `K = min(ceil(#row measurements / 30), 16)` context windows (for small this will be minimum 1, for large this is capped at 16).
 - To produce a tokenized context window from a row list of measurements:
   - Generate a buffer of measurements to tokenize:
     - If number of measurements is `< 1024/30`, if so, we just tokenize the whole thing directly, if we have access measurements we produce another full context (but padded) with the excess until we run out of measurements.
       - Note: `1024/30` is a heuristic for the maximum number of measurements that can fit in a window on average (1024 tokens / 30 tokens per measurement possible avg). It doesn't matter that this is particularly precise as at most it will just cause 1 or 2 more context windows to be generated.
     - If we have more than that amount of measurements, we sample a window size from a log-uniform distribution clamped at the maximum measurement count. We pick an offset from the range `0..num_measurements-window_size` and then we randomly pick items from the selected measurement window and put them in a buffer in timestamp order until we have enough tokens to fill the 1024 token context window. Repeat this until we have satisfied producing K context windows.
   - Once you have a buffer of measurements (and you've determined whether we want to use the entire buffer, or just use the buffer enough to fill up a single context window)
     - Determine whether we are doing full timestamps (40%) partial timestamps (30%) or no timestamps (30%).
       - If partial timestamps, extract a percentage from 10-90% of measurements, and then during tokenization, tokenize either a non-extracted measurement in timestamp order, or tokenize an extracted measurement without the timestamp.
       - If no timestamp, tokenize from the buffer in a random order (but make sure to use all the measurements)
     - When you tokenize a given measurement, tokenize the list of fields in a random order so the model can learn a joint distribution, if you encode a timestamp and there is a previous measurement with a timestamp, encode the timestamp as a delta from the previous timestamp using the various timestamp delta tokens.

