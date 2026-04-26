"""
Biased Matrix Factorization baseline for RTT prediction.

Each IP gets separate source and destination embedding vectors (handles
asymmetric latency). Predicted log(RTT) = dot(X[src], Y[dst]) + bias_src
+ bias_dst + global_bias.

Based on DMFSGD (Liao et al., IEEE/ACM ToN 2013). Internet RTT matrices
are strongly low-rank: r=16 captures ~95% of spectral energy.
"""

import math
import numpy as np
from collections import defaultdict

from ping_llm.data.tokenization import (
    MEASUREMENT_START, RTT_START, FAILED,
    SRC_IPV4, SRC_IPV6, DST_IPV4, DST_IPV6,
    token_to_byte, BYTE_TOKEN_OFFSET,
    decode_rtt_exponent_mantissa,
)
from ping_llm.eval.token_classify import ROLE_BYTE_COUNTS


def _decode_ip_from_tokens(role_token, byte_tokens):
    """Decode IP address from role token + byte tokens to a hashable tuple."""
    return (role_token, tuple(int(t) for t in byte_tokens))


def extract_measurements_from_sequences(sequences):
    """
    Extract (src_ip_key, dst_ip_key, rtt_ms) triples from token sequences.

    Returns list of (src_key, dst_key, rtt_ms) where keys are hashable
    tuples of (role_token, byte_tuple).
    """
    measurements = []
    for tokens in sequences:
        tokens = [int(t) for t in tokens]
        i = 0
        n = len(tokens)
        current_src = None
        current_dst = None
        current_rtt = None

        while i < n:
            t = tokens[i]
            if t == MEASUREMENT_START:
                if current_src is not None and current_dst is not None and current_rtt is not None and current_rtt > 0:
                    measurements.append((current_src, current_dst, current_rtt))
                current_src = None
                current_dst = None
                current_rtt = None
                i += 1
                continue

            if t in (SRC_IPV4, SRC_IPV6):
                nbytes = 4 if t == SRC_IPV4 else 16
                if i + nbytes < n:
                    current_src = _decode_ip_from_tokens(t, tokens[i+1:i+1+nbytes])
                i += 1 + nbytes
                continue

            if t in (DST_IPV4, DST_IPV6):
                nbytes = 4 if t == DST_IPV4 else 16
                if i + nbytes < n:
                    current_dst = _decode_ip_from_tokens(t, tokens[i+1:i+1+nbytes])
                i += 1 + nbytes
                continue

            if t == RTT_START and i + 2 < n:
                try:
                    b1 = token_to_byte(tokens[i+1])
                    b2 = token_to_byte(tokens[i+2])
                    current_rtt = decode_rtt_exponent_mantissa(b1, b2)
                except Exception:
                    pass
                i += 3
                continue

            if t == FAILED:
                current_rtt = -1.0
                i += 1
                continue

            if t in ROLE_BYTE_COUNTS:
                i += 1 + ROLE_BYTE_COUNTS[t]
            else:
                i += 1

        if current_src is not None and current_dst is not None and current_rtt is not None and current_rtt > 0:
            measurements.append((current_src, current_dst, current_rtt))

    return measurements


class BiasedMF:
    """Biased matrix factorization for RTT prediction in log-space."""

    def __init__(self, embed_dim=16, lr=0.01, reg=0.1):
        self.embed_dim = embed_dim
        self.lr = lr
        self.reg = reg
        self.ip_to_idx = {}
        self.X = None  # source embeddings
        self.Y = None  # dest embeddings
        self.bias_src = None
        self.bias_dst = None
        self.global_bias = 0.0

    def _get_idx(self, ip_key):
        if ip_key not in self.ip_to_idx:
            self.ip_to_idx[ip_key] = len(self.ip_to_idx)
        return self.ip_to_idx[ip_key]

    def train(self, measurements, epochs=10, verbose=True):
        """
        Train on list of (src_key, dst_key, rtt_ms) triples.
        """
        if not measurements:
            return

        for src, dst, _ in measurements:
            self._get_idx(src)
            self._get_idx(dst)

        n_ips = len(self.ip_to_idx)
        d = self.embed_dim
        rng = np.random.RandomState(42)
        self.X = rng.uniform(0, 0.1, (n_ips, d)).astype(np.float64)
        self.Y = rng.uniform(0, 0.1, (n_ips, d)).astype(np.float64)
        self.bias_src = np.zeros(n_ips, dtype=np.float64)
        self.bias_dst = np.zeros(n_ips, dtype=np.float64)

        log_rtts = [math.log(max(rtt, 0.001)) for _, _, rtt in measurements]
        self.global_bias = float(np.mean(log_rtts))

        indices = list(range(len(measurements)))
        for epoch in range(epochs):
            rng.shuffle(indices)
            total_se = 0.0
            for idx in indices:
                src_key, dst_key, rtt_ms = measurements[idx]
                si = self.ip_to_idx[src_key]
                di = self.ip_to_idx[dst_key]
                log_rtt = math.log(max(rtt_ms, 0.001))

                pred = (np.dot(self.X[si], self.Y[di])
                        + self.bias_src[si] + self.bias_dst[di]
                        + self.global_bias)
                error = log_rtt - pred
                total_se += error ** 2

                x_old = self.X[si].copy()
                y_old = self.Y[di].copy()

                self.X[si] += self.lr * (error * y_old - self.reg * x_old)
                self.Y[di] += self.lr * (error * x_old - self.reg * y_old)
                self.bias_src[si] += self.lr * (error - self.reg * self.bias_src[si])
                self.bias_dst[di] += self.lr * (error - self.reg * self.bias_dst[di])

                np.maximum(self.X[si], 0, out=self.X[si])
                np.maximum(self.Y[di], 0, out=self.Y[di])

            rmse = math.sqrt(total_se / len(measurements))
            if verbose and (epoch == 0 or epoch == epochs - 1):
                print(f"    MF epoch {epoch+1}/{epochs}: log-RMSE={rmse:.4f}")

    def predict_rtt(self, src_key, dst_key):
        """Predict RTT in ms for a (src, dst) pair. Returns None if unknown."""
        if src_key not in self.ip_to_idx or dst_key not in self.ip_to_idx:
            return None
        si = self.ip_to_idx[src_key]
        di = self.ip_to_idx[dst_key]
        log_pred = (np.dot(self.X[si], self.Y[di])
                    + self.bias_src[si] + self.bias_dst[di]
                    + self.global_bias)
        return math.exp(log_pred)
