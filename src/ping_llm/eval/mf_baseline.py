"""
Matrix Factorization baselines for RTT prediction.

Two variants:
  DMFSGD  - paper-faithful (L1 loss, raw space, NMF). Liao et al., ToN 2013.
  BiasedMF - recommender-style (L2 loss, log-space, per-node biases).
"""

import math
from collections import defaultdict
import numpy as np


class DMFSGD:
    """DMFSGD: asymmetric NMF with L1 loss.

    The original paper normalizes RTTs to [0, 1]. In this dataset a few very
    large RTT outliers make max-scaling collapse typical targets toward zero,
    so we use a robust scale by default and clip only the normalized training
    target. Predictions remain unbounded above through the learned factors.
    """

    def __init__(self, embed_dim=10, lr=0.02, reg=0.001,
                 scale_quantile=0.99, random_state=42):
        self.embed_dim = embed_dim
        self.lr = lr
        self.reg = reg
        self.scale_quantile = scale_quantile
        self.random_state = random_state
        self.ip_to_idx = {}
        self.X = None
        self.Y = None
        self._scale = 1.0

    def _get_idx(self, ip_key):
        if ip_key not in self.ip_to_idx:
            self.ip_to_idx[ip_key] = len(self.ip_to_idx)
        return self.ip_to_idx[ip_key]

    @staticmethod
    def _unpack(measurement):
        if len(measurement) >= 4:
            return measurement[0], measurement[1], measurement[2], measurement[3]
        return measurement[0], measurement[1], measurement[2], None

    def train(self, measurements, epochs=20, verbose=True, shuffle=True):
        if not measurements:
            return

        unpacked = [self._unpack(m) for m in measurements]
        has_time = any(t is not None for _, _, _, t in unpacked)
        if has_time and not shuffle:
            def order_key(indexed_measurement):
                idx, measurement = indexed_measurement
                timestamp = measurement[3]
                return (timestamp is None, timestamp if timestamp is not None else 0, idx)

            unpacked = [m for _, m in sorted(enumerate(unpacked), key=order_key)]

        for src, dst, _, _ in unpacked:
            self._get_idx(src)
            self._get_idx(dst)

        rtts = np.array([m[2] for m in unpacked], dtype=np.float64)
        if self.scale_quantile >= 1.0:
            self._scale = float(np.max(rtts))
        else:
            self._scale = float(np.quantile(rtts, self.scale_quantile))
        self._scale = max(self._scale, 1e-6)
        normed = [(s, d, min(r / self._scale, 1.0)) for s, d, r, _ in unpacked]

        n_ips = len(self.ip_to_idx)
        d = self.embed_dim
        rng = np.random.RandomState(self.random_state)

        # Initialize dot products near the mean normalized RTT instead of O(d).
        target_mean = float(np.mean([r for _, _, r in normed]))
        init = math.sqrt(max(target_mean / d, 1e-8))
        self.X = rng.uniform(0.5 * init, 1.5 * init, (n_ips, d)).astype(np.float64)
        self.Y = rng.uniform(0.5 * init, 1.5 * init, (n_ips, d)).astype(np.float64)

        indices = list(range(len(normed)))
        eta = self.lr

        for epoch in range(epochs):
            if shuffle:
                rng.shuffle(indices)
            total_ae = 0.0

            for idx in indices:
                src_key, dst_key, rtt_n = normed[idx]
                si = self.ip_to_idx[src_key]
                di = self.ip_to_idx[dst_key]

                pred = np.dot(self.X[si], self.Y[di])
                error = rtt_n - pred
                sign_e = 1.0 if error > 0 else (-1.0 if error < 0 else 0.0)
                total_ae += abs(error)

                x_old = self.X[si].copy()
                y_old = self.Y[di].copy()

                self.X[si] = np.maximum(
                    x_old + eta * (sign_e * y_old - self.reg * x_old), 0.0
                )
                self.Y[di] = np.maximum(
                    y_old + eta * (sign_e * x_old - self.reg * y_old), 0.0
                )

            mae_normed = total_ae / len(normed)
            if verbose and (epoch == 0 or epoch == epochs - 1):
                mae_ms = mae_normed * self._scale
                print(f"    DMFSGD epoch {epoch+1}/{epochs}: "
                      f"MAE={mae_ms:.2f} ms, scale={self._scale:.2f}, "
                      f"eta={eta:.6f}")

    def predict_rtt(self, src_key, dst_key):
        """Predict RTT in ms for a (src, dst) pair. Returns None if unknown."""
        if src_key not in self.ip_to_idx or dst_key not in self.ip_to_idx:
            return None
        si = self.ip_to_idx[src_key]
        di = self.ip_to_idx[dst_key]
        return max(float(np.dot(self.X[si], self.Y[di])) * self._scale, 0.001)


class PaperDMFSGD:
    """Paper-style DMFSGD with L1 minibatch updates and line search.

    This keeps the mechanics from Liao et al. closer to the paper than the
    tuned ``DMFSGD`` class above: uniform [0, 1] initialization, lambda=1 by
    default, bounded per-node neighbor caches, optional age-decay weights,
    adaptive line search, and nonnegative projection.
    """

    def __init__(self, embed_dim=10, reg=1.0, eta_init=0.01,
                 line_search_steps=8, line_search_delta=1e-6,
                 neighbor_cap=32, scale_quantile=1.0,
                 use_decay=True, random_state=42):
        self.embed_dim = embed_dim
        self.reg = reg
        self.eta_init = eta_init
        self.line_search_steps = line_search_steps
        self.line_search_delta = line_search_delta
        self.neighbor_cap = neighbor_cap
        self.scale_quantile = scale_quantile
        self.use_decay = use_decay
        self.random_state = random_state
        self.ip_to_idx = {}
        self.X = None
        self.Y = None
        self._scale = 1.0

    def _get_idx(self, ip_key):
        if ip_key not in self.ip_to_idx:
            self.ip_to_idx[ip_key] = len(self.ip_to_idx)
        return self.ip_to_idx[ip_key]

    @staticmethod
    def _unpack(measurement):
        if len(measurement) >= 4:
            return measurement[0], measurement[1], measurement[2], measurement[3]
        return measurement[0], measurement[1], measurement[2], None

    def _cache_put(self, cache, owner_idx, neighbor_idx, rtt_n, step):
        neighbors = cache[owner_idx]
        neighbors[neighbor_idx] = (rtt_n, step)
        if len(neighbors) > self.neighbor_cap:
            oldest = min(neighbors, key=lambda k: neighbors[k][1])
            del neighbors[oldest]

    def _cache_arrays(self, neighbors, step):
        idxs = []
        rtts = []
        ages = []
        for idx, (rtt_n, seen_step) in neighbors.items():
            idxs.append(idx)
            rtts.append(rtt_n)
            ages.append(max(step - seen_step, 0))

        weights = np.ones(len(idxs), dtype=np.float64)
        if self.use_decay and len(idxs) > 1:
            ages = np.asarray(ages, dtype=np.float64)
            max_age = float(np.max(ages))
            weights = max_age - ages
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0:
                weights = np.ones(len(idxs), dtype=np.float64)
            else:
                weights = weights / weight_sum
                return (
                    np.asarray(idxs, dtype=np.int64),
                    np.asarray(rtts, dtype=np.float64),
                    weights,
                )
        weights /= float(len(weights))
        return (
            np.asarray(idxs, dtype=np.int64),
            np.asarray(rtts, dtype=np.float64),
            weights,
        )

    def _loss(self, vec, other, idxs, rtts, weights):
        preds = other[idxs] @ vec
        return float(np.sum(weights * np.abs(rtts - preds)) + self.reg * np.dot(vec, vec))

    def _update_vec(self, vec, other, idxs, rtts, weights):
        if len(idxs) == 0:
            return vec, 0.0

        before = self._loss(vec, other, idxs, rtts, weights)
        preds = other[idxs] @ vec
        signs = np.sign(rtts - preds)
        direction = (weights * signs) @ other[idxs]

        eta = self.eta_init
        best_vec = vec
        best_loss = before
        accepted_eta = 0.0
        for _ in range(self.line_search_steps):
            candidate = np.maximum((1.0 - eta * self.reg) * vec + eta * direction, 0.0)
            loss = self._loss(candidate, other, idxs, rtts, weights)
            if loss < best_loss:
                best_vec = candidate
                best_loss = loss
            if loss < before + self.line_search_delta:
                return candidate, eta
            eta *= 0.5
        return best_vec, accepted_eta

    def train(self, measurements, epochs=3, verbose=True):
        if not measurements:
            return

        unpacked = [self._unpack(m) for m in measurements]
        has_time = any(t is not None for _, _, _, t in unpacked)
        if has_time:
            def order_key(indexed_measurement):
                idx, measurement = indexed_measurement
                timestamp = measurement[3]
                return (timestamp is None, timestamp if timestamp is not None else 0, idx)

            unpacked = [m for _, m in sorted(enumerate(unpacked), key=order_key)]

        for src, dst, _, _ in unpacked:
            self._get_idx(src)
            self._get_idx(dst)

        rtts = np.array([m[2] for m in unpacked], dtype=np.float64)
        if self.scale_quantile >= 1.0:
            self._scale = float(np.max(rtts))
        else:
            self._scale = float(np.quantile(rtts, self.scale_quantile))
        self._scale = max(self._scale, 1e-6)
        normed = [
            (s, d, min(float(r) / self._scale, 1.0), t)
            for s, d, r, t in unpacked
        ]

        rng = np.random.RandomState(self.random_state)
        n_ips = len(self.ip_to_idx)
        self.X = rng.uniform(0.0, 1.0, (n_ips, self.embed_dim)).astype(np.float64)
        self.Y = rng.uniform(0.0, 1.0, (n_ips, self.embed_dim)).astype(np.float64)

        order = list(range(len(normed)))
        out_cache = defaultdict(dict)
        in_cache = defaultdict(dict)
        step = 0

        for epoch in range(epochs):
            if not has_time:
                rng.shuffle(order)
            total_ae = 0.0
            eta_sum = 0.0
            eta_count = 0

            for pos in order:
                src_key, dst_key, rtt_n, _ = normed[pos]
                si = self.ip_to_idx[src_key]
                di = self.ip_to_idx[dst_key]

                self._cache_put(out_cache, si, di, rtt_n, step)
                self._cache_put(in_cache, di, si, rtt_n, step)

                idxs, rtts_n, weights = self._cache_arrays(out_cache[si], step)
                self.X[si], eta = self._update_vec(self.X[si], self.Y, idxs, rtts_n, weights)
                if eta > 0:
                    eta_sum += eta
                    eta_count += 1

                idxs, rtts_n, weights = self._cache_arrays(in_cache[di], step)
                self.Y[di], eta = self._update_vec(self.Y[di], self.X, idxs, rtts_n, weights)
                if eta > 0:
                    eta_sum += eta
                    eta_count += 1

                total_ae += abs(rtt_n - np.dot(self.X[si], self.Y[di]))
                step += 1

            if verbose and (epoch == 0 or epoch == epochs - 1):
                mae_ms = (total_ae / len(normed)) * self._scale
                mean_eta = eta_sum / eta_count if eta_count else 0.0
                print(
                    f"    PaperDMFSGD epoch {epoch+1}/{epochs}: "
                    f"MAE={mae_ms:.2f} ms, scale={self._scale:.2f}, "
                    f"mean_eta={mean_eta:.6f}"
                )

    def predict_rtt(self, src_key, dst_key):
        """Predict RTT in ms for a (src, dst) pair. Returns None if unknown."""
        if src_key not in self.ip_to_idx or dst_key not in self.ip_to_idx:
            return None
        si = self.ip_to_idx[src_key]
        di = self.ip_to_idx[dst_key]
        return max(float(np.dot(self.X[si], self.Y[di])) * self._scale, 0.001)


class BiasedMF:
    """Log-space biased MF with L2 loss (recommender-style)."""

    def __init__(self, embed_dim=16, lr=0.01, reg=0.1):
        self.embed_dim = embed_dim
        self.lr = lr
        self.reg = reg
        self.ip_to_idx = {}
        self.X = None
        self.Y = None
        self.bias_src = None
        self.bias_dst = None
        self.global_bias = 0.0

    def _get_idx(self, ip_key):
        if ip_key not in self.ip_to_idx:
            self.ip_to_idx[ip_key] = len(self.ip_to_idx)
        return self.ip_to_idx[ip_key]

    def train(self, measurements, epochs=10, verbose=True):
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

            rmse = math.sqrt(total_se / len(measurements))
            if verbose and (epoch == 0 or epoch == epochs - 1):
                print(f"    BiasedMF epoch {epoch+1}/{epochs}: log-RMSE={rmse:.4f}")

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
