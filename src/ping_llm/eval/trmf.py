"""
Temporal Regularized Matrix Factorization (TRMF) baseline.

Y ≈ F @ X with autoregressive regularization on the temporal factors X.
Operates in log-RTT space with per-pair z-score normalization.
"""

import math
import numpy as np


class TRMF:
    def __init__(self, K=20, lags=None, bin_width_sec=900,
                 lambda_f=1.0, lambda_x=100.0, eta=0.5,
                 alpha=500.0, lambda_w=1.0, lr=1e-4, n_iters=10000):
        self.K = K
        self.lags = lags or [1, 2, 4, 96, 672]
        self.bin_width_sec = bin_width_sec
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.eta = eta
        self.alpha = alpha
        self.lambda_w = lambda_w
        self.lr = lr
        self.n_iters = n_iters

        self.pair_to_idx = None
        self.min_time = None
        self.T = None
        self.F = None
        self.X = None
        self.W = None
        self.row_mean = None
        self.row_std = None

    def fit(self, measurements):
        """
        Train TRMF on timestamped RTT measurements.

        Args:
            measurements: list of (src_key, dst_key, rtt_ms, timestamp_sec)
        """
        pairs = {}
        for src, dst, rtt, ts in measurements:
            if ts is None or rtt <= 0:
                continue
            key = (src, dst)
            if key not in pairs:
                pairs[key] = len(pairs)
        self.pair_to_idx = pairs
        n_pairs = len(pairs)
        if n_pairs == 0:
            return

        timestamps = [ts for _, _, rtt, ts in measurements if ts is not None and rtt > 0]
        self.min_time = min(timestamps)
        self.T = max(int((max(timestamps) - self.min_time) / self.bin_width_sec) + 1, 1)
        T = self.T
        K = min(self.K, n_pairs, T)
        self.K = K

        # Build observation matrix in log-space, averaging within bins
        Y = np.zeros((n_pairs, T))
        counts = np.zeros((n_pairs, T))
        for src, dst, rtt, ts in measurements:
            if ts is None or rtt <= 0:
                continue
            i = self.pair_to_idx.get((src, dst))
            if i is None:
                continue
            t = min(int((ts - self.min_time) / self.bin_width_sec), T - 1)
            Y[i, t] += math.log(max(rtt, 0.001))
            counts[i, t] += 1

        obs = counts > 0
        Y[obs] /= counts[obs]
        M = obs.astype(np.float64)

        self.row_mean = np.zeros(n_pairs)
        self.row_std = np.ones(n_pairs)
        for i in range(n_pairs):
            vals = Y[i, M[i] > 0]
            if len(vals) > 1:
                self.row_mean[i] = vals.mean()
                self.row_std[i] = max(vals.std(), 1e-6)
                Y[i, M[i] > 0] = (vals - self.row_mean[i]) / self.row_std[i]
            elif len(vals) == 1:
                self.row_mean[i] = vals[0]
                Y[i, M[i] > 0] = 0.0

        rng = np.random.RandomState(42)
        self.F = rng.randn(n_pairs, K) / np.sqrt(K)
        self.X = rng.randn(K, T) / np.sqrt(K)
        n_lags = len(self.lags)
        self.W = np.full((n_lags, K), 1.0 / max(n_lags, 1))

        active = [(idx, l) for idx, l in enumerate(self.lags) if l < T]
        active_indices = [idx for idx, _ in active]
        print(f"    TRMF: {n_pairs} pairs, {T} bins, K={K}, "
              f"{len(active)}/{n_lags} active lags, "
              f"{int(M.sum())}/{n_pairs * T} observed entries")

        for it in range(self.n_iters):
            R = M * (Y - self.F @ self.X)

            E = self.X.copy()
            for l_idx, l in active:
                E[:, l:] -= self.W[l_idx][:, np.newaxis] * self.X[:, :T - l]

            grad_F = -2.0 * R @ self.X.T + 2.0 * self.lambda_f * self.F

            grad_X = -2.0 * self.F.T @ R + 2.0 * self.eta * self.X
            grad_X += 2.0 * self.lambda_x * E
            for l_idx, l in active:
                grad_X[:, :T - l] -= (
                    2.0 * self.lambda_x * self.W[l_idx][:, np.newaxis] * E[:, l:]
                )

            W_sum = (np.sum(self.W[active_indices], axis=0)
                     if active_indices else np.zeros(K))
            for l_idx, l in active:
                grad_W = -2.0 * self.lambda_x * np.sum(
                    self.X[:, :T - l] * E[:, l:], axis=1,
                )
                grad_W += 2.0 * self.lambda_w * self.W[l_idx]
                grad_W += 2.0 * self.alpha * (W_sum - 1.0)
                self.W[l_idx] -= self.lr * grad_W

            self.F -= self.lr * grad_F
            self.X -= self.lr * grad_X

            if (it + 1) % 2000 == 0 or it == 0:
                recon = float(np.sum(R ** 2))
                ar = float(np.sum(E ** 2))
                print(f"    TRMF iter {it + 1}/{self.n_iters}: "
                      f"recon={recon:.1f}, AR={ar:.1f}")

    def predict(self, src_key, dst_key, timestamp_sec=None):
        """Predict RTT in ms. Returns None for unseen pairs."""
        if self.pair_to_idx is None:
            return None
        pair = (src_key, dst_key)
        if pair not in self.pair_to_idx:
            return None
        i = self.pair_to_idx[pair]

        if timestamp_sec is not None and self.min_time is not None:
            t = int((timestamp_sec - self.min_time) / self.bin_width_sec)
            t = max(0, min(t, self.T - 1))
        else:
            t = self.T - 1

        z = float(self.F[i] @ self.X[:, t])
        log_rtt = z * self.row_std[i] + self.row_mean[i]
        return math.exp(min(log_rtt, 20.0))
