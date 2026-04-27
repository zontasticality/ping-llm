"""
Matrix Factorization baselines for RTT prediction.

Two variants:
  DMFSGD  - paper-faithful (L1 loss, raw space, NMF). Liao et al., ToN 2013.
  BiasedMF - recommender-style (L2 loss, log-space, per-node biases).
"""

import math
import numpy as np


class DMFSGD:
    """DMFSGD: asymmetric NMF with L1 loss, line-search LR."""

    def __init__(self, embed_dim=10, lr=0.01, reg=1.0):
        self.embed_dim = embed_dim
        self.lr = lr
        self.reg = reg
        self.ip_to_idx = {}
        self.X = None
        self.Y = None
        self._scale = 1.0

    def _get_idx(self, ip_key):
        if ip_key not in self.ip_to_idx:
            self.ip_to_idx[ip_key] = len(self.ip_to_idx)
        return self.ip_to_idx[ip_key]

    def train(self, measurements, epochs=20, verbose=True):
        if not measurements:
            return

        for src, dst, _ in measurements:
            self._get_idx(src)
            self._get_idx(dst)

        self._scale = max(m[2] for m in measurements)
        normed = [(s, d, r / self._scale) for s, d, r in measurements]

        n_ips = len(self.ip_to_idx)
        d = self.embed_dim
        rng = np.random.RandomState(42)
        self.X = rng.uniform(0, 1, (n_ips, d)).astype(np.float64)
        self.Y = rng.uniform(0, 1, (n_ips, d)).astype(np.float64)

        indices = list(range(len(normed)))
        eta = self.lr

        for epoch in range(epochs):
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

                step_eta = eta
                for _ in range(5):
                    x_new = x_old + step_eta * (sign_e * y_old - self.reg * x_old)
                    y_new = y_old + step_eta * (sign_e * x_old - self.reg * y_old)
                    np.maximum(x_new, 0, out=x_new)
                    np.maximum(y_new, 0, out=y_new)
                    if abs(rtt_n - np.dot(x_new, y_new)) <= abs(error):
                        break
                    step_eta *= 0.5

                self.X[si] = x_new
                self.Y[di] = y_new

            mae_normed = total_ae / len(normed)
            if verbose and (epoch == 0 or epoch == epochs - 1):
                mae_ms = mae_normed * self._scale
                print(f"    DMFSGD epoch {epoch+1}/{epochs}: "
                      f"MAE={mae_ms:.2f} ms, eta={step_eta:.6f}")

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
