"""
Muon optimizer with fallback to standalone implementation.

Tries to import from torch.optim (PyTorch >= 2.6), falls back to
a standalone implementation based on KellerJordan's Muon.
"""

try:
    from torch.optim import Muon  # PyTorch >= 2.6
except ImportError:
    import torch
    from torch.optim import Optimizer

    class Muon(Optimizer):
        """
        Muon optimizer: Momentum + Orthogonalization via Newton-Schulz.

        Designed for 2D weight matrices. Uses Newton-Schulz iterations
        to approximately orthogonalize the momentum, which acts as a
        spectral steepest descent step.

        Based on: https://github.com/KellerJordan/Muon
        """

        def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                     ns_steps=5):
            defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                            ns_steps=ns_steps)
            super().__init__(params, defaults)

        @staticmethod
        def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
            """Approximate orthogonalization via Newton-Schulz iteration."""
            assert G.ndim == 2
            a, b, c = (3.4445, -4.7750, 2.0315)
            # Ensure tall matrix for numerical stability
            transpose = False
            if G.shape[0] < G.shape[1]:
                G = G.T
                transpose = True
            # Normalize
            G = G / (G.norm() + 1e-7)
            for _ in range(steps):
                A = G @ G.T
                G = a * G + b * (A @ G) + c * (A @ (A @ G))
            if transpose:
                G = G.T
            return G

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.ndim != 2:
                        # Fallback to SGD+momentum for non-2D params
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g)
                        if nesterov:
                            g = g + momentum * buf
                        else:
                            g = buf
                        p.add_(g, alpha=-lr)
                        continue

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        update = g + momentum * buf
                    else:
                        update = buf

                    # Newton-Schulz orthogonalization
                    update = self._newton_schulz(update, steps=ns_steps)
                    # Scale by max(m,n) to match gradient scale
                    scale = max(update.shape[0], update.shape[1]) ** 0.5
                    p.add_(update, alpha=-lr * scale)

            return loss
