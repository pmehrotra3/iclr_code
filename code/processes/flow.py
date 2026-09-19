"""
processes/flow.py — Flow Matching with the Optimal-Transport conditional path.

Lipman et al. (2022), Example II. Time convention: t = 0 is noise, t = 1 is data.

Conditional path (Eq. 20):  mu_t(x1) = t x1,  sigma_t = 1 - (1 - sigma_min) t
Conditional flow (Eq. 22):  psi_t(x0) = (1 - (1 - sigma_min) t) x0 + t x1
CFM target        (Eq. 23):  u_t = x1 - (1 - sigma_min) x0
Sampling:                    x0 ~ N(0, I); integrate d phi / dt = v_t(phi) over t in [0, 1]

The network is a velocity field. The analytic reference field is the MARGINAL OT velocity of
the GMM in closed form, INCLUDING the within-mode variance (see _true_velocity); dropping it
(point-mass modes) makes the field 1/sigma_min-stiff at t = 1 and the backtrack non-invertible.
`true_field_forward` / `true_field_backtrack` integrate it with `true_solver`.
"""
from __future__ import annotations
import torch

import core
from processes.base import Process


def _get(cfg, key, default):
    """Read `key` from an OmegaConf/attr config, tolerating None/missing (returns default)."""
    if cfg is None:
        return default
    v = getattr(cfg, key, default)
    return default if v is None else v


class FlowOTProcess(Process):
    name = "flow"

    def __init__(self, means_t, variance, T, device, cfg=None):
        super().__init__(means_t, variance, T, device, cfg)
        proc = getattr(cfg, "process", None) if cfg is not None else None
        self.sigma_min = float(getattr(proc, "sigma_min", 1e-4)) if proc is not None else 1e-4
        self.solver = str(getattr(proc, "solver", "euler")) if proc is not None else "euler"
        # exact-velocity passes (atlas backtrack / forward); defaults to the learned solver
        self.true_solver = str(getattr(proc, "true_solver", self.solver)) if proc is not None else self.solver
        self.ts = torch.linspace(0.0, 1.0, T, device=device)   # uniform time grid on [0, 1]

    # ---- network: same backbone, interpreted as a velocity field ----
    def build_model(self, d):
        return core.ScoreNet(d).to(self.device)

    def train_model(self, K, d, n_steps, lr, batch, seed):
        torch.manual_seed(seed)
        model = self.build_model(d)
        sigma0 = self.variance ** 0.5
        oms = 1.0 - self.sigma_min

        def step():
            k = torch.randint(0, K, (batch,), device=self.device)
            x1 = self.means_t[k] + sigma0 * torch.randn(batch, d, device=self.device)
            x0 = torch.randn(batch, d, device=self.device)
            t = torch.rand(batch, device=self.device)                      # U[0,1]
            psi = (1 - oms * t)[:, None] * x0 + t[:, None] * x1            # Eq. 22
            target = x1 - oms * x0                                         # Eq. 23
            ti = (t * (self.T - 1)).round().long().clamp(0, self.T - 1)    # embedding index
            return ((model(psi, ti) - target) ** 2).mean()

        # same cosine-decay + grad-clip + EMA recipe as the DDIM score (Step 1)
        t = getattr(self.cfg, "train", None)
        return core.run_optimizer(
            model, step, n_steps, lr,
            lr_min=_get(t, "lr_min", None), grad_clip=_get(t, "grad_clip", None),
            ema_decay=_get(t, "ema_decay", None), ema_warmup=int(_get(t, "ema_warmup", 0) or 0),
        )

    # ---- ODE solvers ----
    def _step(self, f, x, t, dt, solver=None):
        solver = solver or self.solver
        if solver == "euler":
            return x + dt * f(x, t)
        if solver == "heun":
            k1 = f(x, t)
            return x + 0.5 * dt * (k1 + f(x + dt * k1, t + dt))
        if solver == "midpoint":
            k1 = f(x, t)
            return x + dt * f(x + 0.5 * dt * k1, t + 0.5 * dt)
        if solver == "rk4":
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"unknown solver {solver!r}")

    def _integrate(self, f, X0, chunk, solver=None):
        """Integrate f FORWARD, t: 0 -> 1."""
        dt = 1.0 / (self.T - 1)
        X = X0.clone()
        for i in range(self.T - 1):
            t = float(self.ts[i])
            for s in range(0, X.shape[0], chunk):
                X[s:s + chunk] = self._step(f, X[s:s + chunk], t, dt, solver)
        return X

    @torch.no_grad()
    def sample(self, model, X0, chunk=50000):
        """Integrate the LEARNED velocity forward, t: 0 -> 1 (noise -> data)."""
        model.eval()

        def f(x, t):
            idx = max(0, min(self.T - 1, int(round(float(t) * (self.T - 1)))))
            ti = torch.full((x.shape[0],), idx, dtype=torch.long, device=self.device)
            return model(x, ti)
        return self._integrate(f, X0, chunk)

    # ---- analytic marginal OT velocity of the GMM (with within-mode variance) ----
    @torch.no_grad()
    def _true_velocity(self, x, t):
        """Marginal OT velocity at (x, t).

        x_t = s_t x0 + t x1 with x0 ~ N(0, I), x1 ~ N(mu_k, sigma^2 I), so
        x_t | k ~ N(t mu_k, (t^2 sigma^2 + s_t^2) I). Marginal velocity is
        (E[x1 | x] - (1 - sigma_min) x) / s_t with
        E[x1 | x] = sum_k r_k m_k,  m_k = mu_k + t sigma^2 (x - t mu_k) / var_t,
        var_t = t^2 sigma^2 + s_t^2, and r_k the responsibilities under the component marginals.
        Keeping the sigma^2 terms is what makes the field finite at t = 1 and the pass invertible.
        """
        oms = 1.0 - self.sigma_min
        st = max(1.0 - oms * t, 1e-6)
        M = self.means_t                                     # (K, d)
        var_t = t * t * self.variance + st * st              # per-component marginal variance
        d2 = torch.cdist(x, t * M) ** 2                      # (N, K)
        r = torch.softmax(-d2 / (2 * var_t), dim=1)          # (N, K)
        gain = t * self.variance / var_t
        # sum_k r_k m_k = (1 - t gain)(r @ M) + gain x   (since m_k = mu_k + gain (x - t mu_k))
        m_bar = (1 - t * gain) * (r @ M) + gain * x
        return (m_bar - oms * x) / st

    @torch.no_grad()
    def true_field_forward(self, X0, chunk=50000):
        """Integrate the analytic marginal velocity FORWARD, t: 0 -> 1 (noise -> data)."""
        return self._integrate(self._true_velocity, X0, chunk, self.true_solver)

    @torch.no_grad()
    def true_field_backtrack(self, Pd, chunk=50000):
        """Integrate the analytic marginal velocity BACKWARD, t: 1 -> 0 (data -> noise)."""
        dt = 1.0 / (self.T - 1)
        X = Pd.clone()
        for i in reversed(range(1, self.T)):
            t = float(self.ts[i])
            for s in range(0, X.shape[0], chunk):
                X[s:s + chunk] = self._step(self._true_velocity, X[s:s + chunk], t, -dt, self.true_solver)
        return X
