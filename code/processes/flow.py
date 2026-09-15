"""
processes/flow.py — Flow Matching with the Optimal-Transport conditional path.

Implements Lipman et al. (2022), "Flow Matching for Generative Modeling", Example II
(Optimal Transport paths). Paper time convention: t = 0 is noise, t = 1 is data.

Conditional OT path (Eq. 20):   mu_t(x1) = t * x1,   sigma_t(x1) = 1 - (1 - sigma_min) * t
Conditional flow / sample (Eq. 22):  psi_t(x0) = (1 - (1 - sigma_min) t) x0 + t x1
CFM target velocity      (Eq. 23):   u_t = x1 - (1 - sigma_min) x0
Training loss            (Eq. 23):   || v_t(psi_t(x0); theta) - (x1 - (1 - sigma_min) x0) ||^2
Sampling:                            x0 ~ N(0, I); integrate d phi/dt = v_t(phi) on t in [0, 1].

The network is a VELOCITY field v_t(x; theta) (not eps). Time t in [0, 1] is passed to the
same ScoreNet backbone via its sinusoidal embedding, scaled to the [0, T) index range the
embedding expects.

Analytic reference field (for the atlas): the MARGINAL OT velocity of the GMM has a closed
form. For an equally/known-weighted isotropic mixture the marginal velocity is the
responsibility-weighted average of the per-component conditional velocities evaluated with
that component as x1. We integrate it backward (t = 1 -> 0) to carry data-space points to
seed space -- the flow analogue of the true-score backtrack.
"""
from __future__ import annotations
import torch
import torch.nn as nn

import core
from processes.base import Process


class FlowOTProcess(Process):
    name = "flow"

    def __init__(self, means_t, variance, T, device, cfg=None):
        super().__init__(means_t, variance, T, device, cfg)
        self.sigma_min = float(getattr(getattr(cfg, "flow", {}), "sigma_min", 1e-4)) \
            if cfg is not None else 1e-4
        self.solver = str(getattr(getattr(cfg, "flow", {}), "solver", "euler")) \
            if cfg is not None else "euler"
        # uniform time grid on [0, 1]
        self.ts = torch.linspace(0.0, 1.0, T, device=device)

    # ---- network: same backbone, interpreted as a velocity field ----
    def build_model(self, d):
        return core.ScoreNet(d).to(self.device)

    def _t_index(self, t_scalar):
        """Map continuous t in [0,1] to the [0, T) float index the time-embedding expects."""
        return torch.tensor(t_scalar * (self.T - 1), device=self.device)

    def train_model(self, K, d, n_steps, lr, batch, seed):
        torch.manual_seed(seed)
        model = self.build_model(d)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sigma0 = self.variance ** 0.5
        oms = 1.0 - self.sigma_min
        for _ in range(n_steps):
            k = torch.randint(0, K, (batch,), device=self.device)
            x1 = self.means_t[k] + sigma0 * torch.randn(batch, d, device=self.device)
            x0 = torch.randn(batch, d, device=self.device)
            t = torch.rand(batch, device=self.device)                      # U[0,1]
            psi = (1 - oms * t)[:, None] * x0 + t[:, None] * x1            # Eq. 22
            target = x1 - oms * x0                                         # Eq. 23
            ti = (t * (self.T - 1)).round().long().clamp(0, self.T - 1)    # embedding index
            v = model(psi, ti)
            loss = ((v - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return model

    # ---- ODE step helpers ----
    def _step(self, f, x, t, dt):
        if self.solver == "euler":
            return x + dt * f(x, t)
        if self.solver == "midpoint":
            k1 = f(x, t)
            return x + dt * f(x + 0.5 * dt * k1, t + 0.5 * dt)
        if self.solver == "rk4":
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"unknown solver '{self.solver}'")

    @torch.no_grad()
    def sample(self, model, X0, chunk=50000):
        """Integrate the LEARNED velocity forward, t: 0 -> 1 (noise -> data)."""
        model.eval()
        dt = 1.0 / (self.T - 1)
        X = X0.clone()

        def f(x, t):
            ti = (t * (self.T - 1)).round().long().clamp(0, self.T - 1)
            ti = torch.full((x.shape[0],), int(ti), dtype=torch.long, device=self.device)
            return model(x, ti)

        for i in range(self.T - 1):
            t = float(self.ts[i])
            for s in range(0, X.shape[0], chunk):
                X[s:s + chunk] = self._step(f, X[s:s + chunk], t, dt)
        return X

    # ---- analytic marginal OT velocity of the GMM ----
    @torch.no_grad()
    def _true_velocity(self, x, t):
        """
        Marginal OT velocity at (x, t) for the isotropic GMM reference.
        Conditional path: psi_t(x0|x1) with mu_t=t x1, sigma_t=1-(1-smin)t.
        p_t(x|x1) = N(x; t x1, sigma_t^2 I). Responsibility r_i favors component i.
        Conditional velocity toward x1=mu_i:  u_i = (x1 - (1-smin) * x0_hat), but expressed
        in x directly:  u_t(x|x1) = (x1 - (1-smin) x) / (1 - (1-smin) t)   (Eq. 21).
        Marginal velocity = sum_i r_i(x,t) u_t(x|mu_i).
        """
        oms = 1.0 - self.sigma_min
        st = 1.0 - oms * t                                   # sigma_t (scalar)
        st = max(st, 1e-6)
        M = self.means_t                                     # (K, d)
        # responsibilities under N(x; t mu_i, sigma_t^2 I)
        d2 = torch.cdist(x, t * M) ** 2                      # (N, K)
        r = torch.softmax(-d2 / (2 * st * st), dim=1)        # (N, K)
        # conditional velocity per component: (mu_i - (1-smin) x) / st
        # marginal = sum_i r_i (mu_i - oms x)/st = (r @ M - oms x) / st
        return (r @ M - oms * x) / st

    @torch.no_grad()
    def true_field_backtrack(self, Pd, chunk=50000):
        """Integrate the analytic marginal velocity BACKWARD, t: 1 -> 0 (data -> noise)."""
        dt = 1.0 / (self.T - 1)
        X = Pd.clone()
        for i in reversed(range(1, self.T)):
            t = float(self.ts[i])
            for s in range(0, X.shape[0], chunk):
                xs = X[s:s + chunk]
                X[s:s + chunk] = self._step(self._true_velocity, xs, t, -dt)
        return X
