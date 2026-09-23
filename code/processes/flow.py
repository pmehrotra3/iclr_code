"""processes/flow.py — flow matching with the optimal-transport path (Lipman et al. 2022, Ex. II).

t = 0 is noise, t = 1 is data. With s_t = 1 - (1 - sigma_min) t:
    path    psi_t(x0) = s_t x0 + t x1                       (Eq. 22)
    target  u_t       = x1 - (1 - sigma_min) x0             (Eq. 23)
The network is a velocity field, integrated t: 0 -> 1 to sample. The exact field is the
marginal OT velocity of the GMM in closed form, keeping the within-mode variance (without it
the field is 1/sigma_min-stiff at t = 1 and the backtrack is not invertible).
"""
from __future__ import annotations

import torch

import core
from processes.base import Process


class FlowOTProcess(Process):
    name = "flow"

    def __init__(self, means_t, variance, T, device, cfg, weights=None):
        super().__init__(means_t, variance, T, device, cfg, weights)
        p = cfg.process
        self.sigma_min = float(p.sigma_min)
        self.solver = str(p.solver)                             # learned sampler
        self.true_solver = str(p.get("true_solver") or self.solver)   # exact field
        self.ts = torch.linspace(0.0, 1.0, T, device=device)

    def train_closure(self, K, d, batch, seed):
        torch.manual_seed(seed)
        model = core.ScoreNet(d, **self.arch(d)).to(self.device)
        oms = 1.0 - self.sigma_min
        g = core.step_generator(seed, self.device)
        draw = core._minibatch(self.means_t, self.variance, batch, self.n_train(), seed,
                               self.device, self.weights, g)

        def step():
            x1 = draw()
            x0 = torch.randn(batch, d, generator=g, device=self.device)
            t = torch.rand(batch, generator=g, device=self.device)
            psi = (1 - oms * t)[:, None] * x0 + t[:, None] * x1            # Eq. 22
            target = x1 - oms * x0                                         # Eq. 23
            ti = (t * (self.T - 1)).round().long().clamp(0, self.T - 1)    # time-embedding index
            return ((model(psi, ti) - target) ** 2).mean()

        step.generator = g
        return model, step

    # ---- ODE integration on the uniform grid self.ts ----
    def _step(self, f, x, t, dt, solver):
        if solver == "euler":
            return x + dt * f(x, t)
        if solver == "heun":
            k1 = f(x, t)
            return x + 0.5 * dt * (k1 + f(x + dt * k1, t + dt))
        if solver == "midpoint":
            return x + dt * f(x + 0.5 * dt * f(x, t), t + 0.5 * dt)
        if solver == "rk4":
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"unknown solver {solver!r}")

    def _integrate(self, f, X, chunk, solver, backward=False, inplace=False):
        """Integrate dx/dt = f(x, t) over the grid, t: 0 -> 1 (or 1 -> 0 when backward)."""
        dt = 1.0 / (self.T - 1)
        X = X if inplace else X.clone()
        steps = reversed(range(1, self.T)) if backward else range(self.T - 1)
        for i in steps:
            t = float(self.ts[i])
            for s in range(0, X.shape[0], chunk):
                X[s:s + chunk] = self._step(f, X[s:s + chunk], t, -dt if backward else dt, solver)
        return X

    @torch.no_grad()
    def sample(self, model, X0, chunk=50000):
        model.eval()

        def f(x, t):
            idx = max(0, min(self.T - 1, int(round(float(t) * (self.T - 1)))))
            return model(x, torch.full((x.shape[0],), idx, dtype=torch.long, device=self.device))
        return self._integrate(f, X0, chunk, self.solver)

    @torch.no_grad()
    def _true_velocity(self, x, t):
        """Marginal OT velocity (E[x1 | x_t = x] - (1 - sigma_min) x) / s_t of the GMM.

        x_t | k ~ N(t mu_k, var_t I) with var_t = t^2 sigma^2 + s_t^2, so
        E[x1 | x, k] = mu_k + gain (x - t mu_k) with gain = t sigma^2 / var_t, and the modes are
        weighted by their posterior responsibilities r_k (mixing weights included)."""
        oms = 1.0 - self.sigma_min
        st = max(1.0 - oms * t, 1e-6)
        M = self.means_t
        var_t = t * t * self.variance + st * st
        logits = -torch.cdist(x, t * M) ** 2 / (2 * var_t)
        if self.logw is not None:
            logits = logits + self.logw
        r = torch.softmax(logits, dim=1)
        gain = t * self.variance / var_t
        m_bar = (1 - t * gain) * (r @ M) + gain * x               # = sum_k r_k E[x1 | x, k]
        return (m_bar - oms * x) / st

    @torch.no_grad()
    def true_field_forward(self, X0, chunk=50000):
        return self._integrate(self._true_velocity, X0, chunk, self.true_solver)

    @torch.no_grad()
    def true_field_backtrack(self, Pd, chunk=50000, inplace=False):
        return self._integrate(self._true_velocity, Pd, chunk, self.true_solver, backward=True,
                               inplace=inplace)
