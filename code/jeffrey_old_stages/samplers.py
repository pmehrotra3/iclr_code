"""samplers.py — the learned sampler as hall_bench sees it, for ddim and for flow matching.

Every loop in hall_bench runs over levels i = T-1, ..., 1 (T-1 steps, noise -> data). A
sampler exposes the pieces the methods need, so ours, IQ and RODS are written once:

    pred(x, i)             network output at level i (ddim: noise eps; flow: velocity v)
    update(x, i, p)        one step from level i given a prediction p (differentiable)
    step(x, i)             update(x, i, pred(x, i))
    score(x, i)            grad_x log p at level i, from the prediction
    denoise(x, i)          E[data | x]   (ddim: Tweedie x0; flow: x1 = s_t v + (1 - sigma_min) x)
    renoise(xd, i, n)      the point at level i made from data xd and noise n
    dsm_residual(x, i, n, xd)  the training residual there (ddim: eps - n; flow: v - (xd - (1 - sigma_min) n))
    guide(p, g, lam, x, i) the prediction whose score is score - lam * g (IQ's guidance)
    t_of(i)                (i + 1) / T: 1 at the first step, ~0 at the last (IQ's windows, iq_t0)

Flow matching (processes/flow.py, OT path, t = 0 noise -> t = 1 data): x_t = s_t x0 + t x1 with
s_t = 1 - (1 - sigma_min) t, and the network predicts v = E[x1 - (1 - sigma_min) x0 | x_t].
Exactly, for this Gaussian path:
    E[x1 | x_t]        = s_t v + (1 - sigma_min) x_t
    grad log p_t(x_t)  = (t v - x_t) / s_t
so the score IQ and RODS use is computed from the velocity with no approximation, and changing
the score by -lam g changes the velocity by -lam (s_t / t) g (the ddim counterpart:
eps + lam sqrt(1 - abar) g). Flow level i is grid point k = T-1-i of the process's t grid, so
both samplers take the same T-1 steps and IQ's window w means the last w of the run for both.
"""
from __future__ import annotations

import torch

from .pullback_iq import Field, ddim_step


class DDIMSampler:
    """DDIM on the process's schedule (same arithmetic as pullback_iq.Sampler)."""
    kind = "ddim"

    def __init__(self, proc, model, field="learned"):
        self.p, self.F, self.T = proc, Field(proc, model, field), proc.T
        self.ab = proc.abar

    def pred(self, x, i):
        return self.F.eps(x, i)

    def update(self, x, i, e):
        return ddim_step(self.p, x, i, e)

    def score(self, x, i):
        return -self.pred(x, i) / torch.sqrt(1 - self.ab[i])

    def denoise(self, x, i):
        return (x - torch.sqrt(1 - self.ab[i]) * self.pred(x, i)) / torch.sqrt(self.ab[i])

    def renoise(self, xd, i, n):
        return torch.sqrt(self.ab[i]) * xd + torch.sqrt(1 - self.ab[i]) * n

    def dsm_residual(self, x, i, n, xd):
        return self.pred(x, i) - n

    def guide(self, e, g, lam, x, i):
        return e + lam * torch.sqrt(1 - self.ab[i]) * g

    def time(self, i):
        """The sampler's own time at level i (ddim: (i+1)/T, 1 = noise)."""
        return (int(i) + 1) / self.T

    # shared
    def t_of(self, i):
        return (int(i) + 1) / self.T

    def tweedie(self, x, i):
        return self.denoise(x, i)

    def step(self, x, i):
        return self.update(x, i, self.pred(x, i))

    def flow(self, x, i_start):
        for i in range(int(i_start), 0, -1):
            x = self.step(x, i)
        return x

    def G(self, z, chunk=20000):
        with torch.no_grad():
            return torch.cat([self.flow(c, self.T - 1) for c in z.split(chunk)])

    def traj(self, z):
        with torch.no_grad():
            xs = [z]
            for i in range(self.T - 1, 0, -1):
                xs.append(self.step(xs[-1], i))
        return torch.stack(xs)                               # (T, B, d), row k = level T-1-k


class FlowSampler(DDIMSampler):
    """Pranav's flow-matching sampler (FlowOTProcess.sample) with the same interface.

    Level i is grid point k = T-1-i, t = ts[k]; the network is called with time index
    round(t (T-1)) exactly as in FlowOTProcess.sample. Euler steps are x + dt v; other solvers
    (process.solver) work for the plain sampler and ours, but IQ and RODS change the velocity
    of a single evaluation and are defined for euler only."""
    kind = "flow"

    def __init__(self, proc, model, field="learned"):
        self.p, self.m, self.field, self.T = proc, model, field, proc.T
        self.oms = 1.0 - float(proc.sigma_min)
        self.dt = 1.0 / (self.T - 1)
        self.solver = str(proc.solver) if field == "learned" else str(proc.true_solver)

    def _t(self, i):
        return float(self.p.ts[self.T - 1 - int(i)])

    def _v(self, x, t):
        """Velocity at time t (the evaluation FlowOTProcess.sample makes)."""
        if self.field == "true":
            return self.p._true_velocity(x, t)
        idx = max(0, min(self.T - 1, int(round(float(t) * (self.T - 1)))))
        return self.m(x, torch.full((x.shape[0],), idx, dtype=torch.long, device=x.device))

    def _s(self, t):
        return max(1.0 - self.oms * t, 1e-6)

    def pred(self, x, i):
        return self._v(x, self._t(i))

    def update(self, x, i, v):
        if self.solver != "euler":
            raise NotImplementedError(f"IQ / RODS on flow need process.solver=euler (got {self.solver})")
        return x + self.dt * v

    def step(self, x, i):
        if self.solver == "euler":
            return x + self.dt * self.pred(x, i)
        return self.p._step(self._v, x, self._t(i), self.dt, self.solver)

    def score(self, x, i):
        t = self._t(i)
        return (t * self.pred(x, i) - x) / self._s(t)

    def denoise(self, x, i):
        t = self._t(i)
        return self._s(t) * self.pred(x, i) + self.oms * x

    def renoise(self, xd, i, n):
        t = self._t(i)
        return self._s(t) * n + t * xd

    def dsm_residual(self, x, i, n, xd):
        return self.pred(x, i) - (xd - self.oms * n)

    def guide(self, v, g, lam, x, i):
        t = max(self._t(i), 1e-6)
        return v - lam * (self._s(t) / t) * g

    def time(self, i):
        """The sampler's own time at level i (flow: t in [0, 1], 0 = noise)."""
        return self._t(i)


def make_sampler(proc, model, field="learned"):
    if hasattr(proc, "abar"):
        return DDIMSampler(proc, model, field)
    if hasattr(proc, "sigma_min"):
        return FlowSampler(proc, model, field)
    raise RuntimeError(f"{type(proc).__name__}: no sampler for this process (ddim or flow only)")


# ------------------------------------------------------------------ IQ, written once
def iq_energy(S, i0, MC):
    """grad_x E(x, i): E = mean over noise n in MC of |dsm_residual(renoise(denoise(x, i), i0, n))|^2,
    the denoising loss at level i0 of the current data estimate (IQ's LID proxy)."""
    def energy(x, i):
        xd = S.denoise(x, i)
        B, M, d = x.shape[0], MC.shape[0], x.shape[1]
        xn = S.renoise(xd[:, None, :], i0, MC[None])
        r = S.dsm_residual(xn.reshape(-1, d), i0, MC[None].expand(B, M, d).reshape(-1, d),
                           xd[:, None, :].expand(B, M, d).reshape(-1, d)).reshape(B, M, d)
        return (r ** 2).sum(-1).mean(-1)

    def grad(x, i):
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            (g,) = torch.autograd.grad(energy(x, i).sum(), x)
        return g

    return grad


def sample_iq(S, gradE, z, lam, window):
    """IQ: every step with t_of(i) <= window uses the prediction whose score is s - lam grad E.
    Returns every state (T, B, d) and the per-step push |x_IQ - x_plain| (T-1, B)."""
    xs, push = [z], []
    x = z
    for i in range(S.T - 1, 0, -1):
        with torch.no_grad():
            p0 = S.pred(x, i)
            x_plain = S.update(x, i, p0)
        if S.t_of(i) <= window and lam > 0:
            p = S.guide(p0, gradE(x, i), lam, x, i)
            with torch.no_grad():
                xn = S.update(x, i, p)
        else:
            xn = x_plain
        push.append((xn - x_plain).norm(dim=1))
        x = xn.detach()
        xs.append(x)
    return torch.stack(xs), torch.stack(push)
