"""processes/ddim.py — variance-preserving diffusion with the deterministic DDIM sampler.

Index i = T-1 is noise, i = 0 is data. The network predicts the added noise (eps); the exact
field is the closed-form GMM score (core.true_score), integrated with Heun steps by default.
"""
from __future__ import annotations

import torch

import core
from processes.base import Process


class DDIMProcess(Process):
    name = "ddim"

    def __init__(self, means_t, variance, T, device, cfg, weights=None):
        super().__init__(means_t, variance, T, device, cfg, weights)
        p = cfg.process
        self.abar = core.make_schedule(T, float(p.beta_min), float(p.beta_max), device=device)
        self.true_order = str(p.true_order)                     # exact-field integrator: heun | euler

    def train_closure(self, K, d, batch, seed):
        return core.learned_closure(self.means_t, d, K, self.abar, self.T, self.variance,
                                    batch, seed, self.device, n_train=self.n_train(),
                                    weights=self.weights, arch=self.arch(d))

    @torch.no_grad()
    def sample(self, model, X0, chunk=50000):
        model.eval()
        X = X0.clone()
        for i in reversed(range(1, self.T)):
            for s in range(0, X.shape[0], chunk):
                xs = X[s:s + chunk]
                eps = model(xs, torch.full((xs.shape[0],), i, dtype=torch.long, device=self.device))
                X[s:s + chunk] = core._ddim_step(xs, eps, self.abar[i], self.abar[i - 1])
        return X

    @torch.no_grad()
    def true_field_forward(self, X0, chunk=50000):
        return core.forward_true(X0, self.means_t, self.abar, self.T, self.variance, chunk=chunk,
                                 order=self.true_order, weights=self.weights)

    @torch.no_grad()
    def true_field_backtrack(self, Pd, chunk=50000, inplace=False):
        return core.backtrack_true(Pd, self.means_t, self.abar, self.T, self.variance, chunk=chunk,
                                   order=self.true_order, weights=self.weights, inplace=inplace)
