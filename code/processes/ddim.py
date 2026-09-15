"""
processes/ddim.py — variance-preserving diffusion with a deterministic DDIM sampler.

Time convention (this process, internal): index i = T-1 is the source (noise), i = 0 is
data. eps-prediction network. The analytic reference field is the GMM score; the atlas
backtrack integrates the true-score reverse ODE from data to noise.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

import core
from processes.base import Process


class DDIMProcess(Process):
    name = "ddim"

    def __init__(self, means_t, variance, T, device, cfg=None):
        super().__init__(means_t, variance, T, device, cfg)
        self.abar = core.make_schedule(T, device=device)

    def build_model(self, d):
        return core.ScoreNet(d).to(self.device)

    def train_model(self, K, d, n_steps, lr, batch, seed):
        return core.train_learned(
            self.means_t, d, K, self.abar, self.T, self.variance,
            n_steps=n_steps, lr=lr, batch=batch, seed=seed, device=self.device,
        )

    @torch.no_grad()
    def sample(self, model, X0, chunk=50000):
        model.eval()
        X = X0.clone()
        for i in reversed(range(1, self.T)):
            ab, abp = self.abar[i], self.abar[i - 1]
            for s in range(0, X.shape[0], chunk):
                xs = X[s:s + chunk]
                ti = torch.full((xs.shape[0],), i, dtype=torch.long, device=self.device)
                eps = model(xs, ti)
                x0 = (xs - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
                X[s:s + chunk] = torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * eps
        return X

    @torch.no_grad()
    def true_field_backtrack(self, Pd, chunk=50000):
        # reuse the shared true-score backtrack (data -> noise)
        return core.backtrack_true(
            Pd, self.means_t, self.abar, self.T, self.variance, chunk=chunk
        )
