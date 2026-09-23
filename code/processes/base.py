"""processes/base.py — what a generative process (DDIM, flow matching) has to provide.

train.py and evaluate.py only talk to these methods, so adding a process means writing one
more subclass:

    train_closure   a fresh model and the function that computes one training step's loss
    sample          where the learned model sends a batch of seeds
    true_field_*    the same journey under the exact field, forwards (seed to data) or
                    backwards (data to seed)
    seeds           standard normal seeds, reproducible from an integer

The exact field (the score for diffusion, the OT velocity for flow matching) has a closed form
for a Gaussian mixture, and that is what makes the atlas possible.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch

import core


class Process(ABC):
    name = "base"

    def __init__(self, means_t, variance, T, device, cfg, weights=None):
        self.means_t, self.variance, self.T = means_t, variance, T    # (K, d), sigma^2, steps
        self.device, self.cfg = device, cfg
        self.K, self.d = means_t.shape
        # mixing weights (None = uniform), shared by the training data and the exact field
        self.weights = None if weights is None else weights.to(device)
        self.logw = core.log_weights(self.weights, device)

    def seeds(self, N, d, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(N, d, generator=g, device=self.device)

    def arch(self, d):
        """ScoreNet kwargs for dimension d (see core.net_arch)."""
        return core.net_arch(d, self.cfg)

    def n_train(self):
        """Size of the fixed training set per (d, K, seed); None = fresh draws every step."""
        return int(self.cfg.train.n_train) if self.cfg.train.n_train else None

    def optim_kwargs(self):
        """The training recipe of cfg.train as keyword arguments of core.run_optimizers."""
        t = self.cfg.train
        return dict(lr_min=t.lr_min, grad_clip=t.grad_clip, ema_decay=t.ema_decay,
                    ema_warmup=int(t.ema_warmup or 0), cuda_graph=bool(t.cuda_graph),
                    branch_streams=bool(t.graph_streams))

    @torch.no_grad()
    def label(self, model, X0, R99, chunk=50000):
        """Fates of seeds X0 under the learned model, or under the exact field if model is None."""
        Xf = self.true_field_forward(X0, chunk) if model is None else self.sample(model, X0, chunk)
        return core.label_fate(Xf, self.means_t, R99)

    @abstractmethod
    def train_closure(self, K, d, batch, seed): ...

    @abstractmethod
    def sample(self, model, X0, chunk=50000): ...

    @abstractmethod
    def true_field_forward(self, X0, chunk=50000): ...

    @abstractmethod
    def true_field_backtrack(self, Pd, chunk=50000, inplace=False): ...
