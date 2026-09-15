"""
processes/base.py — the interface every generative process implements.

train.py and evaluate.py talk only to this contract, so adding a new sampler is a
matter of dropping in another module that subclasses Process. Each process owns its
own time convention internally and exposes a uniform API:

    train_model(...)        -> a trained torch.nn.Module
    sample(model, X0)       -> endpoints (data-space) from seeds X0
    seeds(N, d)             -> initial noise ~ N(0, I)
    true_field_backtrack(P) -> carry data-space points P back to seed space using the
                               ANALYTIC (population) field of the reference GMM

The reference field is the score for diffusion and the OT velocity for flow matching;
both are known in closed form for a Gaussian mixture, which is what the atlas needs.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class Process(ABC):
    name: str = "base"

    def __init__(self, means_t, variance, T, device, cfg=None):
        self.means_t = means_t          # (K, d)
        self.variance = variance        # within-mode variance sigma0^2
        self.T = T                      # number of steps
        self.device = device
        self.cfg = cfg
        self.K, self.d = means_t.shape

    # ---- seeds ----
    def seeds(self, N, d, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(N, d, generator=g, device=self.device)

    # ---- learned model ----
    @abstractmethod
    def train_model(self, K, d, n_steps, lr, batch, seed):
        ...

    @abstractmethod
    def build_model(self, d):
        """Return an untrained network with the right head for this process."""
        ...

    @torch.no_grad()
    @abstractmethod
    def sample(self, model, X0, chunk=50000):
        """Map seeds X0 -> data-space endpoints with the LEARNED model."""
        ...

    # ---- analytic reference field + atlas backtrack ----
    @torch.no_grad()
    @abstractmethod
    def true_field_backtrack(self, Pd, chunk=50000):
        """Carry data-space points Pd back to seed space via the analytic field."""
        ...
