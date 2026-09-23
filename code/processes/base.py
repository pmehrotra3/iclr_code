"""
processes/base.py — the interface every generative process implements.

train.py and evaluate.py talk only to this contract, so adding a new sampler is a
matter of dropping in another module that subclasses Process. Each process owns its
own time convention internally and exposes a uniform API:

    train_closure(...)      -> (fresh torch.nn.Module, per-step loss closure); train.py groups
                               the closures of a cell's seeds into one core.run_optimizers call
    train_model(...)        -> a trained torch.nn.Module (train_closure + core.run_optimizer)
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

import core


class Process(ABC):
    name: str = "base"

    def __init__(self, means_t, variance, T, device, cfg=None, weights=None):
        self.means_t = means_t          # (K, d)
        self.variance = variance        # within-mode variance sigma0^2
        self.T = T                      # number of steps
        self.device = device
        self.cfg = cfg
        self.K, self.d = means_t.shape
        # mixing weights (K,) over the modes; None = uniform. Used by the training data AND
        # the exact reference process so both describe the same (weighted) GMM.
        self.weights = None if weights is None else weights.to(device)
        self.logw = core.log_weights(self.weights, device)

    # ---- seeds ----
    def seeds(self, N, d, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(N, d, generator=g, device=self.device)

    # ---- learned model ----
    def arch(self, d):
        """ScoreNet kwargs for this d (width scales with d via cfg.train, see core.net_arch)."""
        return core.net_arch(d, self.cfg)

    @abstractmethod
    def train_closure(self, K, d, batch, seed):
        """Return (untrained model, loss_closure) for one training run seeded by `seed`."""
        ...

    @abstractmethod
    def train_model(self, K, d, n_steps, lr, batch, seed):
        ...

    def n_train(self):
        """cfg.train.n_train: size of the fixed training set drawn per (d, K, seed); None/0 =
        fresh samples every step (infinite data)."""
        t = getattr(self.cfg, "train", None) if self.cfg is not None else None
        v = getattr(t, "n_train", None) if t is not None else None
        return int(v) if v else None

    def optim_kwargs(self):
        """The shared optimiser recipe from cfg.train (cosine lr, grad clip, EMA, CUDA graph),
        as keyword arguments for core.run_optimizer / core.run_optimizers."""
        t = getattr(self.cfg, "train", None) if self.cfg is not None else None

        def get(key, default):
            v = getattr(t, key, default) if t is not None else default
            return default if v is None else v

        return dict(lr_min=get("lr_min", None), grad_clip=get("grad_clip", None),
                    ema_decay=get("ema_decay", None), ema_warmup=int(get("ema_warmup", 0) or 0),
                    cuda_graph=bool(get("cuda_graph", True)),
                    branch_streams=bool(get("graph_streams", True)))

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
    def true_field_backtrack(self, Pd, chunk=50000, inplace=False):
        """Carry data-space points Pd back to seed space via the analytic field. inplace=True
        overwrites Pd (the caller keeps only the seed-space result) instead of cloning it."""
        ...

    @torch.no_grad()
    @abstractmethod
    def true_field_forward(self, X0, chunk=50000):
        """Carry seeds X0 forward to data space via the analytic field (the inverse of
        true_field_backtrack). No learned model involved."""
        ...

    # ---- fate labels ----
    @torch.no_grad()
    def label(self, model, X0, R99, chunk=50000):
        """Fate labels of seeds X0: under the LEARNED model, or the exact analytic field when
        model is None (the exact-score reference used to calibrate predictors)."""
        import core
        Xf = self.true_field_forward(X0, chunk) if model is None else self.sample(model, X0, chunk)
        return core.label_fate(Xf, self.means_t, R99)
