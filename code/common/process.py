"""common/process.py — the contract every process folder implements, plus the registry.

A process owns its time convention, its training objective and its sampler, and exposes:

    build_model(d)                    -> untrained network for this process
    train_model(K, d, n_steps, lr, batch, seed) -> trained network
    sample(model, X0)                 -> data-space endpoints of seeds X0 under the LEARNED model
    true_forward(X0)                  -> endpoints under the ANALYTIC field of the reference
                                         GMM (true score / true OT velocity): the ideal sampler
    true_backward(Xd)                 -> the reverse: carry data-space points back to seed space
                                         with the analytic field (used to build ring atlases)
    extra_ckpt()                      -> process-specific fields to store in the checkpoint

Register with @register("name") in <name>/process.py; load_process("name") imports that
module and returns the class, so stages never hard-code a process list.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import importlib
import torch

from common import utils

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def load_process(name: str) -> type:
    """Import <name>.process (the process folder) if needed and return its class."""
    if name not in _REGISTRY:
        importlib.import_module(f"{name}.process")
    if name not in _REGISTRY:
        raise ValueError(f"process {name!r} did not register itself; known: {list(_REGISTRY)}")
    return _REGISTRY[name]


def make_process(name: str, means_t, variance, T, device, cfg=None):
    return load_process(name)(means_t, variance, T, device, cfg)


class Process(ABC):
    name: str = "base"

    def __init__(self, means_t, variance, T, device, cfg=None):
        self.means_t = means_t          # (K, d)
        self.variance = variance        # within-mode variance sigma^2
        self.T = T                      # number of integration steps
        self.device = device
        self.cfg = cfg
        self.K, self.d = means_t.shape

    def seeds(self, N, d, seed):
        return utils.seeds(N, d, seed, self.device)

    def net_kwargs(self) -> dict:
        n = getattr(getattr(self.cfg, "train", None), "net", None)
        return dict(n) if n is not None else {}

    def fit(self, model, loss_fn, n_steps, lr, batch):
        """Shared optimisation loop for train_model: Adam(W) with cosine decay of the learning rate
        to lr * train.lr_final, gradient clipping at train.grad_clip, and an exponential moving
        average of the weights (train.ema) that replaces the raw weights at the end -- the
        standard recipe for a low-error score network. loss_fn(batch) -> scalar loss.
        """
        import copy
        tr = getattr(self.cfg, "train", None)
        get = lambda k, dflt: float(tr.get(k, dflt)) if tr is not None else dflt
        ema_decay, lr_final, clip, wd = get("ema", 0.0), get("lr_final", 1.0), get("grad_clip", 0.0), get("weight_decay", 0.0)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if wd > 0 else torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=lr * lr_final) if lr_final < 1 else None
        ema = copy.deepcopy(model) if ema_decay > 0 else None
        for step in range(n_steps):
            loss = loss_fn(batch)
            opt.zero_grad(); loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                decay = min(ema_decay, (1 + step) / (10 + step))      # warm-up: early weights are noise
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(decay).add_(pm, alpha=1 - decay)
        if ema is not None:
            model.load_state_dict(ema.state_dict())
        return model

    @abstractmethod
    def build_model(self, d): ...

    @abstractmethod
    def train_model(self, K, d, n_steps, lr, batch, seed): ...

    @abstractmethod
    def sample(self, model, X0, chunk=50000): ...

    @abstractmethod
    def true_forward(self, X0, chunk=50000): ...

    @abstractmethod
    def true_backward(self, Xd, chunk=50000): ...

    def extra_ckpt(self) -> dict:
        return {}

    @torch.no_grad()
    def label(self, model, X0, R99, chunk=50000):
        """Fate labels of seeds X0 under the learned model (or the analytic field if model is None)."""
        from common import gmm
        Xf = self.true_forward(X0, chunk) if model is None else self.sample(model, X0, chunk)
        return gmm.label_fate(Xf, self.means_t, R99)
