"""processes/factory.py — build the process named in the config."""
from __future__ import annotations

from processes.ddim import DDIMProcess
from processes.flow import FlowOTProcess

_REGISTRY = {
    "ddim": DDIMProcess,
    "flow": FlowOTProcess,
}


def make_process(name, means_t, variance, T, device, cfg=None, weights=None):
    if name not in _REGISTRY:
        raise ValueError(f"unknown sampler '{name}'. choose from {list(_REGISTRY)}")
    return _REGISTRY[name](means_t, variance, T, device, cfg, weights)
