"""processes/factory.py — build the process named in the config."""
from __future__ import annotations

from processes.ddim import DDIMProcess
from processes.dpmpp2m import DPMpp2MProcess
from processes.flow import FlowOTProcess
from processes.heun import HeunProcess
from processes.rk45 import RK45Process

_REGISTRY = {
    "ddim": DDIMProcess,
    "flow": FlowOTProcess,
    "heun": HeunProcess,        # the three below sample the DDIM network (base: processes/pf_ode.py)
    "rk45": RK45Process,
    "dpmpp2m": DPMpp2MProcess,
}


def make_process(name, means_t, variance, T, device, cfg=None, weights=None):
    if name not in _REGISTRY:
        raise ValueError(f"unknown sampler '{name}'. choose from {list(_REGISTRY)}")
    return _REGISTRY[name](means_t, variance, T, device, cfg, weights)


def checkpoint_process(name):
    """The process whose checkpoints `name` uses: itself, or 'ddim' for the samplers that
    integrate the DDIM network with another solver (they are never trained)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown sampler '{name}'. choose from {list(_REGISTRY)}")
    return _REGISTRY[name].ckpt_process or name
