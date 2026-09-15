"""common/checkpoint.py — save / load a trained process model with its reference GMM.

The format is shared by every process (and backward compatible with checkpoints written
before the refactor, which used the key "sampler" instead of "process").
"""
from __future__ import annotations
import torch

from common import nets

SCHEDULE_TAG = "vp-continuous"   # bump when the DDIM schedule changes -> stale ckpts retrain


def save(path, model, *, process, d, K, T, means_t, R99, sigma, mult, hall_rate, extra=None):
    torch.save({
        "state_dict": model.state_dict(),
        "process": process, "sampler": process, "schedule": SCHEDULE_TAG,
        "d": d, "K": K, "T": T,
        "means": means_t.cpu(),
        "R99": R99, "sigma": sigma, "variance": sigma ** 2,
        "hall_rate": hall_rate, "mult": mult,
        "arch": dict(model.arch),
        **(extra or {}),
    }, path)


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["arch"]
    m = nets.ScoreNet(ck["d"], a["h"], a["nb"], a["td"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    ck["means"] = ck["means"].to(device)
    ck.setdefault("process", ck.get("sampler"))
    return m, ck


def is_current(path, process):
    """Only DDIM depends on the noise schedule; other processes never go stale."""
    if process != "ddim":
        return True
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return ck.get("schedule") == SCHEDULE_TAG
