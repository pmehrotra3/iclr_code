"""common/checkpoint.py — save / load a trained process model with its reference GMM.

The format is shared by every process (and backward compatible with checkpoints written
before the refactor, which used the key "sampler" instead of "process").
"""
from __future__ import annotations
import torch

from common import nets

SCHEDULE_TAG = "vp-continuous"   # bump when the DDIM schedule changes -> stale ckpts retrain
RECIPE_KEYS = ("base_steps", "lr", "batch", "hall_target", "hall_excess", "step_growth", "max_attempts",
               "ema", "lr_final", "grad_clip", "weight_decay")


def recipe(train_cfg, data_cfg) -> dict:
    """The training recipe a checkpoint was made with; a checkpoint whose recipe differs from the
    current config is stale and is retrained on demand."""
    r = {k: (float(train_cfg[k]) if train_cfg.get(k) is not None else None) for k in RECIPE_KEYS}
    r["net"] = {k: int(v) for k, v in dict(train_cfg.net).items()}
    r["data"] = {k: float(v) for k, v in dict(data_cfg).items()}
    return r


def save(path, model, *, process, d, K, T, means_t, R99, sigma, mult, hall_rate, recipe=None, extra=None):
    torch.save({
        "state_dict": model.state_dict(),
        "process": process, "sampler": process, "schedule": SCHEDULE_TAG, "recipe": recipe,
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


def is_current(path, process, recipe_now=None):
    """A checkpoint is current when its DDIM schedule tag (ddim only) and its training recipe
    match the present config."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if process == "ddim" and ck.get("schedule") != SCHEDULE_TAG:
        return False
    return recipe_now is None or ck.get("recipe") == recipe_now
