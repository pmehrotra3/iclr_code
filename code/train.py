
"""
train.py — Stage 1. Train one learned model per (d, K) for the SELECTED process and save
a checkpoint. Checkpoints live under data/<process>/checkpoints so ddim and flow runs never
collide. Idempotent unless cfg.train.force_retrain.
"""
from __future__ import annotations
import os
import json
import time
import glob
import torch

import core
from processes.factory import make_process


def sampler_dir(base, sampler):
    return os.path.join(base, sampler)


def run_dir(base, sampler, run_id):
    """<base>/<sampler>/<run_id> — one directory per invocation."""
    return os.path.join(sampler_dir(base, sampler), run_id)


def latest_run_dir(base, sampler):
    """Newest existing run directory, or None. Timestamps sort lexicographically."""
    pat = os.path.join(sampler_dir(base, sampler), "[0-9]*")
    runs = sorted(p for p in glob.glob(pat) if os.path.isdir(p))
    return runs[-1] if runs else None


def resolve_run_dir(base, sampler, run_id):
    """The run_id directory if it has results, else the newest one that does."""
    d = run_dir(base, sampler, run_id)
    if os.path.exists(os.path.join(d, "results.json")):
        return d
    return latest_run_dir(base, sampler)
 
 
def ckpt_path(data_dir, sampler, d, K):
    return os.path.join(sampler_dir(data_dir, sampler), "checkpoints",
                        f"model_d{d}_K{K}.pt")
 
 
class ModePlacementError(RuntimeError):
    """Raised when the requested (d, K, R, sigma) geometry cannot be placed."""
 
 
def train_one(cfg, d, K, device):
    sampler = cfg.process.name
    path = ckpt_path(cfg.paths.data, sampler, d, K)
    if os.path.exists(path) and not cfg.train.force_retrain:
        return {"d": d, "K": K, "path": path, "status": "cached"}
 
    sigma = cfg.data.sigma
    variance = sigma ** 2
    try:
        means_t, min_sep = core.sample_modes(
            K, d, cfg.data.radius, sigma, cfg.data.m_mult, seed=cfg.seed, device=device
        )
    except RuntimeError as e:
        # sample_modes is the only geometry failure; re-raise as a distinct type so run()
        # does not also swallow CUDA OOM and other torch RuntimeErrors as "skipped".
        raise ModePlacementError(str(e)) from e
 
    T = cfg.process.T_train
    R99 = core.r99(d, sigma, cfg.data.mass_q)
 
    proc = make_process(sampler, means_t, variance, T, device, cfg)
 
    if cfg.train.max_attempts < 1:
        raise ValueError("train.max_attempts must be >= 1")
 
    steps = int(cfg.train.base_steps * (1 + d / 16) * (1 + K / 16))
    t0 = time.time()
    model, hall, used_steps = None, 1.0, steps
    for attempt in range(cfg.train.max_attempts):
        used_steps = steps
        model = proc.train_model(K, d, steps, cfg.train.lr, cfg.train.batch, cfg.seed)
        X0 = proc.seeds(cfg.train.probe_n, d, cfg.seed + 7)
        Xf = proc.sample(model, X0)
        hall = float((core.label_fate(Xf, means_t, R99) == -1).float().mean())
        if hall <= cfg.train.hall_target:
            break
        steps = int(steps * cfg.train.step_growth)
 
    converged = hall <= cfg.train.hall_target
 
    arch = {"h": core.ScoreNet.H, "nb": core.ScoreNet.NB, "td": core.ScoreNet.TD}
 
    ckpt = {
        "state_dict": model.state_dict(),
        "sampler": sampler,
        "d": d, "K": K, "T": T,
        "means": means_t.cpu(),
        "R99": R99, "sigma": sigma, "variance": variance,
        "min_sep": min_sep,
        "hall_rate": hall,
        "converged": converged,
        "steps": used_steps,
        "attempts": attempt + 1,
        "arch": arch,
    }
    if sampler == "flow":
        ckpt["flow"] = {"sigma_min": float(cfg.process.sigma_min),
                        "solver": str(cfg.process.solver)}
 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
 
    return {"d": d, "K": K, "path": path,
            "status": "trained" if converged else "not_converged",
            "hall_rate": hall, "converged": converged,
            "steps": used_steps, "attempts": attempt + 1,
            "secs": round(time.time() - t0, 1)}
 
 
def run(cfg):
    device = core.get_device(cfg.device)
    sampler = cfg.process.name
    os.makedirs(os.path.join(sampler_dir(cfg.paths.data, sampler), "checkpoints"),
                exist_ok=True)
 
    manifest = {}
    n_bad = 0
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            try:
                info = train_one(cfg, int(d), int(K), device)
            except ModePlacementError as e:
                info = {"d": int(d), "K": int(K), "status": "skipped", "reason": str(e)}
            manifest[f"{d}_{K}"] = info
 
            status = info.get("status")
            line = f"[train:{sampler}] d={d:>2} K={K:>2} -> {status}"
            if status in ("trained", "not_converged"):
                line += f" hall={info['hall_rate']:.4f} steps={info['steps']} {info['secs']}s"
            if status == "not_converged":
                line += f"  ** above hall_target={cfg.train.hall_target} **"
                n_bad += 1
            print(line)
 
    mpath = os.path.join(sampler_dir(cfg.paths.data, sampler), "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[train:{sampler}] manifest written: {mpath}")
    if n_bad:
        print(f"[train:{sampler}] WARNING: {n_bad} cell(s) did not reach hall_target; "
              f"their results reflect training error, not sampler geometry")
    return manifest
 