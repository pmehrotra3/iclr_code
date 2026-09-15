"""
train.py — Stage 1. Train one learned model per (d, K) for the SELECTED sampler and save
a checkpoint. Checkpoints live under data/<sampler>/checkpoints so ddim and flow runs never
collide. Idempotent unless cfg.train.force_retrain.
"""
from __future__ import annotations
import os
import json
import time
import torch

import core
from processes.factory import make_process


def sampler_dir(base, sampler):
    return os.path.join(base, sampler)


def ckpt_path(data_dir, sampler, d, K):
    return os.path.join(sampler_dir(data_dir, sampler), "checkpoints",
                        f"model_d{d}_K{K}.pt")


def train_one(cfg, d, K, device):
    sampler = cfg.sampler
    path = ckpt_path(cfg.paths.data, sampler, d, K)
    if os.path.exists(path) and not cfg.train.force_retrain:
        return {"d": d, "K": K, "path": path, "status": "cached"}

    sigma = cfg.data.sigma
    variance = sigma ** 2
    means_t, mult = core.sample_modes(
        K, d, cfg.data.radius, sigma, cfg.data.m_mult, seed=cfg.seed, device=device
    )
    T = cfg.sweep.T_train
    R99 = core.r99(d, sigma, cfg.data.mass_q)

    proc = make_process(sampler, means_t, variance, T, device, cfg)

    base = int(cfg.train.base_steps * (1 + d / 16) * (1 + K / 16))
    steps = base
    t0 = time.time()
    model, hall = None, 1.0
    for _ in range(cfg.train.max_attempts):
        model = proc.train_model(K, d, steps, cfg.train.lr, cfg.train.batch, cfg.seed)
        X0 = proc.seeds(cfg.train.probe_n, d, cfg.seed + 7)
        Xf = proc.sample(model, X0)
        hall = float((core.label_fate(Xf, means_t, R99) == -1).float().mean())
        if hall <= cfg.train.hall_target:
            break
        steps = int(steps * cfg.train.step_growth)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "sampler": sampler,
        "d": d, "K": K, "T": T,
        "means": means_t.cpu(),
        "R99": R99, "sigma": sigma, "variance": variance,
        "hall_rate": hall, "mult": mult,
        "arch": {"h": 256, "nb": 4, "td": 128},
        "flow": {"sigma_min": float(getattr(cfg.flow, "sigma_min", 1e-4)),
                 "solver": str(getattr(cfg.flow, "solver", "euler"))},
    }, path)
    return {"d": d, "K": K, "path": path, "status": "trained",
            "hall_rate": hall, "steps": steps, "secs": round(time.time() - t0, 1)}


def run(cfg):
    device = core.get_device(cfg.device)
    sampler = cfg.sampler
    os.makedirs(os.path.join(sampler_dir(cfg.paths.data, sampler), "checkpoints"),
                exist_ok=True)
    manifest = {}
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            try:
                info = train_one(cfg, int(d), int(K), device)
            except RuntimeError as e:
                info = {"d": int(d), "K": int(K), "status": "skipped", "reason": str(e)}
            manifest[f"{d}_{K}"] = info
            print(f"[train:{sampler}] d={d:>2} K={K:>2} -> {info.get('status')}"
                  + (f" hall={info['hall_rate']:.4f} {info['secs']}s"
                     if info.get("status") == "trained" else ""))
    mpath = os.path.join(sampler_dir(cfg.paths.data, sampler), "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[train:{sampler}] manifest written: {mpath}")
    return manifest
