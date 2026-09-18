"""stages/train.py — the learned sampler: one score network per (d, K), saved to checkpoints/.

Cached and reused unless train.force_retrain or the stored training recipe differs from the
current train/data config (then it is retrained on demand). Each model is retrained with more
steps until its hallucination rate on probe seeds is <= floor + train.hall_excess, the floor
being 1 - data.mass_q (a perfect sampler leaves that much mass outside the R99 balls).
The optimiser recipe itself (EMA, cosine lr, clipping) lives in common.process.Process.fit.
"""
from __future__ import annotations
import os
import json
import time

from common import gmm, checkpoint, utils
from common.process import make_process


def train_one(cfg, d, K, device):
    proc_name = cfg.process
    path = utils.ckpt_path(cfg.paths.checkpoints, proc_name, d, K, int(cfg.sweep.T_train))
    rec = checkpoint.recipe(cfg.train, cfg.data)
    if os.path.exists(path) and not cfg.train.force_retrain:
        if checkpoint.is_current(path, proc_name, rec):
            return {"d": d, "K": K, "path": path, "status": "cached"}
        print(f"[train:{proc_name}] d={d:>2} K={K:>2} checkpoint has a stale schedule or recipe -> retraining")

    sigma = cfg.data.sigma
    means_t, mult = gmm.sample_modes(K, d, cfg.data.radius, sigma, cfg.data.m_mult,
                                     seed=cfg.seed, device=device)
    R99 = gmm.r99(d, sigma, cfg.data.mass_q)
    proc = make_process(proc_name, means_t, sigma ** 2, cfg.sweep.T_train, device, cfg)

    steps = int(cfg.train.base_steps * (1 + d / 16) * (1 + K / 16))
    # a perfect sampler still puts (1 - mass_q) of its mass outside the R99 balls, so the
    # stopping rule is phrased as an excess over that floor when train.hall_excess is set
    excess = cfg.train.get("hall_excess", None)
    target = (1.0 - float(cfg.data.mass_q)) + float(excess) if excess is not None else float(cfg.train.hall_target)
    t0 = time.time()
    model, hall, attempts = None, 1.0, 0
    for attempts in range(1, int(cfg.train.max_attempts) + 1):
        model = proc.train_model(K, d, steps, cfg.train.lr, cfg.train.batch, cfg.seed)
        X0 = proc.seeds(cfg.train.probe_n, d, cfg.seed + 7)
        hall = float((proc.label(model, X0, R99) == -1).float().mean())
        if hall <= target:
            break
        steps = int(steps * cfg.train.step_growth)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint.save(path, model, process=proc_name, d=d, K=K, T=cfg.sweep.T_train,
                    means_t=means_t, R99=R99, sigma=sigma, mult=mult, hall_rate=hall,
                    recipe=rec, extra=proc.extra_ckpt())
    return {"d": d, "K": K, "path": path, "status": "trained", "hall_rate": hall, "hall_target": target,
            "attempts": attempts, "steps": steps, "secs": round(time.time() - t0, 1)}


def run(cfg):
    device = utils.get_device(cfg.device)
    proc_name = cfg.process
    pdir = utils.process_dir(cfg.paths.checkpoints, proc_name)
    os.makedirs(pdir, exist_ok=True)
    manifest = {}
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            try:
                info = train_one(cfg, int(d), int(K), device)
            except RuntimeError as e:
                info = {"d": int(d), "K": int(K), "status": "skipped", "reason": str(e)}
            manifest[f"{d}_{K}"] = info
            print(f"[train:{proc_name}] d={d:>2} K={K:>2} -> {info.get('status')}"
                  + (f" hall={info['hall_rate']:.4f} (target {info['hall_target']:.4f}, "
                     f"{info['attempts']} attempts, {info['steps']} steps) {info['secs']}s"
                     if info.get("status") == "trained" else ""), flush=True)
    mpath = os.path.join(pdir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[train:{proc_name}] manifest written: {mpath}")
    return manifest
