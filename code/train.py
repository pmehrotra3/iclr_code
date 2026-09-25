"""train.py — Stage 1: one learned sampler per (d, K, seed) for the selected process.

Each repeat seed draws its own mode placement, model init and training data. The seeds of a
cell are trained together (one CUDA graph; each seed has its own RNG, so grouping does not
change the models). A model whose probe hallucination rate exceeds train.hall_target is
retrained with train.step_growth x more steps, up to train.max_attempts. Existing
checkpoints are reused unless train.force_retrain.

Files (<variant> = weighted | unweighted, see data.weighted):
    checkpoints/<process>/<variant>/checkpoints/model_d<d>_K<K>_s<seed>.pt
    checkpoints/<process>/<variant>/gt_cache/d<d>_K<K>_s<seed>.pt
    checkpoints/<process>/<variant>/manifest.json
"""
from __future__ import annotations
import fcntl
import json
import os
import subprocess
import time

import torch
from omegaconf import OmegaConf

import core
from processes.factory import checkpoint_process, make_process


# ---- paths and caches (shared with evaluate / visualize / combine) -----------------------
def variant_of(cfg) -> str:
    return "weighted" if bool(cfg.data.weighted) else "unweighted"


def sampler_dir(base, sampler, variant="unweighted"):
    return os.path.join(base, sampler, variant)


def sweep_dir(base, run_id, sampler, T_true, variant="unweighted"):
    """output/<run_id>/<process>/<variant>/T<T_true>: d and K are axes of its tables."""
    return os.path.join(base, str(run_id), sampler, variant, f"T{int(T_true)}")


def resolve_sweep_dir(base, run_id, sampler, T_true, variant="unweighted"):
    """This run's sweep_dir if it has results, else None (never another run's)."""
    d = sweep_dir(base, run_id, sampler, T_true, variant)
    return d if os.path.exists(os.path.join(d, "results.json")) else None


def ckpt_path(data_dir, sampler, d, K, seed, variant="unweighted"):
    return os.path.join(sampler_dir(data_dir, sampler, variant), "checkpoints", f"model_d{d}_K{K}_s{seed}.pt")


def gt_cache_path(data_dir, sampler, d, K, seed, variant="unweighted"):
    return os.path.join(sampler_dir(data_dir, sampler, variant), "gt_cache", f"d{d}_K{K}_s{seed}.pt")


def n_eval_for(cfg, K):
    """Ground-truth seeds of a K-mode cell: the same number per mode whatever K is."""
    return int(cfg.eval.n_eval_per_mode) * int(K)


def _save_atomic(obj, path):
    """torch.save via a temporary file: an interrupted save never leaves half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _device_name(X):
    return torch.cuda.get_device_name(X.device) if X.is_cuda else "cpu"


def save_gt_cache(data_dir, sampler, d, K, X, gt, n_eval, T_train, R99, seed, variant="unweighted"):
    """The eval seeds X themselves with their fates under the learned sampler: evaluation reads
    both back and never regenerates the points (a regenerated set could differ on another machine)."""
    _save_atomic({"X": X.detach().cpu(), "gt": gt.detach().cpu(), "n_eval": int(n_eval), "seed": int(seed),
                  "T_train": int(T_train), "R99": float(R99), "device": _device_name(X)},
                 gt_cache_path(data_dir, sampler, d, K, seed, variant))


def load_gt_cache(data_dir, sampler, d, K, n_eval, seed, device, variant="unweighted"):
    """(eval seeds, their fates) on `device` if cached for (n_eval, seed), else None. Caches
    without the points (written before they were stored) are misses."""
    p = gt_cache_path(data_dir, sampler, d, K, seed, variant)
    if not os.path.exists(p):
        return None
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if int(blob.get("n_eval", -1)) != int(n_eval) or int(blob.get("seed", -999)) != int(seed):
        return None
    if blob.get("X") is None or blob["X"].shape != (int(n_eval), int(d)):
        return None
    return blob["X"].to(device), blob["gt"].to(device)


def _git_commit():
    """HEAD hash (+ '-dirty' with uncommitted changes), or None outside a repository."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                    cwd=here).decode().strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL,
                               cwd=here).returncode != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


# ---- one cell --------------------------------------------------------------------------------
class ModePlacementError(RuntimeError):
    """The (d, K, R, sigma) geometry cannot be placed: the cell is skipped."""


def _cell_geometry(cfg, d, K, seed, device):
    """(means, min_sep, R99, mixing weights) of one repeat. The sphere radius grows like
    sqrt(d/2) (d=2 -> data.radius) so the modes stay resolvable as d grows."""
    sigma = cfg.data.sigma
    try:
        means_t, min_sep = core.sample_modes(K, d, float(cfg.data.radius) * (d / 2.0) ** 0.5, sigma,
                                             cfg.data.m_mult, seed=seed, device=device)
    except RuntimeError as e:
        raise ModePlacementError(str(e)) from e
    weights = core.mode_weights(K, seed, bool(cfg.data.weighted), device=device, base_seed=int(cfg.seed))
    return means_t, float(min_sep), core.r99(d, sigma, cfg.data.mass_q), weights


def _save_cell(cfg, d, K, seed, proc, model, means_t, min_sep, R99, hall, used_steps, attempts,
               weights):
    """Write the checkpoint (with its full config and git commit, for provenance) and cache the
    learned sampler's ground truth for the eval seeds (the same ones evaluate.py uses)."""
    sampler, variant, sigma, T = cfg.process.name, variant_of(cfg), cfg.data.sigma, cfg.process.T_train
    path = ckpt_path(cfg.paths.checkpoints, sampler, d, K, seed, variant)
    converged = hall <= cfg.train.hall_target
    ckpt = {"state_dict": model.state_dict(), "sampler": sampler, "d": d, "K": K, "T": T, "seed": seed,
            "means": means_t.cpu(), "weights": weights.cpu(), "weighted": variant == "weighted",
            "R99": R99, "sigma": sigma, "variance": sigma ** 2, "min_sep": min_sep,
            "hall_rate": hall, "converged": converged, "steps": used_steps, "attempts": attempts,
            "arch": proc.arch(d), "config": OmegaConf.to_container(cfg, resolve=True),
            "run_id": str(cfg.run_id), "git_commit": _git_commit(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if sampler == "flow":
        ckpt["flow"] = {"sigma_min": float(cfg.process.sigma_min), "solver": str(cfg.process.solver)}
    _save_atomic(ckpt, path)
    try:
        n_eval = n_eval_for(cfg, K)
        X = proc.seeds(n_eval, d, seed + 1)
        gt = core.label_fate(proc.sample(model, X), means_t, R99)
        save_gt_cache(cfg.paths.checkpoints, sampler, d, K, X, gt, n_eval, T, R99, seed, variant)
    except Exception as e:                        # evaluate.py recomputes it on a cache miss
        print(f"[train:{sampler}] gt cache skipped d={d} K={K} seed={seed}: {e}")
    return path, converged


def _ckpt_T(path):
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)["T"])
    except Exception:
        return None


def train_cell(cfg, d, K, seeds, device):
    """Train every missing repeat of the (d, K) cell; returns {seed: summary}."""
    sampler, T = cfg.process.name, cfg.process.T_train
    out, cells = {}, {}
    for seed in seeds:
        path = ckpt_path(cfg.paths.checkpoints, sampler, d, K, seed, variant_of(cfg))
        if os.path.exists(path) and not cfg.train.force_retrain:
            ck_T = _ckpt_T(path)
            if ck_T == T:
                out[seed] = {"d": d, "K": K, "seed": seed, "path": path, "status": "cached"}
                continue
            print(f"[train:{sampler}] d={d} K={K} seed={seed}: checkpoint has T={ck_T}, "
                  f"want T_train={T} -> retraining")
        try:
            means_t, min_sep, R99, weights = _cell_geometry(cfg, d, K, seed, device)
        except ModePlacementError as e:
            out[seed] = {"d": d, "K": K, "seed": seed, "status": "skipped", "reason": str(e)}
            continue
        cells[seed] = {"means": means_t, "min_sep": min_sep, "R99": R99, "weights": weights,
                       "proc": make_process(sampler, means_t, cfg.data.sigma ** 2, T, device, cfg, weights)}

    steps = int(cfg.train.base_steps * (1 + d / 16) * (1 + K / 16))
    t0 = time.time()
    pending, done = list(cells), {}               # done: seed -> (model, hall, steps, attempts)
    for attempt in range(int(cfg.train.max_attempts)):
        if not pending:
            break
        models, closures = zip(*(cells[s]["proc"].train_closure(K, d, cfg.train.batch, s) for s in pending))
        models = core.run_optimizers(list(models), list(closures), steps, cfg.train.lr,
                                     **cells[pending[0]]["proc"].optim_kwargs())
        still = []
        for seed, model in zip(pending, models):
            c = cells[seed]
            Xf = c["proc"].sample(model, c["proc"].seeds(cfg.train.probe_n, d, seed + 7))
            hall = float((core.label_fate(Xf, c["means"], c["R99"]) == -1).float().mean())
            done[seed] = (model, hall, steps, attempt + 1)
            if hall > cfg.train.hall_target:
                still.append(seed)
        pending = still
        steps = int(steps * cfg.train.step_growth)

    for seed, (model, hall, used_steps, attempts) in done.items():
        c = cells[seed]
        path, converged = _save_cell(cfg, d, K, seed, c["proc"], model, c["means"], c["min_sep"],
                                     c["R99"], hall, used_steps, attempts, c["weights"])
        out[seed] = {"d": d, "K": K, "seed": seed, "path": path,
                     "status": "trained" if converged else "not_converged", "hall_rate": hall,
                     "converged": converged, "steps": used_steps, "attempts": attempts,
                     "net": core.net_size(d, cfg), "width": c["proc"].arch(d)["h"],
                     "secs": round(time.time() - t0, 1)}          # wall time of the whole cell
    return out


def run(cfg):
    sampler, variant = cfg.process.name, variant_of(cfg)
    if checkpoint_process(sampler) != sampler:
        print(f"[train:{sampler}] samples the {checkpoint_process(sampler)} checkpoints with another "
              f"solver: nothing to train (train process={checkpoint_process(sampler)})")
        return
    device = core.get_device(cfg.device)
    manifest, n_bad = {}, 0
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            seeds = core.seed_list(cfg)
            infos = train_cell(cfg, int(d), int(K), seeds, device)
            for seed in seeds:
                info = infos[seed]
                manifest[f"{d}_{K}_s{seed}"] = info
                line = f"[train:{sampler}/{variant}] d={d:>2} K={K:>2} seed={seed:<4} -> {info['status']}"
                if "hall_rate" in info:
                    line += (f" hall={info['hall_rate']:.4f} steps={info['steps']} attempts={info['attempts']}"
                             f" net={info['net']}/{info['width']} {info['secs']}s")
                if info["status"] == "not_converged":
                    line += f"  ** above hall_target={cfg.train.hall_target} **"
                    n_bad += 1
                print(line)

    # one manifest per (process, variant), merged under a lock (parallel jobs of scripts/main.sh)
    mpath = os.path.join(sampler_dir(cfg.paths.checkpoints, sampler, variant), "manifest.json")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            merged = json.load(f)
        except ValueError:
            merged = {}
        merged.update(manifest)
        f.seek(0); f.truncate()
        json.dump(merged, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"[train:{sampler}] manifest written: {mpath}")
    if n_bad:
        print(f"[train:{sampler}] WARNING: {n_bad} model(s) above hall_target: their results "
              f"reflect training error, not sampler geometry")
