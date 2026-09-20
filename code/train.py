
"""
train.py — Stage 1. Train one learned model per (d, K, seed) for the SELECTED process and save
a checkpoint. Checkpoints live under data/<process>/checkpoints so ddim and flow runs never
collide. The seeds are cfg.n_seeds repeats from cfg.seed (core.seed_list); each repeat draws
its own mode placement. Idempotent unless cfg.train.force_retrain.
"""
from __future__ import annotations
import os
import json
import time
import glob
import fcntl
import subprocess
import torch
from omegaconf import OmegaConf

import core
from processes.factory import make_process


# weighted (non-uniform mixing weights) and unweighted GMMs are separate experiments that share
# everything else, so their checkpoints, caches and results live in sibling folders:
#   data/<sampler>/<variant>/{checkpoints,gt_cache,manifest.json}
#   output/<run_id>/<sampler>/<variant>/T<T_true>/
def variant_of(cfg) -> str:
    """'weighted' or 'unweighted' from cfg.data.weighted."""
    return "weighted" if bool(cfg.data.get("weighted", False)) else "unweighted"


def sampler_dir(base, sampler, variant="unweighted"):
    return os.path.join(base, sampler, variant)


def run_dir(base, sampler, run_id, variant="unweighted"):
    """<base>/<sampler>/<variant>/<run_id> — one directory per invocation."""
    return os.path.join(sampler_dir(base, sampler, variant), run_id)


def latest_run_dir(base, sampler, variant="unweighted"):
    """Newest existing run directory, or None. Timestamps sort lexicographically."""
    pat = os.path.join(sampler_dir(base, sampler, variant), "[0-9]*")
    runs = sorted(p for p in glob.glob(pat) if os.path.isdir(p))
    return runs[-1] if runs else None


def resolve_run_dir(base, sampler, run_id, variant="unweighted"):
    """The run_id directory if it has results, else the newest one that does."""
    d = run_dir(base, sampler, run_id, variant)
    if os.path.exists(os.path.join(d, "results.json")):
        return d
    return latest_run_dir(base, sampler, variant)


def sweep_dir(base, run_id, sampler, T_true, variant="unweighted"):
    """output/<run_id>/<sampler>/<variant>/T<T_true> — one folder per (run, process, variant, T).
    d is a heatmap axis (not a folder), so a whole d x K grid lives in one folder."""
    return os.path.join(base, str(run_id), sampler, variant, f"T{int(T_true)}")


def resolve_sweep_dir(base, run_id, sampler, T_true, variant="unweighted"):
    """The sweep_dir for this run if it has results, else the newest run that does."""
    d = sweep_dir(base, run_id, sampler, T_true, variant)
    if os.path.exists(os.path.join(d, "results.json")):
        return d
    pat = os.path.join(base, "*", sampler, variant, f"T{int(T_true)}", "results.json")
    hits = sorted(glob.glob(pat))
    return os.path.dirname(hits[-1]) if hits else None


def ckpt_path(data_dir, sampler, d, K, seed, variant="unweighted"):
    return os.path.join(sampler_dir(data_dir, sampler, variant), "checkpoints",
                        f"model_d{d}_K{K}_s{seed}.pt")


def gt_cache_path(data_dir, sampler, d, K, seed, variant="unweighted"):
    """Cached learned-sampler ground-truth fate labels, so evaluate does not recompute the
    N-seed forward pass once per T_true. Keyed by (sampler, variant, d, K, seed)."""
    return os.path.join(sampler_dir(data_dir, sampler, variant), "gt_cache",
                        f"d{d}_K{K}_s{seed}.pt")


def n_eval_for(cfg, K):
    """Number of ground-truth eval seeds for a K-mode cell: eval.n_eval when set (fixed), else
    eval.n_eval_per_mode * K so every mode gets the same number of seeds whatever K is."""
    fixed = cfg.eval.get("n_eval", None)
    if fixed:
        return int(fixed)
    return int(cfg.eval.n_eval_per_mode) * int(K)


def save_gt_cache(data_dir, sampler, d, K, gt, n_eval, T_train, R99, seed, variant="unweighted"):
    p = gt_cache_path(data_dir, sampler, d, K, seed, variant)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save({"gt": gt.detach().cpu(), "n_eval": int(n_eval), "seed": int(seed),
                "T_train": int(T_train), "R99": float(R99)}, p)


def load_gt_cache(data_dir, sampler, d, K, n_eval, seed, device, variant="unweighted"):
    """Return cached gt labels (on `device`) when present and matching (n_eval, seed), else None."""
    p = gt_cache_path(data_dir, sampler, d, K, seed, variant)
    if not os.path.exists(p):
        return None
    try:
        blob = torch.load(p, map_location=device)
    except Exception:
        return None
    if int(blob.get("n_eval", -1)) != int(n_eval) or int(blob.get("seed", -999)) != int(seed):
        return None
    return blob["gt"].to(device)
 
 
class ModePlacementError(RuntimeError):
    """Raised when the requested (d, K, R, sigma) geometry cannot be placed."""
 
 
def _cell_geometry(cfg, d, K, seed, device):
    """Mode placement for one repeat: (means_t, min_sep, R99, weights). weights (K,) are the
    mixing weights (uniform unless cfg.data.weighted). Raises ModePlacementError."""
    sigma = cfg.data.sigma
    # the mode-sphere radius scales with d (R(d) = data.radius * sqrt(d/2), so d=2 -> data.radius),
    # keeping the modes resolvable as the dimension grows.
    radius = float(cfg.data.radius) * (d / 2.0) ** 0.5
    try:
        means_t, min_sep = core.sample_modes(
            K, d, radius, sigma, cfg.data.m_mult, seed=seed, device=device
        )
    except RuntimeError as e:
        # sample_modes is the only geometry failure; re-raise as a distinct type so run()
        # does not also swallow CUDA OOM and other torch RuntimeErrors as "skipped".
        raise ModePlacementError(str(e)) from e
    weights = core.mode_weights(K, seed, bool(cfg.data.get("weighted", False)), device=device)
    return means_t, float(min_sep), core.r99(d, sigma, cfg.data.mass_q), weights


def _git_commit():
    """Current commit hash (with '-dirty' if the tree has changes), or None outside a repo."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                    cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL,
                               cwd=os.path.dirname(os.path.abspath(__file__))).returncode != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


def _save_cell(cfg, d, K, seed, proc, model, means_t, min_sep, R99, hall, used_steps, attempts,
               weights=None):
    """Write the checkpoint and the learned-sampler ground-truth cache for one repeat.
    The full resolved Hydra config that produced the model is stored under ckpt["config"]
    (a plain dict) so a checkpoint can always be traced back to exactly how it was trained."""
    sampler = cfg.process.name
    variant = variant_of(cfg)
    sigma = cfg.data.sigma
    T = cfg.process.T_train
    path = ckpt_path(cfg.paths.data, sampler, d, K, seed, variant)
    converged = hall <= cfg.train.hall_target
    arch = {"h": core.ScoreNet.H, "nb": core.ScoreNet.NB, "td": core.ScoreNet.TD}
    ckpt = {
        "state_dict": model.state_dict(),
        "sampler": sampler,
        "d": d, "K": K, "T": T, "seed": seed,
        "means": means_t.cpu(),
        # mixing weights over the modes (uniform 1/K when not weighted) and which experiment
        "weights": (weights if weights is not None else torch.full((K,), 1.0 / K)).cpu(),
        "weighted": variant == "weighted",
        "R99": R99, "sigma": sigma, "variance": sigma ** 2,
        "min_sep": min_sep,
        "hall_rate": hall,
        "converged": converged,
        "steps": used_steps,
        "attempts": attempts,
        "arch": arch,
        # provenance: the exact config this checkpoint was trained with
        "config": OmegaConf.to_container(cfg, resolve=True),
        "run_id": str(cfg.run_id),
        "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if sampler == "flow":
        ckpt["flow"] = {"sigma_min": float(cfg.process.sigma_min),
                        "solver": str(cfg.process.solver)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

    # Cache the learned-sampler ground truth now (once), so every downstream (evaluate, T_true)
    # job loads it instead of re-running the N-seed forward pass. Same seeds evaluate will use.
    try:
        n_eval = n_eval_for(cfg, K)
        X_te = proc.seeds(n_eval, d, seed + 1)
        gt = core.label_fate(proc.sample(model, X_te), means_t, R99)
        save_gt_cache(cfg.paths.data, sampler, d, K, gt, n_eval, T, R99, seed, variant)
    except Exception as e:
        print(f"[train:{sampler}] gt cache skipped d={d} K={K} seed={seed}: {e}")
    return path, converged


def _ckpt_T(path):
    """T_train stored in a checkpoint, or None if it cannot be read."""
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)["T"])
    except Exception:
        return None


def train_cell(cfg, d, K, seeds, device):
    """Train the (d, K) cell for every repeat in `seeds`; returns {seed: info}.

    Each repeat is an independent draw of the whole experiment: its mode placement, model
    init and probe seeds all derive from its seed. The repeats are trained TOGETHER in one
    core.run_optimizers call (one CUDA graph, one branch per seed) since a single small model
    cannot fill a GPU. The retry loop is per seed: after each attempt only the seeds still
    above hall_target are retrained (from scratch, with step_growth x more steps). `secs` in
    the returned info is the wall time of the whole cell (all seeds), not of one seed."""
    sampler = cfg.process.name
    variant = variant_of(cfg)
    T = cfg.process.T_train
    variance = cfg.data.sigma ** 2
    if cfg.train.max_attempts < 1:
        raise ValueError("train.max_attempts must be >= 1")

    out, cells = {}, {}
    for seed in seeds:
        path = ckpt_path(cfg.paths.data, sampler, d, K, seed, variant)
        if os.path.exists(path) and not cfg.train.force_retrain:
            # reuse only if it was trained with the requested T_train; a checkpoint left by a
            # sweep with a different T would otherwise be evaluated (and averaged) silently
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
                       "proc": make_process(sampler, means_t, variance, T, device, cfg, weights)}

    steps = int(cfg.train.base_steps * (1 + d / 16) * (1 + K / 16))
    t0 = time.time()
    pending = list(cells)                       # seeds still to (re)train
    done = {}                                   # seed -> (model, hall, steps, attempts)
    for attempt in range(cfg.train.max_attempts):
        if not pending:
            break
        models, closures = [], []
        for seed in pending:
            m, c = cells[seed]["proc"].train_closure(K, d, cfg.train.batch, seed)
            models.append(m); closures.append(c)
        models = core.run_optimizers(models, closures, steps, cfg.train.lr,
                                     **cells[pending[0]]["proc"].optim_kwargs())
        still = []
        for seed, model in zip(pending, models):
            c = cells[seed]
            X0 = c["proc"].seeds(cfg.train.probe_n, d, seed + 7)
            Xf = c["proc"].sample(model, X0)
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
                     "status": "trained" if converged else "not_converged",
                     "hall_rate": hall, "converged": converged,
                     "steps": used_steps, "attempts": attempts,
                     "secs": round(time.time() - t0, 1)}
    return out


def run(cfg):
    device = core.get_device(cfg.device)
    sampler = cfg.process.name
    variant = variant_of(cfg)
    os.makedirs(os.path.join(sampler_dir(cfg.paths.data, sampler, variant), "checkpoints"),
                exist_ok=True)
 
    manifest = {}
    n_bad = 0
    seeds = core.seed_list(cfg)
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            infos = train_cell(cfg, int(d), int(K), seeds, device)
            for seed in seeds:
                info = infos[seed]
                manifest[f"{d}_{K}_s{seed}"] = info

                status = info.get("status")
                line = f"[train:{sampler}/{variant}] d={d:>2} K={K:>2} seed={seed:<4} -> {status}"
                if status in ("trained", "not_converged"):
                    line += (f" hall={info['hall_rate']:.4f} steps={info['steps']} "
                             f"attempts={info['attempts']} {info['secs']}s")
                if status == "not_converged":
                    line += f"  ** above hall_target={cfg.train.hall_target} **"
                    n_bad += 1
                print(line)

    # one manifest per process, merged under a file lock so the concurrent per-cell jobs
    # scripts/main.sh fans out do not overwrite each other's entries
    mpath = os.path.join(sampler_dir(cfg.paths.data, sampler, variant), "manifest.json")
    with open(mpath, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            merged = json.load(f)
        except Exception:
            merged = {}
        merged.update(manifest)
        f.seek(0); f.truncate()
        json.dump(merged, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"[train:{sampler}] manifest written: {mpath}")
    if n_bad:
        print(f"[train:{sampler}] WARNING: {n_bad} cell(s) did not reach hall_target; "
              f"their results reflect training error, not sampler geometry")
    return manifest
 