"""evaluate.py — Stage 2: build the seed-fate atlas and score it (same for ddim and flow).

For every (d, K, seed) cell and every anchor budget b we
  1. plant labelled anchors around the modes in data space: b points per mode inside its R99
     ball plus a thin band just outside it, which counts as hallucination (core.ball_anchors);
     altered_knn gets its own weighted rings instead,
  2. carry them back to seed space with the exact field (proc.true_field_backtrack),
  3. fit each predictor in classifier.models on those (seed, fate) pairs (fate.py),
  4. take the ground truth: the fates of eval.n_eval_per_mode * K fresh seeds under the learned
     sampler (train.py caches these), and
  5. score each predictor against that ground truth (fate.fate_metrics).

Results go to output/<run_id>/<process>/<variant>/T<T_true>/. Each cell's rows are kept in
cells/d<d>_K<K>_s<seed>.json, and a row that is already there is not recomputed (unless
eval.force=true), so adding a budget or a predictor only computes the new rows. Every row uses
fixed seeds, so a cell built up in pieces is identical to one computed in one go.
results.json (per-seed rows plus the mean / std over seeds) and the two csv files are then
rebuilt from all cell files.
"""
from __future__ import annotations
import fcntl
import glob
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

import core
import fate
from processes.factory import make_process
from train import ckpt_path, sweep_dir, load_gt_cache, save_gt_cache, n_eval_for, variant_of


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["arch"]
    model = core.ScoreNet(ck["d"], a["h"], a["nb"], a["td"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ck["means"] = ck["means"].to(device)
    return model, ck


def budgets(cfg):
    b = cfg.sweep.anchors
    return [int(x) for x in ([b] if isinstance(b, (int, float, str)) else b)]


def _specs(cfg):
    """Each predictor of classifier.models merged over classifier._shared, as plain dicts."""
    shared = OmegaConf.to_container(cfg.classifier["_shared"], resolve=True)
    return [{**shared, **OmegaConf.to_container(m, resolve=True)} for m in cfg.classifier.models.values()]


def _primary(cfg):
    """The predictor summarised on the console: classifier.primary, else the first one."""
    return str(cfg.classifier.get("primary") or next(iter(cfg.classifier.models)))


def preflight(cfg, d, K, device):
    """Stop early (not with a CUDA OOM hours in) when the largest budget cannot fit the GPU.
    Peak ~ ball+shell anchors + altered_knn rings + eval seeds (fp32) + ~3 GB working memory."""
    if device.type != "cuda":
        return
    b, f = max(budgets(cfg)), 4 * d
    n_alt = b * K if any(s["arch"] == "altered_knn" for s in _specs(cfg)) else 0
    need = (b * K * (1 + float(cfg.anchors.shell_frac)) * f + n_alt * f + n_eval_for(cfg, K) * f + 3e9) / 1e9
    free, total = (x / 1e9 for x in torch.cuda.mem_get_info(device))
    tag = f"[eval:{cfg.process.name}] d={d} K={K} budgets={budgets(cfg)}"
    print(f"{tag}: est. peak {need:.1f} GB, GPU free {free:.1f}/{total:.1f} GB")
    if need > 0.92 * free:
        raise MemoryError(f"{tag}: estimated peak {need:.1f} GB exceeds free GPU memory {free:.1f} GB; "
                          f"lower sweep.anchors, K or eval.n_eval_per_mode (or one eval job per GPU)")


# ---- cell files --------------------------------------------------------------------------
def cell_path(out_dir, d, K, seed):
    return os.path.join(out_dir, "cells", f"d{int(d)}_K{int(K)}_s{int(seed)}.json")


def _load_cell(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["rows"]


def _save_cell(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"rows": rows}, f)
    os.replace(tmp, path)                                   # atomic: never half a file


def _key(r):
    return int(r["n_per_mode"]), r["model"]


def row_order(cfg):
    """Sort key: (d, K, seed, budget, predictor in config order, others after by name)."""
    names = [s["name"] for s in _specs(cfg)]
    rank = lambda m: (names.index(m), m) if m in names else (len(names), m)
    return lambda r: (int(r["d"]), int(r["K"]), int(r["seed"]), int(r["n_per_mode"]), rank(r["model"]))


# ---- one cell -------------------------------------------------------------------------------
def _ground_truth(cfg, proc, model, ck, d, K, seed, device):
    """The eval seeds and their fates under the learned sampler (from train.py's cache when
    it is there, otherwise computed and cached now)."""
    process, variant = cfg.process.name, variant_of(cfg)
    n_eval = n_eval_for(cfg, K)
    X = proc.seeds(n_eval, d, seed + 1)
    gt = load_gt_cache(cfg.paths.checkpoints, process, d, K, n_eval, seed, device, variant)
    if gt is None:
        gt = core.label_fate(proc.sample(model, X), ck["means"], ck["R99"])
        save_gt_cache(cfg.paths.checkpoints, process, d, K, gt, n_eval, ck["T"], ck["R99"], seed, variant)
    return X, gt


def eval_one(cfg, d, K, seed, device):
    """All rows of one cell: the cached ones plus the missing (budget, predictor) rows, which
    are computed and saved. None when the cell has no checkpoint."""
    process, variant = cfg.process.name, variant_of(cfg)
    cpath = cell_path(sweep_dir(cfg.paths.output, cfg.run_id, process, cfg.process.T_true, variant),
                      d, K, seed)
    want = {(b, s["name"]) for b in budgets(cfg) for s in _specs(cfg)}
    cached = _load_cell(cpath)
    if bool(cfg.eval.force):
        cached = [r for r in cached if _key(r) not in want]
    todo = want - {_key(r) for r in cached}
    if not todo:
        return cached
    path = ckpt_path(cfg.paths.checkpoints, process, d, K, seed, variant)
    if not os.path.exists(path):
        return None

    model, ck = load_ckpt(path, device)
    means_t, R99, variance = ck["means"], ck["R99"], ck["variance"]
    sigma = float(variance ** 0.5)
    weights = ck.get("weights")                              # the GMM the model was trained on
    w_row = [round(float(x), 4) for x in weights] if weights is not None else None
    proc = make_process(process, means_t, variance, ck["T"], device, cfg, weights)
    proc_true = make_process(process, means_t, variance, int(cfg.process.T_true), device, cfg, weights)

    X_te, gt = _ground_truth(cfg, proc, model, ck, d, K, seed, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cell = {"hall_gt": float((gt == -1).float().mean()), "n_eval": int(gt.numel()),
            "n_mode": int((gt >= 0).sum()), "n_hall": int((gt == -1).sum()), "weights": w_row}

    a = cfg.anchors
    # a small set of seeds labelled by the exact field, used to calibrate altered_knn's cut
    X_cal = proc.seeds(int(a.n_calibrate), d, seed + 2)
    y_cal = proc_true.label(None, X_cal, R99)

    rows = []
    for b in budgets(cfg):
        specs = [s for s in _specs(cfg) if (b, s["name"]) in todo]
        if not specs:
            continue
        P, y = core.ball_anchors(means_t, R99, b, float(a.shell_frac), float(a.shell_sigma),
                                 sigma, int(a.seed), device)                                 # step 1
        A = proc_true.true_field_backtrack(P, inplace=True)                                  # step 2
        del P
        Aa = ya = wa = None
        if any(s["arch"] == "altered_knn" for s in specs):
            r = a.altered_knn
            Pa, ya, wa = core.altered_knn_anchors(means_t, R99, b, int(r.n_rings), float(r.r_max),
                                                  str(r.weight), float(r.w_min), sigma=sigma,
                                                  seed=int(a.seed), device=device)
            Aa = proc_true.true_field_backtrack(Pa, inplace=True)
            del Pa

        for spec in specs:
            t0 = time.time()
            altered = spec["arch"] == "altered_knn"
            nets = (fate.train_ensemble(Aa, ya, K, spec, device, w=wa) if altered
                    else fate.train_ensemble(A, y, K, spec, device))                          # step 3
            if altered and spec["threshold"] == "auto":
                for net in nets:
                    net.calibrate(X_cal, y_cal)
            met = fate.fate_metrics(fate.predict_fate(nets, X_te), gt)                        # steps 4-5
            rows.append({"seed": seed, "d": d, "K": K, "n_per_mode": b,
                         "n_anchors": int(nets[0].X.shape[0]) if hasattr(nets[0], "X") else int(A.shape[0]),
                         "model": spec["name"], "arch": spec["arch"], "hall_gt": cell["hall_gt"],
                         "n_eval": cell["n_eval"], "n_mode": cell["n_mode"], "n_hall": cell["n_hall"],
                         "weights": w_row, "secs": round(time.time() - t0, 1), **met})
            del nets
        del A, Aa                                   # free this budget before the next (bigger) one
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = sorted(cached + rows, key=row_order(cfg))
    _save_cell(cpath, rows)
    return rows


# ---- aggregation and results files -----------------------------------------------------------
AGG_KEYS = ("hall_gt",) + tuple(fate.METRICS)     # per-seed scalars reported as mean +- std


def aggregate(results):
    """One row per (d, K, budget, model): each AGG_KEYS value becomes its mean over seeds with a
    <key>_std companion (population std), plus n_seeds, the seeds and their mixing weights."""
    groups = {}
    for r in results:
        groups.setdefault((r["d"], r["K"], int(r["n_per_mode"]), r["model"]), []).append(r)
    out = []
    for (d, K, b, model), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        row = {"d": d, "K": K, "n_per_mode": b, "n_anchors": rs[0]["n_anchors"],
               "model": model, "arch": rs[0]["arch"], "n_seeds": len(rs),
               "seeds": [r["seed"] for r in rs], "weights": [r.get("weights") for r in rs]}
        for k in AGG_KEYS:
            v = np.array([r[k] for r in rs], dtype=np.float64)
            ok = np.isfinite(v).any()
            row[k] = float(np.nanmean(v)) if ok else float("nan")
            row[k + "_std"] = float(np.nanstd(v)) if ok else float("nan")
        out.append(row)
    return out


def _console(process, d, K, cell, primary):
    """One line per (d, K): hall rate and the primary predictor's best budget, mean +- std."""
    agg = aggregate(cell)
    line = (f"[eval:{process}] d={d:>2} K={K:>2} seeds={agg[0]['n_seeds']} "
            f"hall_gt={agg[0]['hall_gt']:.3f}+-{agg[0]['hall_gt_std']:.3f}")
    prim = [r for r in agg if r["model"] == primary]
    if prim:
        b = max(prim, key=lambda r: r["full_acc"])
        line += (f" | {primary}: full={b['full_acc']:.3f}+-{b['full_acc_std']:.3f}"
                 f" modeF1={b['mode_f1']:.3f}+-{b['mode_f1_std']:.3f}"
                 f" hallF1={b['hall_f1']:.3f}+-{b['hall_f1_std']:.3f} @ {b['n_anchors']} anchors")
    print(line)


def run(cfg):
    device = core.get_device(cfg.device)
    process = cfg.process.name
    n_rows = 0
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            preflight(cfg, int(d), int(K), device)
            cell = []
            for seed in core.seed_list(cfg):
                rows = eval_one(cfg, int(d), int(K), seed, device)
                if rows is None:
                    print(f"[eval:{process}] d={d:>2} K={K:>2} seed={seed:<4} -> no checkpoint, skipped")
                else:
                    cell += rows
            if cell:
                n_rows += len(cell)
                _console(process, d, K, cell, _primary(cfg))
    if bool(cfg.eval.part):          # scripts/main.sh: one job per d, then a single `merge`
        print(f"[eval:{process}] {n_rows} rows in cell files; run stages=[merge] to rebuild results.json")
        return
    merge(cfg)


def _write_results(cfg, out_dir, results):
    """results.json (per-seed rows + aggregate), results.csv (aggregate), results_per_seed.csv."""
    process, variant = cfg.process.name, variant_of(cfg)
    agg = aggregate(results)
    seeds = sorted({int(r["seed"]) for r in results})
    js = os.path.join(out_dir, "results.json")
    with open(js, "w") as f:
        json.dump({"sampler": process, "variant": variant, "weighted": variant == "weighted",
                   "run_id": cfg.run_id,
                   "config_sweep": {"d": sorted({int(r["d"]) for r in results}),
                                    "K": sorted({int(r["K"]) for r in results}),
                                    "anchors": sorted({int(r["n_per_mode"]) for r in results}),
                                    "T_true": int(cfg.process.T_true), "seeds": seeds},
                   "primary": _primary(cfg), "metrics": list(fate.METRICS),
                   "results": results, "aggregate": agg}, f, indent=2)

    M = fate.METRICS
    with open(os.path.join(out_dir, "results.csv"), "w") as f:
        f.write("sampler,d,K,n_seeds,hall_gt,hall_gt_std,model,arch,n_per_mode,n_anchors,"
                + ",".join(f"{k},{k}_std" for k in M) + "\n")
        for r in agg:
            f.write(f"{process},{r['d']},{r['K']},{r['n_seeds']},{r['hall_gt']:.4f},{r['hall_gt_std']:.4f},"
                    f"{r['model']},{r['arch']},{r['n_per_mode']},{r['n_anchors']},"
                    + ",".join(f"{r[k]:.4f},{r[k + '_std']:.4f}" for k in M) + "\n")
    with open(os.path.join(out_dir, "results_per_seed.csv"), "w") as f:
        f.write("sampler,seed,d,K,hall_gt,model,arch,n_per_mode,n_anchors," + ",".join(M) + ",weights\n")
        for r in results:
            w = " ".join(f"{x:.4f}" for x in r["weights"]) if r.get("weights") else ""
            f.write(f"{process},{r['seed']},{r['d']},{r['K']},{r['hall_gt']:.4f},{r['model']},{r['arch']},"
                    f"{r['n_per_mode']},{r['n_anchors']}," + ",".join(f"{r[k]:.4f}" for k in M) + f",{w}\n")
    print(f"[eval:{process}] wrote {js} and the csv files (mean +- std over {len(seeds)} seeds)")


def merge(cfg):
    """Rebuild results.json / csv from every cell file of this (process, variant, T_true),
    whichever invocation or machine wrote it. Locked: parallel evaluate jobs may all call it."""
    process, variant = cfg.process.name, variant_of(cfg)
    out_dir = sweep_dir(cfg.paths.output, cfg.run_id, process, cfg.process.T_true, variant)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        results = sorted((r for p in glob.glob(os.path.join(out_dir, "cells", "*.json"))
                          for r in _load_cell(p)), key=row_order(cfg))
        if not results:
            raise FileNotFoundError(f"[merge:{process}] no cell files under {out_dir}/cells")
        cells = {(r["d"], r["K"], r["seed"]) for r in results}
        print(f"[merge:{process}/{variant}] T={int(cfg.process.T_true)}: {len(cells)} cells, "
              f"d={sorted({c[0] for c in cells})}, {len(results)} rows")
        _write_results(cfg, out_dir, results)
