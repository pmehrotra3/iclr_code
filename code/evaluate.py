"""
evaluate.py — Stage 2. The seed-fate atlas, run identically for every process (ddim, flow).

Per (d, K) cell, per anchor budget:
  1. Plant labelled anchors in DATA space (core.ball_anchors: mode balls + a hallucination
     band; core.altered_knn_anchors: mode-only weighted rings). No model involved.
  2. Backtrack them to SEED space with the exact analytic field (proc.true_field_backtrack).
  3. Fit each predictor in classifier.models on the (seed, label) pairs -- knn, altered_knn,
     quadratic, polar3. altered_knn's confidence cut is calibrated on fresh seeds labelled by the
     exact score (proc.label(None, ...)).
  4. Ground truth: eval.n_eval_per_mode * K fresh seeds pushed through the LEARNED sampler,
     labelled with L.
  5. Score every predictor against that ground truth (fate.fate_metrics).

The whole thing is repeated for every seed in core.seed_list(cfg) (each seed = its own mode
placement, learned model and eval seeds) and every metric is reported as mean +- std over the
seeds. Results go to output/<run_id>/<process>/<variant>/T<T_true>/:
    cells/d<d>_K<K>_s<seed>.json   the rows of one (d, K, seed) cell -- THE record. A cell file
                           that already holds a (budget, predictor) row is not recomputed
                           (eval.force=true recomputes); a new budget or predictor adds only its
                           own rows. Every row is independent of the others (fixed anchor seed,
                           fixed fit seeds), so computing a cell in pieces gives the same rows.
    results.json           rebuilt from ALL cell files of the run after every invocation, in
                           (d, K, seed, budget, predictor) order, so a sweep evaluated in pieces
                           gives the results.json of one big invocation:
                           "results": one row per (seed, d, K, budget, model)
                           "aggregate": one row per (d, K, budget, model) with <metric> = mean
                           over seeds, <metric>_std = std (ddof=0), n_seeds
    results.csv            the aggregate (mean, std columns)
    results_per_seed.csv   the per-seed rows
No plotting here.
"""
from __future__ import annotations
import os
import json
import glob
import fcntl
import time
import numpy as np
import torch
from omegaconf import OmegaConf

import core
import fate
from processes.factory import make_process
from train import ckpt_path, sweep_dir, load_gt_cache, save_gt_cache, n_eval_for, variant_of


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["arch"]
    m = core.ScoreNet(ck["d"], a["h"], a["nb"], a["td"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    ck["means"] = ck["means"].to(device)
    return m, ck


def _spec(cfg, m) -> dict:
    """Merge a classifier entry with the preset's shared defaults into a plain dict."""
    base = OmegaConf.to_container(cfg.classifier.get("_shared", {}), resolve=True) or {}
    spec = dict(base)
    spec.update(OmegaConf.to_container(m, resolve=True))
    return spec


def budgets(cfg):
    b = cfg.sweep.anchors
    return [int(x) for x in ([b] if isinstance(b, (int, float, str)) else b)]


def anchor_memory_gb(cfg, d, K):
    """Estimated peak GPU memory (GB) of eval_one for the LARGEST anchor budget, from the
    standing fp32 tensors: the seed-space ball anchors plus the altered_knn ring anchors
    (each set is allocated once and backtracked in place, so it exists exactly once), plus
    the eval seeds, plus ~3 GB of working memory (a 50k-row backtrack chunk, the
    classifiers, kNN distance blocks). Scales as budget * K * d: d=512 / K=16 / 300k per
    mode is ~28 GB."""
    a = cfg.anchors
    b = max(budgets(cfg))
    f = 4 * d                                          # bytes per fp32 point
    n_ball = b * K * (1 + float(a.shell_frac))
    _models = cfg.classifier.models
    _entries = _models.values() if OmegaConf.is_dict(_models) else _models
    need_altered = any(_spec(cfg, m)["arch"] == "altered_knn" for m in _entries)
    n_alt = b * K if need_altered else 0
    x_te = n_eval_for(cfg, K) * f
    return ((n_ball + n_alt) * f + x_te + 3e9) / 1e9


def preflight(cfg, d, K, device):
    """Abort with a clear message (instead of a CUDA OOM hours in) when the largest budget of
    this cell cannot fit the GPU; prints the estimate either way."""
    need = anchor_memory_gb(cfg, d, K)
    if device.type != "cuda":
        return
    free, total = (x / 1e9 for x in torch.cuda.mem_get_info(device))
    tag = f"[eval:{cfg.process.name}] d={d} K={K} budgets={budgets(cfg)}"
    print(f"{tag}: est. peak {need:.1f} GB, GPU free {free:.1f}/{total:.1f} GB")
    if need > 0.92 * free:
        raise MemoryError(f"{tag}: estimated peak {need:.1f} GB exceeds free GPU memory "
                          f"{free:.1f} GB; lower sweep.anchors, K, or eval.n_eval_per_mode "
                          f"(or run one eval job per GPU)")


def stratified_eval_idx(gt, K, n_total, seed):
    """Indices of n_total eval seeds drawn without replacement from the pool labelled gt:
    hallucinations (-1) in the pool's ratio, the remaining seeds split evenly across the K
    modes (the first n % K modes take one extra). Raises if a class has too few seeds."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    gt_c = gt.cpu()
    n_hall = int(round(n_total * float((gt_c == -1).float().mean())))
    base, extra = divmod(n_total - n_hall, K)
    want = [(-1, n_hall)] + [(k, base + (1 if k < extra else 0)) for k in range(K)]
    idx = []
    for lab, n in want:
        pool = torch.nonzero(gt_c == lab).squeeze(1)
        if n > pool.numel():
            raise ValueError(f"eval_per_anchor: need {n} seeds of class {lab}, pool has "
                             f"{pool.numel()}; raise eval.n_eval_per_mode")
        idx.append(pool[torch.randperm(pool.numel(), generator=g)[:n]])
    return torch.cat(idx).to(gt.device)


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
    os.replace(tmp, path)                                   # atomic: a crash never leaves half a file


def _key(r):
    return int(r["n_per_mode"]), r["model"]


def _specs(cfg):
    # classifier.models may be a list (legacy) or a name->spec dict (composed per-method
    # from conf/classifier/method/*.yaml). Iterate the entries either way.
    _models = cfg.classifier.models
    _entries = _models.values() if OmegaConf.is_dict(_models) else _models
    return [_spec(cfg, m) for m in _entries]


def row_order(cfg):
    """(d, K, seed, budget, predictor in config order; unknown predictors after, by name)."""
    names = [s["name"] for s in _specs(cfg)]
    rank = lambda m: (names.index(m), m) if m in names else (len(names), m)
    return lambda r: (int(r["d"]), int(r["K"]), int(r["seed"]), int(r["n_per_mode"]), rank(r["model"]))


def wanted(cfg):
    """The (budget, predictor) rows this invocation asks of every cell."""
    return {(int(b), s["name"]) for b in budgets(cfg) for s in _specs(cfg)}


def eval_one(cfg, d, K, seed, device):
    """Evaluate one (d, K) cell for one repeat `seed`, resuming from its cell file: only the
    (budget, predictor) rows it does not hold yet are computed. Returns every row of the cell
    (cached and new); None when the cell has no checkpoint."""
    sampler = cfg.process.name
    variant = variant_of(cfg)
    out_dir = sweep_dir(cfg.paths.output, cfg.run_id, sampler, cfg.process.T_true, variant)
    cpath = cell_path(out_dir, d, K, seed)
    want = wanted(cfg)
    cached = _load_cell(cpath)
    if bool(cfg.eval.get("force", False)):
        cached = [r for r in cached if _key(r) not in want]
    todo = want - {_key(r) for r in cached}
    if not todo:
        return cached
    path = ckpt_path(cfg.paths.checkpoints, sampler, d, K, seed, variant)
    if not os.path.exists(path):
        return None

    model, ck = load_ckpt(path, device)
    means_t = ck["means"]
    T, R99, variance = ck["T"], ck["R99"], ck["variance"]
    sigma = float(variance ** 0.5)
    a = cfg.anchors
    # mixing weights the model was trained on (older checkpoints: none = uniform); the exact
    # reference process must describe the same weighted GMM
    weights = ck.get("weights", None)
    # recorded per row so the results show which imbalance each repeat was trained on
    w_row = [round(float(x), 4) for x in weights] if weights is not None else None

    proc = make_process(sampler, means_t, variance, T, device, cfg, weights)
    proc_true = make_process(sampler, means_t, variance, int(cfg.process.T_true), device, cfg,
                             weights)

    # ---- ground truth: where the LEARNED model actually sends each eval seed ----
    # Reuse the cache written by train (identical for every T_true); recompute + write through
    # only on a miss, so a full T sweep runs the N-seed forward pass at most once per (d, K).
    n_eval = n_eval_for(cfg, K)
    X_te = proc.seeds(n_eval, d, seed + 1)
    gt = load_gt_cache(cfg.paths.checkpoints, sampler, d, K, n_eval, seed, device, variant)
    if gt is None:
        gt = core.label_fate(proc.sample(model, X_te), means_t, R99)
        save_gt_cache(cfg.paths.checkpoints, sampler, d, K, gt, n_eval, T, R99, seed, variant)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    m_mode, m_hall = gt >= 0, gt == -1
    hall_gt = float(m_hall.float().mean())

    all_specs = _specs(cfg)

    # seeds labelled by the EXACT score at T_true: altered_knn's calibration set
    X_cal = proc.seeds(int(a.n_calibrate), d, seed + 2)
    y_cal = proc_true.label(None, X_cal, R99)

    per_anchor = cfg.eval.get("eval_per_anchor", None)

    rows = []
    for b in budgets(cfg):
        specs = [s for s in all_specs if (int(b), s["name"]) in todo]
        if not specs:
            continue
        need_altered = any(s["arch"] == "altered_knn" for s in specs)
        # scored seeds: all n_eval, or (eval_per_anchor) a stratified subset of the pool sized
        # per_anchor * (ball + shell anchors) -- the same subset for every predictor
        if per_anchor:
            n_ball = K * (int(b) + max(1, int(round(float(a.shell_frac) * int(b)))))
            sel = stratified_eval_idx(gt, K, int(round(float(per_anchor) * n_ball)), seed + 3)
            X_b, gt_b = X_te[sel], gt[sel]
        else:
            X_b, gt_b = X_te, gt
        # step 1 + 2: plant in data space, backtrack to seed space with the exact field
        P, y = core.ball_anchors(means_t, R99, int(b), float(a.shell_frac), float(a.shell_sigma),
                                 sigma, int(a.seed), device)
        A = proc_true.true_field_backtrack(P, inplace=True)   # P is overwritten: one (n, d) tensor, not two
        del P
        Aa = ya = wa = None
        if need_altered:
            rg = a.altered_knn
            Pa, ya, wa = core.altered_knn_anchors(
                means_t, R99, int(b), int(rg.n_rings), float(rg.r_max), str(rg.weight),
                float(rg.w_min), sigma=sigma, seed=int(a.seed), device=device)
            Aa = proc_true.true_field_backtrack(Pa, inplace=True)
            del Pa

        for spec in specs:                                          # step 3
            t0 = time.time()
            altered = spec["arch"] == "altered_knn"
            nets = (fate.train_ensemble(Aa, ya, K, spec, device, w=wa) if altered
                    else fate.train_ensemble(A, y, K, spec, device))
            if altered and spec.get("threshold") == "auto":
                for net in nets:
                    net.calibrate(X_cal, y_cal)
            n_anchors = int(nets[0].X.shape[0]) if hasattr(nets[0], "X") else int(A.shape[0])
            met = fate.fate_metrics(fate.predict_fate(nets, X_b), gt_b)   # steps 4 + 5
            rows.append({"seed": seed, "d": d, "K": K, "n_per_mode": int(b), "n_anchors": n_anchors,
                         "model": spec["name"], "arch": spec["arch"],
                         "hall_gt": hall_gt, "n_eval": int(gt_b.numel()),
                         "n_mode": int((gt_b >= 0).sum()), "n_hall": int((gt_b == -1).sum()),
                         "weights": w_row,
                         "secs": round(time.time() - t0, 1), **met})
            del nets
        del A, Aa, X_b, gt_b                               # release this budget's anchors before the next (bigger) one
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = sorted(cached + rows, key=row_order(cfg))
    _save_cell(cpath, rows)
    return rows


AGG_KEYS = ("hall_gt",) + tuple(fate.METRICS)     # per-seed scalars summarised as mean +- std


def aggregate(results):
    """Collapse per-seed rows into one row per (d, K, n_per_mode, model): every key in AGG_KEYS
    becomes its mean over seeds, with a <key>_std companion (population std, ddof=0), plus
    n_seeds and the seed list. n_anchors is the same for every seed and is carried through."""
    groups = {}
    for r in results:
        groups.setdefault((r["d"], r["K"], int(r["n_per_mode"]), r["model"]), []).append(r)
    out = []
    for (d, K, b, model), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        row = {"d": d, "K": K, "n_per_mode": b, "n_anchors": rs[0]["n_anchors"],
               "model": model, "arch": rs[0]["arch"],
               "n_seeds": len(rs), "seeds": [r["seed"] for r in rs],
               "weights": [r.get("weights") for r in rs]}   # per-seed mixing weights, in seed order
        for k in AGG_KEYS:
            v = np.array([r[k] for r in rs], dtype=np.float64)
            row[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[k + "_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
        out.append(row)
    return out


def run(cfg):
    device = core.get_device(cfg.device)
    sampler = cfg.process.name
    variant = variant_of(cfg)
    out_dir = sweep_dir(cfg.paths.output, cfg.run_id, sampler, cfg.process.T_true, variant)
    os.makedirs(out_dir, exist_ok=True)
    # console model: classifier.primary if set, else the first configured model
    _models = cfg.classifier.models
    _names = list(_models.keys()) if OmegaConf.is_dict(_models) else [str(m) for m in _models]
    primary = str(cfg.classifier.get("primary", _names[0] if _names else ""))
    seeds = core.seed_list(cfg)

    results = []
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            preflight(cfg, int(d), int(K), device)
            cell = []
            for seed in seeds:
                rows = eval_one(cfg, int(d), int(K), seed, device)
                if rows is None:
                    print(f"[eval:{sampler}] d={d:>2} K={K:>2} seed={seed:<4} -> no checkpoint, skipped")
                    continue
                cell += rows
            if not cell:
                continue
            results += cell
            # console: the primary model's best budget (by mean full accuracy) for this cell
            agg = aggregate(cell)
            prim = [r for r in agg if r["model"] == primary]
            best = max(prim, key=lambda r: r["full_acc"]) if prim else None
            hg, hs = agg[0]["hall_gt"], agg[0]["hall_gt_std"]
            line = f"[eval:{sampler}] d={d:>2} K={K:>2} seeds={agg[0]['n_seeds']} hall_gt={hg:.3f}+-{hs:.3f}"
            if best:
                line += (f" | {primary}: full={best['full_acc']:.3f}+-{best['full_acc_std']:.3f}"
                         f" modeF1={best['mode_f1']:.3f}+-{best['mode_f1_std']:.3f}"
                         f" hallF1={best['hall_f1']:.3f}+-{best['hall_f1_std']:.3f}"
                         f" @ {best['n_anchors']} anchors")
            print(line)

    if bool(cfg.eval.get("part", False)):
        # scripts/main.sh fans evaluate out per d and runs `merge` once afterwards
        print(f"[eval:{sampler}] {len(results)} rows in {out_dir}/cells; run stages=[merge] to rebuild results.json")
        return {"cells": os.path.join(out_dir, "cells"), "n_results": len(results)}
    return merge(cfg)


def collect(cfg, out_dir):
    """Every row of every cell file of the run, in single-invocation order."""
    results = []
    for p in glob.glob(os.path.join(out_dir, "cells", "*.json")):
        results += _load_cell(p)
    return sorted(results, key=row_order(cfg))


def _write_results(cfg, out_dir, sampler, variant, primary, results):
    """Aggregate the per-seed rows and write results.json / results.csv / results_per_seed.csv."""
    agg = aggregate(results)
    seeds = sorted({int(r["seed"]) for r in results})
    js = os.path.join(out_dir, "results.json")
    with open(js, "w") as f:
        json.dump({"sampler": sampler, "variant": variant, "weighted": variant == "weighted",
                   "run_id": cfg.run_id,
                   "config_sweep": {"d": sorted({int(r["d"]) for r in results}),
                                    "K": sorted({int(r["K"]) for r in results}),
                                    "anchors": sorted({int(r["n_per_mode"]) for r in results}),
                                    "T_true": int(cfg.process.T_true), "seeds": seeds},
                   "primary": primary, "metrics": list(fate.METRICS),
                   "results": results, "aggregate": agg}, f, indent=2)

    csv = os.path.join(out_dir, "results.csv")
    with open(csv, "w") as f:
        f.write("sampler,d,K,n_seeds,hall_gt,hall_gt_std,model,arch,n_per_mode,n_anchors,"
                + ",".join(f"{k},{k}_std" for k in fate.METRICS) + "\n")
        for r in agg:
            f.write(f"{sampler},{r['d']},{r['K']},{r['n_seeds']},{r['hall_gt']:.4f},{r['hall_gt_std']:.4f},"
                    f"{r['model']},{r['arch']},{r['n_per_mode']},{r['n_anchors']},"
                    + ",".join(f"{r[k]:.4f},{r[k + '_std']:.4f}" for k in fate.METRICS) + "\n")

    csv_s = os.path.join(out_dir, "results_per_seed.csv")
    with open(csv_s, "w") as f:
        f.write("sampler,seed,d,K,hall_gt,model,arch,n_per_mode,n_anchors,"
                + ",".join(fate.METRICS) + ",weights\n")
        for r in results:
            w = " ".join(f"{x:.4f}" for x in r["weights"]) if r.get("weights") else ""
            f.write(f"{sampler},{r['seed']},{r['d']},{r['K']},{r['hall_gt']:.4f},{r['model']},{r['arch']},"
                    f"{r['n_per_mode']},{r['n_anchors']},"
                    + ",".join(f"{r[k]:.4f}" for k in fate.METRICS) + f",{w}\n")

    print(f"[eval:{sampler}] wrote {js}, {csv} (mean +- std over {len(seeds)} seeds) and {csv_s}")
    return {"json": js, "csv": csv, "n_results": len(results), "n_aggregate": len(agg)}


def merge(cfg):
    """Rebuild results.json / csv from every cell file of the run (output/<run_id>/<process>/
    <variant>/T<T>/cells/), whichever invocations wrote them. Locked, so the parallel evaluate
    jobs of scripts/main.sh can each call it."""
    sampler = cfg.process.name
    variant = variant_of(cfg)
    out_dir = sweep_dir(cfg.paths.output, cfg.run_id, sampler, cfg.process.T_true, variant)
    os.makedirs(out_dir, exist_ok=True)
    _models = cfg.classifier.models
    _names = list(_models.keys()) if OmegaConf.is_dict(_models) else [str(m) for m in _models]
    primary = str(cfg.classifier.get("primary", _names[0] if _names else ""))
    with open(os.path.join(out_dir, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        results = collect(cfg, out_dir)
        if not results:
            raise FileNotFoundError(f"[merge:{sampler}] no cell files under {out_dir}/cells")
        cells = {(r["d"], r["K"], r["seed"]) for r in results}
        print(f"[merge:{sampler}/{variant}] T={int(cfg.process.T_true)}: {len(cells)} cells, "
              f"d={sorted({c[0] for c in cells})}, {len(results)} rows")
        return _write_results(cfg, out_dir, sampler, variant, primary, results)
