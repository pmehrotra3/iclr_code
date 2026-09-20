"""
evaluate.py — Stage 2. The seed-fate atlas, run identically for every process (ddim, flow).

Per (d, K) cell, per anchor budget:
  1. Plant labelled anchors in DATA space (core.ball_anchors: mode balls + a hallucination
     band; core.altered_knn_anchors: mode-only weighted rings). No model involved.
  2. Backtrack them to SEED space with the exact analytic field (proc.true_field_backtrack).
  3. Fit each predictor in classifier.models on the (seed, label) pairs -- knn, altered_knn,
     quadratic, polar8. altered_knn's confidence cut is calibrated on fresh seeds labelled by the
     exact score (proc.label(None, ...)); each parametric model also gets a <name>_cal variant.
  4. Ground truth: eval.n_eval fresh seeds pushed through the LEARNED sampler, labelled with L.
  5. Score every predictor against that ground truth (fate.fate_metrics).

The whole thing is repeated for every seed in core.seed_list(cfg) (each seed = its own mode
placement, learned model and eval seeds) and every metric is reported as mean +- std over the
seeds. Results go to output/<run_id>/<process>/T<T_true>/:
    results.json           "results": one row per (seed, d, K, budget, model)
                           "aggregate": one row per (d, K, budget, model) with <metric> = mean
                           over seeds, <metric>_std = std (ddof=0), n_seeds
    results.csv            the aggregate (mean, std columns)
    results_per_seed.csv   the per-seed rows
No plotting here.
"""
from __future__ import annotations
import os
import json
import time
import numpy as np
import torch
from omegaconf import OmegaConf

import core
import fate
from processes.factory import make_process
from train import ckpt_path, sweep_dir, load_gt_cache, save_gt_cache


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


def eval_one(cfg, d, K, seed, device):
    """Evaluate one (d, K) cell for one repeat `seed`: every predictor at every anchor budget.
    Returns result rows."""
    sampler = cfg.process.name
    path = ckpt_path(cfg.paths.data, sampler, d, K, seed)
    if not os.path.exists(path):
        return None

    model, ck = load_ckpt(path, device)
    means_t = ck["means"]
    T, R99, variance = ck["T"], ck["R99"], ck["variance"]
    sigma = float(variance ** 0.5)
    a = cfg.anchors

    proc = make_process(sampler, means_t, variance, T, device, cfg)
    proc_true = make_process(sampler, means_t, variance, int(cfg.process.T_true), device, cfg)

    # ---- ground truth: where the LEARNED model actually sends each eval seed ----
    # Reuse the cache written by train (identical for every T_true); recompute + write through
    # only on a miss, so a full T sweep runs the N-seed forward pass at most once per (d, K).
    n_eval = int(cfg.eval.n_eval)
    X_te = proc.seeds(n_eval, d, seed + 1)
    gt = load_gt_cache(cfg.paths.data, sampler, d, K, n_eval, seed, device)
    if gt is None:
        gt = core.label_fate(proc.sample(model, X_te), means_t, R99)
        save_gt_cache(cfg.paths.data, sampler, d, K, gt, n_eval, T, R99, seed)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    m_mode, m_hall = gt >= 0, gt == -1
    hall_gt = float(m_hall.float().mean())

    # classifier.models may be a list (legacy) or a name->spec dict (composed per-method
    # from conf/classifier/method/*.yaml). Iterate the entries either way.
    _models = cfg.classifier.models
    _entries = _models.values() if OmegaConf.is_dict(_models) else _models
    specs = [_spec(cfg, m) for m in _entries]
    need_altered = any(s["arch"] == "altered_knn" for s in specs)

    # seeds labelled by the EXACT score at T_true: altered_knn's calibration set AND the
    # prior-calibration target for the parametric models (Step 4).
    X_cal = proc.seeds(int(a.n_calibrate), d, seed + 2)
    y_cal = proc_true.label(None, X_cal, R99)
    cal_rate = float((y_cal == -1).float().mean())      # exact-score hallucination fraction

    rows = []
    for b in budgets(cfg):
        # step 1 + 2: plant in data space, backtrack to seed space with the exact field
        P, y = core.ball_anchors(means_t, R99, int(b), float(a.shell_frac), float(a.shell_sigma),
                                 sigma, int(a.seed), device)
        A = proc_true.true_field_backtrack(P)
        Aa = ya = wa = None
        if need_altered:
            rg = a.altered_knn
            Pa, ya, wa = core.altered_knn_anchors(
                means_t, R99, int(b), int(rg.n_rings), float(rg.r_max), str(rg.weight),
                float(rg.w_min), sigma=sigma, seed=int(a.seed), device=device)
            Aa = proc_true.true_field_backtrack(Pa)

        for spec in specs:                                          # step 3
            t0 = time.time()
            altered = spec["arch"] == "altered_knn"
            nets = (fate.train_ensemble(Aa, ya, K, spec, device, w=wa) if altered
                    else fate.train_ensemble(A, y, K, spec, device))
            if altered and spec.get("threshold") == "auto":
                for net in nets:
                    net.calibrate(X_cal, y_cal)
            n_anchors = int(nets[0].X.shape[0]) if hasattr(nets[0], "X") else int(A.shape[0])
            met = fate.fate_metrics(fate.predict_fate(nets, X_te), gt)   # steps 4 + 5
            rows.append({"seed": seed, "d": d, "K": K, "n_per_mode": int(b), "n_anchors": n_anchors,
                         "model": spec["name"], "arch": spec["arch"],
                         "hall_gt": hall_gt, "n_mode": int(m_mode.sum()), "n_hall": int(m_hall.sum()),
                         "secs": round(time.time() - t0, 1), **met})

            # Step 4 -- prior calibration: shift the hallucination logit so the parametric model
            # calls exactly the exact-score hallucination fraction; reported as <name>_cal.
            if spec["arch"] in fate.PARAMETRIC:
                bias = fate.hall_bias_for_rate(nets, X_cal, cal_rate)
                met_c = fate.fate_metrics(fate.predict_fate(nets, X_te, hall_bias=bias), gt)
                rows.append({"seed": seed, "d": d, "K": K, "n_per_mode": int(b), "n_anchors": n_anchors,
                             "model": spec["name"] + "_cal", "arch": spec["arch"],
                             "hall_gt": hall_gt, "n_mode": int(m_mode.sum()), "n_hall": int(m_hall.sum()),
                             "secs": round(time.time() - t0, 1), **met_c})

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
               "n_seeds": len(rs), "seeds": [r["seed"] for r in rs]}
        for k in AGG_KEYS:
            v = np.array([r[k] for r in rs], dtype=np.float64)
            row[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[k + "_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
        out.append(row)
    return out


def run(cfg):
    device = core.get_device(cfg.device)
    sampler = cfg.process.name
    out_dir = sweep_dir(cfg.paths.output, cfg.run_id, sampler, cfg.process.T_true)
    os.makedirs(out_dir, exist_ok=True)
    # console model: classifier.primary if set, else the first configured model
    _models = cfg.classifier.models
    _names = list(_models.keys()) if OmegaConf.is_dict(_models) else [str(m) for m in _models]
    primary = str(cfg.classifier.get("primary", _names[0] if _names else ""))
    seeds = core.seed_list(cfg)

    results = []
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
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

    agg = aggregate(results)
    js = os.path.join(out_dir, "results.json")
    with open(js, "w") as f:
        json.dump({"sampler": sampler, "run_id": cfg.run_id,
                   "config_sweep": {"d": list(cfg.sweep.d), "K": list(cfg.sweep.K),
                                    "anchors": budgets(cfg), "T_true": int(cfg.process.T_true),
                                    "seeds": seeds},
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
                + ",".join(fate.METRICS) + "\n")
        for r in results:
            f.write(f"{sampler},{r['seed']},{r['d']},{r['K']},{r['hall_gt']:.4f},{r['model']},{r['arch']},"
                    f"{r['n_per_mode']},{r['n_anchors']},"
                    + ",".join(f"{r[k]:.4f}" for k in fate.METRICS) + "\n")

    print(f"[eval:{sampler}] wrote {js}, {csv} (mean +- std over {len(seeds)} seeds) and {csv_s}")
    return {"json": js, "csv": csv, "n_results": len(results), "n_aggregate": len(agg)}
