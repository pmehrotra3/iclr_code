"""
evaluate.py — Stage 2. The seed-fate atlas, run identically for every process (ddim, flow).

Per (d, K) cell, per anchor budget:
  1. Plant labelled anchors in DATA space (core.ball_anchors: mode balls + a hallucination
     band; core.altered_knn_anchors: mode-only weighted rings). No model involved.
  2. Backtrack them to SEED space with the exact analytic field (proc.true_field_backtrack).
  3. Fit each predictor in classifier.models on the (seed, label) pairs -- knn, altered_knn,
     polar3. altered_knn's confidence cut is calibrated on fresh seeds labelled by the exact
     score (proc.label(None, ...)).
  4. Ground truth: eval.n_eval fresh seeds pushed through the LEARNED sampler, labelled with L.
  5. Score every predictor against that ground truth (fate.fate_metrics).

Results go to output/<process>/<run_id>/results.{json,csv}: one row per (d, K, budget, model).
No plotting here.
"""
from __future__ import annotations
import os
import json
import time
import torch
from omegaconf import OmegaConf

import core
import fate
from processes.factory import make_process
from train import ckpt_path, run_dir


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


def eval_one(cfg, d, K, device):
    """Evaluate one (d, K) cell: every predictor at every anchor budget. Returns result rows."""
    sampler = cfg.process.name
    path = ckpt_path(cfg.paths.data, sampler, d, K)
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
    X_te = proc.seeds(cfg.eval.n_eval, d, cfg.seed + 1)
    gt = core.label_fate(proc.sample(model, X_te), means_t, R99)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    m_mode, m_hall = gt >= 0, gt == -1
    hall_gt = float(m_hall.float().mean())

    specs = [_spec(cfg, m) for m in cfg.classifier.models]
    need_altered = any(s["arch"] == "altered_knn" for s in specs)

    # seeds labelled by the EXACT score at T_true: altered_knn's calibration set
    X_cal = proc.seeds(int(a.n_calibrate), d, cfg.seed + 2)
    y_cal = proc_true.label(None, X_cal, R99)

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
            rows.append({"d": d, "K": K, "n_per_mode": int(b), "n_anchors": n_anchors,
                         "model": spec["name"], "arch": spec["arch"],
                         "hall_gt": hall_gt, "n_mode": int(m_mode.sum()), "n_hall": int(m_hall.sum()),
                         "secs": round(time.time() - t0, 1), **met})

    return rows


def run(cfg):
    device = core.get_device(cfg.device)
    sampler = cfg.process.name
    out_dir = run_dir(cfg.paths.output, sampler, cfg.run_id)
    os.makedirs(out_dir, exist_ok=True)
    primary = str(cfg.classifier.primary)

    results = []
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            rows = eval_one(cfg, int(d), int(K), device)
            if rows is None:
                print(f"[eval:{sampler}] d={d:>2} K={K:>2} -> no checkpoint, skipped")
                continue
            results += rows
            # console: the primary model's best budget for this cell
            prim = [r for r in rows if r["model"] == primary]
            best = max(prim, key=lambda r: r["full_acc"]) if prim else None
            hg = rows[0]["hall_gt"]
            line = f"[eval:{sampler}] d={d:>2} K={K:>2} hall_gt={hg:.3f}"
            if best:
                line += (f" | {primary}: full={best['full_acc']:.3f} modeF1={best['mode_f1']:.3f} "
                         f"hallF1={best['hall_f1']:.3f} @ {best['n_anchors']} anchors")
            print(line)

    js = os.path.join(out_dir, "results.json")
    with open(js, "w") as f:
        json.dump({"sampler": sampler, "run_id": cfg.run_id,
                   "config_sweep": {"d": list(cfg.sweep.d), "K": list(cfg.sweep.K),
                                    "anchors": budgets(cfg), "T_true": int(cfg.process.T_true)},
                   "primary": primary, "metrics": list(fate.METRICS),
                   "results": results}, f, indent=2)

    csv = os.path.join(out_dir, "results.csv")
    with open(csv, "w") as f:
        f.write("sampler,d,K,hall_gt,model,arch,n_per_mode,n_anchors,"
                + ",".join(fate.METRICS) + "\n")
        for r in results:
            f.write(f"{sampler},{r['d']},{r['K']},{r['hall_gt']:.4f},{r['model']},{r['arch']},"
                    f"{r['n_per_mode']},{r['n_anchors']},"
                    + ",".join(f"{r[k]:.4f}" for k in fate.METRICS) + "\n")

    print(f"[eval:{sampler}] wrote {js} and {csv}")
    return {"json": js, "csv": csv, "n_results": len(results)}
