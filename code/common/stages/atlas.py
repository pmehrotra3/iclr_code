"""stages/atlas.py — the ring atlas: true-score backtrack -> seed-fate predictor.

Two sweeps run inside each (d, K) cell: the number of true-score steps T, and the anchor
budget (anchors.budgets). Every predictor is fitted at every budget, so the figures can
report the BEST budget per cell and say which one won.

Per (d, K):
  1. learned sampler at sweep.T_train (trained on demand until its hallucination rate is
     <= train.hall_target; cached in checkpoints/ unless train.force_retrain)
  2. ground truth: eval.n_eval seeds pushed FORWARD through the learned sampler, labeled by fate
  3. for every T in sweep.T_atlas:
       - the analytic forward pass at T (eval.analytic) -- independent of the anchors
       - for every budget b in anchors.budgets:
           data-space anchors on rings around each mode (gmm.ring_anchors: b points uniform in
           the R99 ball -> label k; shell R99..(1+w)R99 -> hallucination), backtracked to seed
           space with the TRUE score at T steps, then every classifier in classifier.models is
           fitted on the (seed, label) anchors and scored on the learned-sampler ground truth

Writes, under output/<process>/:
  atlas_results.json   every cell, every T, every budget (the figures read this)
  summary.csv          mean over cells of each metric, per T and model, at the best budget
  T_<T>/results.csv    flat per-cell rows for that T, one line per (model, budget)

The backtracked anchor sets themselves are NOT written: only d=2 is ever drawn and
proc_true.true_backward regenerates any of them in well under a second.
"""
from __future__ import annotations
import os
import json
import time
import numpy as np
import torch
from omegaconf import OmegaConf

from common import fate, gmm, checkpoint, utils
from common.process import make_process
from common.stages.train import train_one
from common.stages.evaluate import _spec, _label

BEST_KEY = "full_acc"          # which metric picks the winning budget in summary.csv


def run_dir(cfg):
    """output/<process> -- the atlas pipeline writes one tree per process, no run_tag."""
    return os.path.join(cfg.paths.output, cfg.process)


def budgets(cfg):
    b = cfg.anchors.budgets
    return [int(x) for x in ([b] if isinstance(b, (int, float, str)) else b)]


def atlas_one(cfg, d, K, device):
    pname, T_train = cfg.process, int(cfg.sweep.T_train)
    info = train_one(cfg, d, K, device)                      # cached unless force_retrain
    model, ck = checkpoint.load(info["path"], device)
    means_t, R99, variance = ck["means"], ck["R99"], ck["variance"]
    proc = make_process(pname, means_t, variance, T_train, device, cfg)
    X_te, gt = _label(proc, model, R99, d, cfg.eval.n_eval, cfg.seed + 1)
    cell = {"d": d, "K": K, "T_train": T_train, "hall_learned": float(ck["hall_rate"]),
            "hall_gt": float((gt == -1).float().mean()), "n_eval": int(cfg.eval.n_eval),
            "n_mode": int((gt >= 0).sum()), "n_hall": int((gt == -1).sum()), "per_T": []}
    a = cfg.anchors
    specs = [_spec(cfg, m) for m in cfg.classifier.models]
    need_rings = any(s.get("anchors", "shell") == "rings" for s in specs)
    bs = budgets(cfg)

    # data-space anchors depend only on the budget, so build them once per budget
    anchor_sets = {}
    for b in bs:
        P, yA = gmm.ring_anchors(means_t, R99, b, float(a.shell_w), float(a.shell_frac),
                                 int(a.seed), device)
        entry = {"P": P, "y": yA}
        if need_rings:
            rg = a.rings
            entry["Pr"], entry["yr"], entry["wr"] = gmm.ring_anchors_weighted(
                means_t, R99, b, int(rg.n_rings), float(rg.r_max), str(rg.weight),
                float(rg.w_min), sigma=ck["sigma"], seed=int(a.seed), device=device)
        anchor_sets[b] = entry

    for T in cfg.sweep.T_atlas:
        T = int(T)
        proc_true = make_process(pname, means_t, variance, T, device, cfg)
        row = {"T": T, "budgets": []}
        if cfg.eval.analytic:
            row["analytic"] = fate.fate_metrics(proc_true.label(None, X_te, R99), gt)
        cal = None
        for b in bs:
            st = anchor_sets[b]
            t0 = time.time()
            A = proc_true.true_backward(st["P"])
            brow = {"n_per_mode": b, "n_anchors": int(st["P"].shape[0]),
                    "secs_backtrack": round(time.time() - t0, 2), "classifiers": []}
            if a.roundtrip:
                brow["roundtrip_acc"] = float((proc_true.label(None, A, R99) == st["y"]).float().mean())
            Ar = proc_true.true_backward(st["Pr"]) if need_rings else None
            for spec in specs:
                t1 = time.time()
                if spec.get("anchors", "shell") == "rings":
                    nets = fate.train_ensemble(Ar, st["yr"], K, spec, device, w=st["wr"])
                else:
                    nets = fate.train_ensemble(A, st["y"], K, spec, device)
                if spec["arch"] == "altered_knn" and spec.get("threshold") == "auto":
                    # calibrate the confidence cut on seeds labeled by the TRUE score at this T
                    if cal is None:
                        X_cal = proc.seeds(int(cfg.anchors.n_calibrate), d, cfg.seed + 2)
                        cal = (X_cal, proc_true.label(None, X_cal, R99))
                    for net in nets:
                        net.calibrate(cal[0], cal[1])
                r = {"name": spec["name"], "arch": spec["arch"], "degree": spec["degree"],
                     "anchors": spec.get("anchors", "shell"),
                     "n_anchors": int(nets[0].X.shape[0]) if hasattr(nets[0], "X") else int(A.shape[0]),
                     "secs_train": round(time.time() - t1, 1), "fit": nets[0].describe()}
                r.update(fate.fate_metrics(fate.predict_fate(nets, X_te), gt))
                brow["classifiers"].append(r)
            row["budgets"].append(brow)
        cell["per_T"].append(row)
        best = _best_over_budgets(row, cfg.classifier.primary)
        print(f"[atlas:{pname}] d={d:>2} K={K:>2} T={T:>4} budgets={bs} "
              + (f"analytic={row['analytic']['full_acc']:.3f} " if cfg.eval.analytic else "")
              + (f"{cfg.classifier.primary}: full={best['full_acc']:.4f} modeF1={best['mode_f1']:.3f} "
                 f"hallF1={best['hall_f1']:.3f} @ {best['n_anchors']} anchors" if best else ""), flush=True)
    return cell


# ------------------------------------------------------------------ best-budget selection
def model_at(row, name, key=BEST_KEY):
    """The best-over-budgets entry for `name` in a per_T row, or None.

    `analytic` does not use anchors, so it is returned as-is with n_anchors = 0.
    """
    if name == "analytic":
        return {**row["analytic"], "n_anchors": 0, "n_per_mode": 0} if "analytic" in row else None
    best = None
    for brow in row.get("budgets", []):
        m = next((m for m in brow["classifiers"] if m["name"] == name), None)
        if m is None:
            continue
        cand = {**m, "n_per_mode": brow["n_per_mode"]}
        if best is None or cand[key] > best[key]:
            best = cand
    return best


def _best_over_budgets(row, name):
    return model_at(row, name)


def model_names(cells):
    names = []
    for c in cells:
        for r in c["per_T"]:
            for b in r.get("budgets", []):
                for m in b["classifiers"]:
                    if m["name"] not in names:
                        names.append(m["name"])
    return names


# ------------------------------------------------------------------ writing
def write_results(cfg, cells, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {"process": cfg.process, "kind": "atlas",
            "sweep": OmegaConf.to_container(cfg.sweep, resolve=True),
            "anchors": OmegaConf.to_container(cfg.anchors, resolve=True),
            "classifier": OmegaConf.to_container(cfg.classifier, resolve=True)}
    with open(os.path.join(out_dir, "atlas_results.json"), "w") as f:
        json.dump({**meta, "cells": cells}, f, indent=2)

    Ts = sorted({r["T"] for c in cells for r in c["per_T"]})
    names = model_names(cells)
    families = (["analytic"] if cfg.eval.analytic else []) + names
    header = ("process,d,K,T_train,hall_gt,T,model,arch,n_per_mode,n_anchors,roundtrip_acc,"
              + ",".join(fate.METRICS) + "\n")
    summary = {}
    for T in Ts:
        tdir = os.path.join(out_dir, f"T_{T}")
        os.makedirs(tdir, exist_ok=True)
        rows = []
        for c in cells:
            row = next((r for r in c["per_T"] if r["T"] == T), None)
            if row is None:
                continue
            pre = f"{cfg.process},{c['d']},{c['K']},{c['T_train']},{c['hall_gt']:.4f},{T}"
            if "analytic" in row:
                rows.append(f"{pre},analytic,analytic,0,0,nan,"
                            + ",".join(f"{row['analytic'][k]:.4f}" for k in fate.METRICS))
            for brow in row.get("budgets", []):
                rt = brow.get("roundtrip_acc", float("nan"))
                for m in brow["classifiers"]:
                    rows.append(f"{pre},{m['name']},{m['arch']},{brow['n_per_mode']},{m['n_anchors']},"
                                f"{rt:.4f}," + ",".join(f"{m[k]:.4f}" for k in fate.METRICS))
            # summary uses the best budget per cell
            for fam in families:
                best = model_at(row, fam)
                if best:
                    summary.setdefault((T, fam), []).append(best)
        with open(os.path.join(tdir, "results.csv"), "w") as f:
            f.write(header + "\n".join(rows) + "\n")

    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("T,model,n_cells," + ",".join(f"mean_{k}" for k in fate.METRICS)
                + ",mean_n_anchors\n")
        for (T, name), ms in sorted(summary.items()):
            f.write(f"{T},{name},{len(ms)},"
                    + ",".join(f"{sum(m[k] for m in ms) / len(ms):.4f}" for k in fate.METRICS)
                    + f",{sum(m['n_anchors'] for m in ms) / len(ms):.0f}\n")
    print(f"[atlas:{cfg.process}] wrote {out_dir}/atlas_results.json, summary.csv and "
          f"T_*/results.csv ({len(cells)} cells, {len(Ts)} T values, {len(budgets(cfg))} budgets)")
    return {"dir": out_dir, "n_cells": len(cells), "T": Ts}


def run(cfg):
    device = utils.get_device(cfg.device)
    cells = []
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            try:
                cells.append(atlas_one(cfg, int(d), int(K), device))
            except RuntimeError as e:
                print(f"[atlas:{cfg.process}] d={d} K={K} skipped: {e}")
    return write_results(cfg, cells, run_dir(cfg))


def merge(cfg):
    """Combine output/_parts/<process>/*/<process>/atlas_results.json (scripts/run_parallel.sh)."""
    import glob
    parts = sorted(glob.glob(os.path.join(cfg.paths.output, "_parts", cfg.process, "*",
                                          cfg.process, "atlas_results.json")))
    if not parts:
        raise FileNotFoundError(f"no atlas parts under {cfg.paths.output}/_parts/{cfg.process}/*/")
    cells = []
    for p in parts:
        with open(p) as f:
            blob = json.load(f)
        if blob.get("kind") != "atlas" or blob.get("process") != cfg.process:
            raise ValueError(f"{p} is not an atlas results file (stale _parts?)")
        cells += blob["cells"]
    cells.sort(key=lambda c: (c["d"], c["K"]))
    print(f"[atlas_merge:{cfg.process}] {len(parts)} parts -> {len(cells)} cells")
    return write_results(cfg, cells, run_dir(cfg))
