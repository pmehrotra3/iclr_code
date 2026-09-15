"""stages/evaluate.py — predict a seed's fate from the seed alone, score against the truth.

Ground truth (eval.labels): the fate of held-out seeds under the LEARNED sampler (default),
or under the ANALYTIC-field sampler (eval.labels=true, the exact-score control).

Predictors:
  analytic     : push the seed through the analytic-field sampler and read off its fate.
                 No training. Can only predict hallucinations the exact sampler makes.
  classifiers  : every entry of cfg.classifier.models (see common/conf/classifier/*.yaml),
                 each trained on `n` seeds labeled by the ground-truth sampler for every
                 budget n in cfg.sweep.budgets. Different `arch`es restrict the boundary
                 geometry (common/fate.py); this is the capacity ladder.

Writes output/<process>/results[_<tag>].{json,csv}.
"""
from __future__ import annotations
import os
import json
import time
import torch
from omegaconf import OmegaConf

from common import fate, checkpoint, utils
from common.process import make_process


def _spec(cfg, m) -> dict:
    """Merge a classifier entry with the preset's shared defaults into a plain dict."""
    base = OmegaConf.to_container(cfg.classifier.get("_shared", {}), resolve=True) or {}
    spec = dict(base); spec.update(OmegaConf.to_container(m, resolve=True))
    return spec


@torch.no_grad()
def _label(proc, model, R99, d, n, seed, chunk=400000):
    """Draw n seeds and label each by running the ground-truth sampler once."""
    Xs, ys = [], []
    for j in range(0, n, chunk):
        X = proc.seeds(min(chunk, n - j), d, seed + j)
        ys.append(proc.label(model, X, R99))
        Xs.append(X)
    return torch.cat(Xs), torch.cat(ys)


def eval_one(cfg, d, K, device):
    pname = cfg.process
    path = utils.ckpt_path(cfg.paths.checkpoints, pname, d, K, int(cfg.sweep.T_train))
    if not os.path.exists(path):
        return None
    model, ck = checkpoint.load(path, device)
    means_t, R99, variance = ck["means"], ck["R99"], ck["variance"]
    proc = make_process(pname, means_t, variance, ck["T"], device, cfg)
    proc_true = make_process(pname, means_t, variance, cfg.sweep.T_true, device, cfg)
    truth_model = None if cfg.eval.labels == "true" else model     # None -> analytic sampler
    truth_proc = proc_true if cfg.eval.labels == "true" else proc

    X_te, gt = _label(truth_proc, truth_model, R99, d, cfg.eval.n_eval, cfg.seed + 1)
    row = {"d": d, "K": K, "labels": cfg.eval.labels,
           "hall_gt": float((gt == -1).float().mean()),
           "n_mode": int((gt >= 0).sum()), "n_hall": int((gt == -1).sum())}

    if cfg.eval.analytic:
        row["analytic"] = fate.fate_metrics(proc_true.label(None, X_te, R99), gt)

    row["classifiers"] = []
    for n_train in cfg.sweep.budgets:
        t0 = time.time()
        X_tr, y_tr = _label(truth_proc, truth_model, R99, d, int(n_train), cfg.seed + cfg.eval.seed_offset)
        t_label = time.time() - t0
        for m in cfg.classifier.models:
            spec = _spec(cfg, m)
            t1 = time.time()
            nets = fate.train_ensemble(X_tr, y_tr, K, spec, device)
            r = {"name": spec["name"], "arch": spec["arch"], "degree": spec["degree"],
                 "n_train": int(n_train), "secs_label": round(t_label, 1),
                 "secs_train": round(time.time() - t1, 1), "fit": nets[0].describe()}
            r.update(fate.fate_metrics(fate.predict_fate(nets, X_te), gt))
            row["classifiers"].append(r)
            print(f"[eval:{pname}]    d={d:>2} K={K:>2} n={int(n_train):>8} {spec['name']:>10}: "
                  f"full={r['full_acc']:.4f} mode={r['mode_acc']:.4f} hallF1={r['hall_f1']:.3f} "
                  f"({r['secs_train']:.0f}s)", flush=True)
            del nets
        del X_tr, y_tr
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # best budget per model, by full accuracy
    row["best"] = {}
    for r in row["classifiers"]:
        b = row["best"].get(r["name"])
        if b is None or r["full_acc"] > b["full_acc"]:
            row["best"][r["name"]] = r
    return row


def write_results(cfg, results, out_dir):
    """results.json (nested) + results.csv (flat). Shared with stages/merge.py."""
    os.makedirs(out_dir, exist_ok=True)
    tag = cfg.eval.tag
    js = utils.results_path(cfg.paths.output, cfg.process, tag, "json")
    js = os.path.join(out_dir, os.path.basename(js))
    with open(js, "w") as f:
        json.dump({"process": cfg.process, "labels": cfg.eval.labels,
                   "sweep": OmegaConf.to_container(cfg.sweep, resolve=True),
                   "classifier": OmegaConf.to_container(cfg.classifier, resolve=True),
                   "results": results}, f, indent=2)
    csv = js[:-5] + ".csv"
    with open(csv, "w") as f:
        f.write("process,labels,d,K,hall_gt,model,arch,n_train," + ",".join(fate.METRICS) + "\n")
        for r in results:
            pre = f"{cfg.process},{r['labels']},{r['d']},{r['K']},{r['hall_gt']:.4f}"
            if "analytic" in r:
                f.write(f"{pre},analytic,analytic,0," + ",".join(f"{r['analytic'][c]:.4f}" for c in fate.METRICS) + "\n")
            for a in r["classifiers"]:
                f.write(f"{pre},{a['name']},{a['arch']},{a['n_train']},"
                        + ",".join(f"{a[c]:.4f}" for c in fate.METRICS) + "\n")
    print(f"[eval:{cfg.process}] wrote {js} and {csv}")
    return {"json": js, "csv": csv, "n_results": len(results)}


def run(cfg):
    device = utils.get_device(cfg.device)
    pname = cfg.process
    out_dir = utils.process_dir(cfg.paths.output, pname)
    results = []
    for d in cfg.sweep.d:
        for K in cfg.sweep.K:
            r = eval_one(cfg, int(d), int(K), device)
            if r is None:
                print(f"[eval:{pname}] d={d:>2} K={K:>2} -> no checkpoint, skipped")
                continue
            results.append(r)
            prim = r["best"].get(cfg.classifier.primary)
            an = f"analytic={r['analytic']['full_acc']:.3f} " if "analytic" in r else ""
            print(f"[eval:{pname}] d={d:>2} K={K:>2} hall_gt={r['hall_gt']:.3f} {an}"
                  + (f"{cfg.classifier.primary}={prim['full_acc']:.4f} (n={prim['n_train']}) "
                     f"hallF1={prim['hall_f1']:.3f}" if prim else ""), flush=True)
    return write_results(cfg, results, out_dir)
