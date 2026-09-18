"""stages/atlas.py — the seed-fate atlas: plant in data space, backtrack with the exact score,
fit predictors on the labelled seed picture, score them against the learned sampler.

Per (d, K) cell:
  1. Plant labelled anchors in DATA space (no model): per mode, points filling the R99 ball
     plus a thin band R99 .. R99 + anchors.shell_sigma * sigma just outside it (the planted
     hypothesis that hallucinations live in that band). Every anchor is coloured with the
     ground-truth rule L (nearest mode within R99, else hallucination). gmm.ball_anchors.
  2. Backtrack every anchor to SEED space with the exact score (proc.true_backward, a second-
     order integrator so the pass is the inverse of the forward pass), carrying its colour.
     Repeated for every T in sweep.T_atlas and every anchor budget in anchors.budgets.
  3. Fit every predictor in classifier.models on the (seed, colour) pairs -- and nothing else.
     Parametric families are also reported prior-calibrated (<name>_cal): the hallucination
     cut is shifted so they call as many seeds hallucinations as the exact score does at that
     T. altered_knn (mode-only anchors, low-confidence vote = hallucination) is reported
     separately and takes its own weighted spheres (gmm.altered_knn_anchors).
  4. Ground truth: eval.n_eval fresh Gaussian seeds pushed through the LEARNED sampler
     (checkpoints/<process>/model_d{d}_K{K}_T{T_train}.pt, trained on demand by stages/train.py)
     and labelled with L. This is the only place the trained network is used.
  5. Score every predictor on those seeds; alongside, the analytic control: the same seeds
     pushed through the exact score at the atlas T and at T_train. The control is the ceiling
     -- how much of the learned sampler the exact score explains at all.
  Diagnostics per cell: roundtrip_acc (anchors that return to their colour under the exact
  forward pass; 1.0 with the Heun integrators) and hall_dist (where the learned sampler's
  hallucinations actually land, in sigma beyond R99 -- tests the band hypothesis).

Writes, under output/<run_tag>/<process>/ (run_tag "" -> output/<process>/):
  atlas_results.json   every cell, every T, every budget (the figures read this)
  summary.csv          mean over cells of each metric, per T and model, at the best budget
  T_<T>/results.csv    flat per-cell rows for that T, one line per (model, budget)
"""
from __future__ import annotations
import os
import json
import time
import torch
from omegaconf import OmegaConf

from common import fate, gmm, checkpoint, utils
from common.process import make_process
from common.stages.train import train_one

BEST_KEY = "full_acc"          # which metric picks the winning budget in summary.csv


def run_dir(cfg):
    """output/<run_tag>/<process> (results and figures); output/<process> when run_tag is empty."""
    tag = str(cfg.run_tag or "")
    return os.path.join(cfg.paths.output, tag, cfg.process) if tag else os.path.join(cfg.paths.output, cfg.process)


def _spec(cfg, m) -> dict:
    """Merge a classifier entry with the preset's shared defaults into a plain dict."""
    base = OmegaConf.to_container(cfg.classifier.get("_shared", {}), resolve=True) or {}
    spec = dict(base); spec.update(OmegaConf.to_container(m, resolve=True))
    return spec


def per_mode(cfg, b, K):
    """Anchor budget b -> mode-ball anchors per mode. anchors.unit = per_mode (b itself) or total
    (b split over the K modes, counting the band anchors)."""
    if str(cfg.anchors.get("unit", "total")) == "per_mode":
        return int(b)
    return max(1, int(round(b / (K * (1.0 + float(cfg.anchors.shell_frac))))))


def budgets(cfg):
    b = cfg.anchors.budgets
    return [int(x) for x in ([b] if isinstance(b, (int, float, str)) else b)]


@torch.no_grad()
def _label_endpoints(proc, model, R99, d, n, seed, chunk=400000):
    """Draw n seeds, push them through the learned sampler once, label with L; also return the
    data-space endpoints (for the hallucination-distance diagnostic)."""
    Xs, ys, Xfs = [], [], []
    for j in range(0, n, chunk):
        X = proc.seeds(min(chunk, n - j), d, seed + j)
        Xf = proc.sample(model, X)
        Xs.append(X); Xfs.append(Xf); ys.append(gmm.label_fate(Xf, proc.means_t, R99))
    return torch.cat(Xs), torch.cat(ys), torch.cat(Xfs)


@torch.no_grad()
def hall_distance(Xh, means_t, R99, sigma):
    """Where the learned sampler's hallucinations land: distance to the nearest mode centre beyond
    R99, in units of sigma. Tests the planted assumption that hallucinations live in the band
    R99 .. R99 + anchors.shell_sigma * sigma."""
    if Xh.shape[0] == 0:
        return {"n": 0}
    excess = (torch.cdist(Xh, means_t).min(1).values - R99) / sigma
    q = torch.quantile(excess, torch.tensor([0.25, 0.5, 0.75, 0.9], device=excess.device)).tolist()
    return {"n": int(Xh.shape[0]),
            "frac_within_2sigma": float((excess <= 2).float().mean()),
            "frac_2_to_4sigma": float(((excess > 2) & (excess <= 4)).float().mean()),
            "frac_beyond_4sigma": float((excess > 4).float().mean()),
            "median_sigma_beyond_R99": q[1], "quartiles_sigma_beyond_R99": q}


def atlas_one(cfg, d, K, device):
    pname, T_train = cfg.process, int(cfg.sweep.T_train)
    a, sigma = cfg.anchors, float(cfg.data.sigma)
    info = train_one(cfg, d, K, device)                      # cached unless stale / force_retrain
    model, ck = checkpoint.load(info["path"], device)
    means_t, R99, variance = ck["means"], ck["R99"], ck["variance"]
    proc = make_process(pname, means_t, variance, T_train, device, cfg)

    # -- ground truth: the learned sampler's real behaviour (step 4)
    X_te, gt, Xf = _label_endpoints(proc, model, R99, d, cfg.eval.n_eval, cfg.seed + 1)
    cell = {"d": d, "K": K, "T_train": T_train, "hall_learned": float(ck["hall_rate"]),
            "hall_gt": float((gt == -1).float().mean()), "n_eval": int(cfg.eval.n_eval),
            "n_mode": int((gt >= 0).sum()), "n_hall": int((gt == -1).sum()), "per_T": [],
            "hall_dist": hall_distance(Xf[gt == -1], means_t, R99, sigma),
            # the control at the sampler's own step count: model error alone, no step mismatch
            "analytic_T_train": fate.fate_metrics(proc.label(None, X_te, R99), gt)}
    hd = cell["hall_dist"]
    print(f"[atlas:{pname}] d={d:>2} K={K:>2} T_train={T_train} hall_gt={cell['hall_gt']:.4f} "
          f"analytic@T_train={cell['analytic_T_train']['full_acc']:.4f} | learned hallucinations within "
          f"{float(a.shell_sigma):g}sigma of R99: {hd.get('frac_within_2sigma', float('nan')):.2f} "
          f"(median {hd.get('median_sigma_beyond_R99', float('nan')):.1f} sigma beyond R99)", flush=True)

    # -- step 1: plant anchors in data space, once per budget (they do not depend on T)
    specs = [_spec(cfg, m) for m in cfg.classifier.models]
    bs = budgets(cfg)
    need_altered = any(sp["arch"] == "altered_knn" for sp in specs)
    rg = a.altered_knn
    planted = {}
    for b in bs:
        n = per_mode(cfg, b, K)
        P, y = gmm.ball_anchors(means_t, R99, n, float(a.shell_frac), float(a.shell_sigma), sigma,
                                int(a.seed), device)
        entry = {"P": P, "y": y}
        if need_altered:
            entry["Pa"], entry["ya"], entry["wa"] = gmm.altered_knn_anchors(
                means_t, R99, n, int(rg.n_rings), float(rg.r_max), str(rg.weight), float(rg.w_min),
                sigma=sigma, seed=int(a.seed), device=device)
        planted[b] = entry

    prior = cfg.classifier.get("calibrate_prior", None)
    for T in cfg.sweep.T_atlas:
        T = int(T)
        proc_true = make_process(pname, means_t, variance, T, device, cfg)
        row = {"T": T, "analytic": fate.fate_metrics(proc_true.label(None, X_te, R99), gt), "budgets": []}
        # seeds labelled by the exact score at this T: calibrate altered_knn's cut and the priors
        X_cal = proc.seeds(int(a.n_calibrate), d, cfg.seed + 2)
        y_cal = proc_true.label(None, X_cal, R99)
        rate = float((y_cal == -1).float().mean()) if str(prior) == "true_score" else (float(prior) if prior else None)
        for b in bs:
            st = planted[b]
            t0 = time.time()
            A = proc_true.true_backward(st["P"])                      # step 2
            Aa = proc_true.true_backward(st["Pa"]) if need_altered else None
            brow = {"budget": b, "n_per_mode": per_mode(cfg, b, K), "n_anchors": int(A.shape[0]),
                    "secs_backtrack": round(time.time() - t0, 2),
                    "roundtrip_acc": float((proc_true.label(None, A, R99) == st["y"]).float().mean()),
                    "classifiers": []}
            for spec in specs:                                        # step 3
                t1 = time.time()
                altered = spec["arch"] == "altered_knn"
                nets = (fate.train_ensemble(Aa, st["ya"], K, spec, device, w=st["wa"]) if altered
                        else fate.train_ensemble(A, st["y"], K, spec, device))
                if altered and spec.get("threshold") == "auto":
                    for net in nets:
                        net.calibrate(X_cal, y_cal)
                r = {"name": spec["name"], "arch": spec["arch"], "degree": spec["degree"],
                     "n_anchors": int(nets[0].X.shape[0]) if hasattr(nets[0], "X") else int(A.shape[0]),
                     "secs_train": round(time.time() - t1, 1), "fit": nets[0].describe()}
                r.update(fate.fate_metrics(fate.predict_fate(nets, X_te), gt))   # step 5
                brow["classifiers"].append(r)
                if rate is not None and spec["arch"] in fate.PARAMETRIC:
                    bias = fate.hall_bias_for_rate(nets, X_cal, rate)
                    rc = {**r, "name": spec["name"] + "_cal", "hall_bias": bias, "target_hall_rate": rate}
                    rc.update(fate.fate_metrics(fate.predict_fate(nets, X_te, hall_bias=bias), gt))
                    brow["classifiers"].append(rc)
            row["budgets"].append(brow)
        cell["per_T"].append(row)
        best = model_at(row, cfg.classifier.primary)
        print(f"[atlas:{pname}] d={d:>2} K={K:>2} T={T:>4} budgets={bs} analytic={row['analytic']['full_acc']:.4f} "
              + (f"{cfg.classifier.primary}: full={best['full_acc']:.4f} modeF1={best['mode_f1']:.3f} "
                 f"hallF1={best['hall_f1']:.3f} @ {best['n_anchors']} anchors "
                 f"(roundtrip {max(bb['roundtrip_acc'] for bb in row['budgets']):.4f})" if best else ""), flush=True)
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
    families = ["analytic"] + names
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
            if "analytic_T_train" in c:
                rows.append(f"{pre},analytic_T_train,analytic,0,0,nan,"
                            + ",".join(f"{c['analytic_T_train'][k]:.4f}" for k in fate.METRICS))
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
            if "analytic_T_train" in c:          # same for every T; listed so summary.csv shows it
                summary.setdefault((T, "analytic_T_train"), []).append({**c["analytic_T_train"], "n_anchors": 0})
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
    """Combine output/_parts/<process>/*/<run_tag>/<process>/atlas_results.json (scripts/run.sh shards by d)."""
    import glob
    rel = os.path.relpath(run_dir(cfg), cfg.paths.output)          # <run_tag>/<process>
    parts = sorted(glob.glob(os.path.join(cfg.paths.output, "_parts", cfg.process, "*", rel,
                                          "atlas_results.json")))
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
