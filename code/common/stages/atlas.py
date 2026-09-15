"""stages/atlas.py — the ring atlas: true-score backtrack -> seed-fate predictor, swept over T.

Per (d, K):
  1. learned sampler at sweep.T_train (trained on demand until its hallucination rate is
     <= train.hall_target; cached in checkpoints/ unless train.force_retrain)
  2. ground truth: eval.n_eval seeds pushed FORWARD through the learned sampler, labeled by fate
  3. for every T in sweep.T_atlas:
       - data-space anchors on rings around each mode (gmm.ring_anchors: uniform in the R99
         ball -> label k; shell R99..(1+w)R99 -> hallucination)
       - backtracked to seed space with the TRUE score at T steps (process.true_backward)
       - every classifier in classifier.models is fit on the (seed, label) anchors and scored
         on the learned-sampler ground truth
       - optionally the analytic forward pass at T is scored as well (eval.analytic)

Writes output/<run_tag>/<process>/T_<T>/results.{json,csv} (one folder per T),
T_<T>/tables.{tex,txt} (one (K x d) grid per predictor family: overall acc / mode F1 / hall F1),
all_tables.{tex,txt} (every T), summary.csv (all T, all models, mean over cells), and the anchor
sets themselves: anchors/d{d}_K{K}_T{T}.npz with the data-space points `P`, labels `y`, the
backtracked seed-space points `A` and each anchor's round-trip fate `y_roundtrip`.
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


def run_dir(cfg):
    return os.path.join(cfg.paths.output, cfg.run_tag, cfg.process)


def anchors_dir(cfg):
    return os.path.join(run_dir(cfg), "anchors")


def atlas_one(cfg, d, K, device):
    pname, T_train = cfg.process, int(cfg.sweep.T_train)
    os.makedirs(anchors_dir(cfg), exist_ok=True)
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
    P, yA = gmm.ring_anchors(means_t, R99, int(a.n_per_mode), float(a.shell_w),
                             float(a.shell_frac), int(a.seed), device)
    need_rings = any(s.get("anchors", "shell") == "rings" for s in specs)
    if need_rings:
        rg = a.rings
        Pr, yr, wr = gmm.ring_anchors_weighted(means_t, R99, int(a.n_per_mode), int(rg.n_rings), float(rg.r_max),
                                               str(rg.weight), float(rg.w_min), sigma=ck["sigma"],
                                               seed=int(a.seed), device=device)
    for T in cfg.sweep.T_atlas:
        T = int(T)
        proc_true = make_process(pname, means_t, variance, T, device, cfg)
        t0 = time.time()
        A = proc_true.true_backward(P)
        row = {"T": T, "n_anchors": int(P.shape[0]), "secs_backtrack": round(time.time() - t0, 1),
               "classifiers": []}
        y_rt = proc_true.label(None, A, R99)
        if a.roundtrip:
            row["roundtrip_acc"] = float((y_rt == yA).float().mean())
        np.savez_compressed(os.path.join(anchors_dir(cfg), f"d{d}_K{K}_T{T}.npz"),
                            P=P.cpu().numpy(), y=yA.cpu().numpy(), A=A.cpu().numpy(),
                            y_roundtrip=y_rt.cpu().numpy(), means=means_t.cpu().numpy(), R99=R99,
                            n_per_mode=int(a.n_per_mode), shell_frac=float(a.shell_frac), shell_w=float(a.shell_w))
        if need_rings:
            Ar = proc_true.true_backward(Pr)
            np.savez_compressed(os.path.join(anchors_dir(cfg), f"d{d}_K{K}_T{T}_rings.npz"),
                                P=Pr.cpu().numpy(), y=yr.cpu().numpy(), w=wr.cpu().numpy(), A=Ar.cpu().numpy(),
                                y_roundtrip=proc_true.label(None, Ar, R99).cpu().numpy(), means=means_t.cpu().numpy(),
                                R99=R99, n_rings=int(rg.n_rings), r_max=float(rg.r_max), weight=str(rg.weight))
        if cfg.eval.analytic:
            row["analytic"] = fate.fate_metrics(proc_true.label(None, X_te, R99), gt)
        for spec in specs:
            t1 = time.time()
            if spec.get("anchors", "shell") == "rings":
                nets = fate.train_ensemble(Ar, yr, K, spec, device, w=wr)
            else:
                nets = fate.train_ensemble(A, yA, K, spec, device)
            if spec["arch"] == "altered_knn" and spec.get("threshold") == "auto":
                # calibrate the confidence cut on seeds labeled by the TRUE score at this T (no learned info)
                if "cal" not in locals() or cal[0] != T:
                    X_cal = proc.seeds(int(cfg.anchors.n_calibrate), d, cfg.seed + 2)
                    cal = (T, X_cal, proc_true.label(None, X_cal, R99))
                for net in nets:
                    net.calibrate(cal[1], cal[2])
            r = {"name": spec["name"], "arch": spec["arch"], "degree": spec["degree"],
                 "anchors": spec.get("anchors", "shell"),
                 "n_anchors": int(nets[0].X.shape[0]) if hasattr(nets[0], "X") else int(A.shape[0]),
                 "secs_train": round(time.time() - t1, 1), "fit": nets[0].describe()}
            r.update(fate.fate_metrics(fate.predict_fate(nets, X_te), gt))
            row["classifiers"].append(r)
        cell["per_T"].append(row)
        prim = next((c for c in row["classifiers"] if c["name"] == cfg.classifier.primary), row["classifiers"][0])
        print(f"[atlas:{pname}] d={d:>2} K={K:>2} T={T:>4} anchors={P.shape[0]:>5} "
              + (f"roundtrip={row['roundtrip_acc']:.3f} " if a.roundtrip else "")
              + (f"analytic={row['analytic']['full_acc']:.3f} " if cfg.eval.analytic else "")
              + f"{prim['name']}: full={prim['full_acc']:.4f} modeF1={prim['mode_f1']:.3f} "
              f"hallF1={prim['hall_f1']:.3f}", flush=True)
    return cell


def _grid_tables(cfg, cells, T, families, fmt):
    """One (K rows x d cols) table per family for a given T; cells are acc/modeF1/hallF1 in %."""
    ds = sorted({c["d"] for c in cells}); Ks = sorted({c["K"] for c in cells})
    a = cfg.anchors
    n_pm = int(a.n_per_mode); n_sh = int(round(float(a.shell_frac) * n_pm))
    dk = {(c["d"], c["K"]): next((r for r in c["per_T"] if r["T"] == T), None) for c in cells}

    def cell(d, K, fam):
        r = dk.get((d, K))
        if r is None:
            return None
        return r.get("analytic") if fam == "analytic" else next((m for m in r["classifiers"] if m["name"] == fam), None)

    def fmt3(v):
        return f"{100 * v['full_acc']:.1f} / {100 * v['mode_f1']:.1f} / {100 * v['hall_f1']:.1f}" if v else "--"

    out = []
    for fam in families:
        vals = {(d, K): cell(d, K, fam) for d in ds for K in Ks}
        got = [v for v in vals.values() if v]
        if not got:
            continue
        mean = {k: 100 * sum(v[k] for v in got) / len(got) for k in ("full_acc", "mode_f1", "hall_f1")}
        title = (f"{cfg.process}, true-score T={T}, predictor {fam}: overall acc / mode-basin F1 / "
                 f"hallucination F1 (%), anchors per mode {n_pm} disk + {n_sh} shell; "
                 f"mean {mean['full_acc']:.1f} / {mean['mode_f1']:.1f} / {mean['hall_f1']:.1f}")
        if fmt == "tex":
            L = ["\\begin{table}[t]", "\\centering", "\\caption{" + title.replace("_", "\\_") + "}",
                 f"\\label{{tab:atlas-{cfg.process}-T{T}-{fam}}}",
                 "\\begin{tabular}{l " + "r" * len(ds) + "}", "\\toprule",
                 "$K \\backslash d$ & " + " & ".join(f"${d}$" for d in ds) + " \\\\", "\\midrule"]
            for K in Ks:
                L.append(f"{K} & " + " & ".join(fmt3(vals[(d, K)]) for d in ds) + " \\\\")
            L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        else:
            w = 20
            L = [title, "K \\ d".ljust(6) + "".join(str(d).rjust(w) for d in ds), "-" * (6 + w * len(ds))]
            for K in Ks:
                L.append(str(K).ljust(6) + "".join(fmt3(vals[(d, K)]).replace(" ", "").rjust(w) for d in ds))
            L.append("")
        out.append("\n".join(L))
    return "\n".join(out)


def write_tables(cfg, cells, out_dir):
    Ts = sorted({r["T"] for c in cells for r in c["per_T"]})
    families = []
    for c in cells:
        for r in c["per_T"]:
            for m in r["classifiers"]:
                if m["name"] not in families:
                    families.append(m["name"])
    if cfg.eval.analytic:
        families = ["analytic"] + families
    head_tex = ("% Auto-generated by common/stages/atlas.py -- requires \\usepackage{booktabs}\n"
                "% cell = overall accuracy / mode-basin F1 / hallucination F1, in %\n\n")
    head_txt = "cell = overall accuracy / mode-basin F1 / hallucination F1 (%)   rows K, columns d\n\n"
    all_tex, all_txt = [head_tex], [head_txt]
    for T in Ts:
        tdir = os.path.join(out_dir, f"T_{T}")
        os.makedirs(tdir, exist_ok=True)
        tex = _grid_tables(cfg, cells, T, families, "tex")
        txt = _grid_tables(cfg, cells, T, families, "txt")
        with open(os.path.join(tdir, "tables.tex"), "w") as f:
            f.write(head_tex + tex)
        with open(os.path.join(tdir, "tables.txt"), "w") as f:
            f.write(head_txt + txt)
        all_tex.append(f"% ===== T = {T} =====\n" + tex)
        all_txt.append(f"===== T = {T} =====\n\n" + txt)
    with open(os.path.join(out_dir, "all_tables.tex"), "w") as f:
        f.write("\n".join(all_tex))
    with open(os.path.join(out_dir, "all_tables.txt"), "w") as f:
        f.write("\n".join(all_txt))


def write_results(cfg, cells, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {"process": cfg.process, "run_tag": cfg.run_tag, "kind": "atlas",
            "sweep": OmegaConf.to_container(cfg.sweep, resolve=True),
            "anchors": OmegaConf.to_container(cfg.anchors, resolve=True),
            "classifier": OmegaConf.to_container(cfg.classifier, resolve=True)}
    with open(os.path.join(out_dir, "atlas_results.json"), "w") as f:
        json.dump({**meta, "cells": cells}, f, indent=2)
    Ts = sorted({r["T"] for c in cells for r in c["per_T"]})
    header = "process,d,K,T_train,hall_gt,T,n_anchors,roundtrip_acc,model,arch," + ",".join(fate.METRICS) + "\n"
    summary = {}
    for T in Ts:
        tdir = os.path.join(out_dir, f"T_{T}")
        os.makedirs(tdir, exist_ok=True)
        rows = []
        for c in cells:
            for r in c["per_T"]:
                if r["T"] != T:
                    continue
                pre = (f"{cfg.process},{c['d']},{c['K']},{c['T_train']},{c['hall_gt']:.4f},{T},"
                       f"{r['n_anchors']},{r.get('roundtrip_acc', float('nan')):.4f}")
                if "analytic" in r:
                    rows.append(f"{pre},analytic,analytic," + ",".join(f"{r['analytic'][k]:.4f}" for k in fate.METRICS))
                    summary.setdefault((T, "analytic"), []).append(r["analytic"])
                for m in r["classifiers"]:
                    rows.append(f"{pre},{m['name']},{m['arch']}," + ",".join(f"{m[k]:.4f}" for k in fate.METRICS))
                    summary.setdefault((T, m["name"]), []).append(m)
        with open(os.path.join(tdir, "results.csv"), "w") as f:
            f.write(header + "\n".join(rows) + "\n")
        with open(os.path.join(tdir, "results.json"), "w") as f:
            json.dump({**meta, "T": T,
                       "cells": [{**{k: v for k, v in c.items() if k != "per_T"},
                                  **next(r for r in c["per_T"] if r["T"] == T)} for c in cells
                                 if any(r["T"] == T for r in c["per_T"])]}, f, indent=2)
    n_per_mode = int(cfg.anchors.n_per_mode); n_shell = int(round(float(cfg.anchors.shell_frac) * n_per_mode))
    with open(os.path.join(out_dir, "anchors.json"), "w") as f:
        json.dump({"geometry": "per mode: n_per_mode points uniform in the R99 ball (label = mode) + "
                               "n_shell points uniform in radius over R99..(1+shell_w) R99 (label = -1, "
                               "hallucination); backtracked to seed space with the true score at T steps",
                   "n_per_mode": n_per_mode, "n_shell_per_mode": n_shell, "shell_w": float(cfg.anchors.shell_w),
                   "n_anchors_by_K": {str(K): K * (n_per_mode + n_shell) for K in sorted({c["K"] for c in cells})},
                   "files": "anchors/d{d}_K{K}_T{T}.npz : P (data space), y, A (seed space), y_roundtrip",
                   "roundtrip_acc": {f"d{c['d']}_K{c['K']}_T{r['T']}": r.get("roundtrip_acc")
                                     for c in cells for r in c["per_T"]}}, f, indent=2)
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("T,model,n_cells," + ",".join(f"mean_{k}" for k in fate.METRICS) + "\n")
        for (T, name), ms in sorted(summary.items()):
            f.write(f"{T},{name},{len(ms)}," + ",".join(f"{sum(m[k] for m in ms) / len(ms):.4f}" for k in fate.METRICS) + "\n")
    write_tables(cfg, cells, out_dir)
    print(f"[atlas:{cfg.process}] wrote {out_dir}/T_*/{{results.json,results.csv,tables.tex,tables.txt}}, "
          f"all_tables.{{tex,txt}} and summary.csv ({len(cells)} cells, {len(Ts)} T values)")
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
    """Combine output/_parts/<process>/*/<run_tag>/<process>/atlas_results.json (run_parallel.sh)."""
    import glob
    parts = sorted(glob.glob(os.path.join(cfg.paths.output, "_parts", cfg.process, "*",
                                          cfg.run_tag, cfg.process, "atlas_results.json")))
    if not parts:
        raise FileNotFoundError(f"no atlas parts under {cfg.paths.output}/_parts/{cfg.process}/*/{cfg.run_tag}/")
    import shutil
    cells = []
    os.makedirs(anchors_dir(cfg), exist_ok=True)
    for p in parts:
        with open(p) as f:
            blob = json.load(f)
        if blob.get("kind") != "atlas" or blob.get("process") != cfg.process:
            raise ValueError(f"{p} is not an atlas results file (stale _parts?)")
        cells += blob["cells"]
        for npz in glob.glob(os.path.join(os.path.dirname(p), "anchors", "*.npz")):
            shutil.copy2(npz, anchors_dir(cfg))
    cells.sort(key=lambda c: (c["d"], c["K"]))
    print(f"[atlas_merge:{cfg.process}] {len(parts)} parts -> {len(cells)} cells")
    return write_results(cfg, cells, run_dir(cfg))
