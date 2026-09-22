"""stages/hall_bench.py — how many ground-truth hallucinations does each method remove? (DDIM)

Checkpoints come from checkpoints/<process>/ like every other stage, and are sampled with the
schedule stored in the checkpoint (use_ckpt_schedule), so Pranav's checkpoints can simply be
dropped into that folder. His gt_cache labels are used as the ground truth after an agreement
check (bench.backend = pranav instead imports his code directly; not needed).

For every cell of bench.d x bench.K:

  1. Ground truth. Take the seeds whose ground-truth fate is "hallucinated" (label -1):
       bench.gt_source = cache   : read bench.gt_fmt (Pranav's gt_cache), which must hold the
                                   seeds and either their labels or their endpoints;
                         learned : run the learned sampler on bench.n seeds and label the
                                   endpoints (nearest mode within R99, else -1);
                         true    : the same with the exact-GMM field.
     If the cache is missing or unreadable, it falls back to `learned` and says so.
  2. Apply four repairs to exactly those seeds, each with the learned sampler, and count how
     many still land outside every core:
       ours     : move the seed by a fixed eps along -n, n = J^T nu / ||J^T nu|| (Prop. 4, one
                  VJP through the whole sampler), nu toward the nearest mode of its endpoint;
       IQ       : score s - lam grad E, E = DSM loss at t0 of the Tweedie estimate, applied only
                  for t <= bench.iq_window ("for small t", default 0.2);
       RODS-SAS : at each step in the window, evaluate eps at x + delta, delta = -rho s/||s||;
       RODS-CAS : the same with delta = rho grad||s|| / ||grad||s|| ||  (RODS, Tian et al. 2025,
                  eq. 8 and App. B.4; correction gated by the curvature index H if
                  bench.rods_thresh > 0, otherwise applied at every step in the window).
     Each method's one knob (eps, lam, rho) is chosen from its grid on bench.tune_n of the
     seeds, then the chosen value is run on all of them.

Only ground-truth hallucinations are touched, so this is a repair benchmark for every method.
Writes output/<run_tag>/<process>/hall_bench/{summary.csv,cells.json} after every cell, and
skips cells already in cells.json, so re-running the same command resumes a crashed run.
bench.grad_chunk bounds the seeds per autograd pass (the normal backprops through every step).
Stage hall_bench_viz draws the table.

    python code/ddim/main.py 'stages=[hall_bench,hall_bench_viz]' sweep.T_train=100
"""
from __future__ import annotations
import os
import sys
import csv
import json
import time
import importlib
from types import SimpleNamespace
import numpy as np
import torch

from common import gmm, checkpoint, utils
from common.process import make_process
from common.stages.pullback_iq import (Field, Sampler, pulled_normal, make_energy, sample_iq,
                                       viz_root, use_ckpt_schedule, level)


def bench_dir(cfg):
    return os.path.join(cfg.paths.output, cfg.run_tag, cfg.process, "hall_bench")


def _fmt(s, cfg, d, K):
    T = int(cfg.sweep.T_train) if "sweep" in cfg and "T_train" in cfg.sweep else 0
    p = os.path.expanduser(str(s).format(
        root=cfg.paths.root, home=os.path.expanduser("~"), d=d, K=K,
        seed=int(cfg.bench.ckpt_seed), T=T, process=str(cfg.bench.ckpt_process or cfg.process)))
    v = cfg.bench.get("variant", None)                      # weighted | unweighted | ...
    if v and v != "weighted":
        p = p.replace("/weighted/", f"/{v}/")
    return p


# ------------------------------------------------------------------ Pranav's checkpoints
def _pranav(cfg):
    """Import Pranav's own core + process factory, so his checkpoints are loaded and sampled
    by his code (his ScoreNet, his noise schedule, his seeds)."""
    code = _fmt(cfg.bench.get("pranav_code", "~/pranav/code"), cfg, 0, 0)
    if not os.path.isdir(code):
        raise FileNotFoundError(f"bench.pranav_code={code} does not exist; see the setup notes")
    if code not in sys.path:
        sys.path.append(code)
    return importlib.import_module("core"), importlib.import_module("processes.factory")


def load_pranav(cfg, d, K, device):
    pcore, pfac = _pranav(cfg)
    path = _fmt(cfg.bench.ckpt_fmt, cfg, d, K)
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["arch"]
    model = pcore.ScoreNet(ck["d"], a["h"], a["nb"], a["td"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    pcfg = (ck.get("config") or {}).get("process") or {"beta_min": 1e-4, "beta_max": 0.02,
                                                       "true_order": "heun"}
    means = ck["means"].to(device)
    proc = pfac.make_process("ddim", means, ck["variance"], int(ck["T"]), device,
                             SimpleNamespace(process=SimpleNamespace(**pcfg)), ck.get("weights"))

    def _ddim_step(x, i, e, _p=proc):                      # his sampler's update, one step
        ab, abp = _p.abar[i], _p.abar[i - 1]
        x0 = (x - torch.sqrt(1 - ab) * e) / torch.sqrt(ab)
        return torch.sqrt(abp) * x0 + torch.sqrt(1 - abp) * e
    proc._ddim_step = _ddim_step
    return model, ck, proc, means, float(ck["R99"]), path


def _ckpt_path(cfg, d, K):
    p = _fmt(cfg.bench.ckpt_fmt, cfg, d, K) if cfg.bench.ckpt_fmt else ""
    if p and os.path.exists(p):
        return p
    return utils.ckpt_path(cfg.paths.checkpoints, str(cfg.bench.ckpt_process or cfg.process),
                           d, K, int(cfg.sweep.T_train))


# ------------------------------------------------------------------ ground truth
SEED_KEYS = ("Z", "z", "seeds", "seed", "x_T", "xT", "X_T", "noise", "z0", "Z0", "init")
LABEL_KEYS = ("labels", "label", "y", "fate", "fates", "gt", "gt_labels", "labels_true", "L")
END_KEYS = ("X0", "x0", "X", "endpoints", "samples", "Xf", "x_0")


def _find(obj, keys):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and torch.is_tensor(obj[k]):
                return k, obj[k]
        for v in obj.values():                               # one level of nesting
            if isinstance(v, dict):
                r = _find(v, keys)
                if r[0] is not None:
                    return r
    return None, None


def _bget(b, k, v):
    return b.get(k, v)


def ground_truth(cfg, d, K, S, S_true, means_t, R99, device, proc=None):
    """Seeds whose ground-truth fate is -1, plus where that ground truth came from.

    Pranav backend: his gt_cache holds labels for proc.seeds(n_eval, d, seed + 1), drawn by his
    own Process.seeds (what his train/evaluate do), so the seeds are regenerated the same way
    and checked against his model before the labels are trusted."""
    b = cfg.bench
    src = str(b.gt_source)
    cap = None if b.get("n", None) in (None, "null", 0) else int(b.n)
    lab = lambda Z: torch.cat([gmm.label_fate(S.G(c), means_t, R99) for c in Z.split(int(b.chunk))])
    if src == "cache":
        path = _fmt(b.gt_fmt, cfg, d, K)
        gt = None
        try:
            gt = torch.load(path, map_location="cpu", weights_only=False)
            L = gt["gt"].long().reshape(-1)
            n_eval, seed = int(gt["n_eval"]), int(gt["seed"])
            if proc is not None and hasattr(proc, "seeds"):
                Z = proc.seeds(n_eval, d, seed + 1)          # exactly his eval seeds
            else:
                Z = utils.seeds(n_eval, d, seed + 1, device)
            n = n_eval if cap is None else min(cap, n_eval)
            Z, L = Z[:n].contiguous(), L[:n].to(device)
            n_gt = n
            # Full labels (mode index or -1), not just hallucinated-or-not: with ~1% hallucinating,
            # a hallucination-only comparison agrees ~98% even for unrelated seeds.
            # Check seeds spread over the WHOLE set, not just the first ones: GPU random numbers are
            # generated in parallel blocks, so the first few thousand seeds can reproduce exactly on
            # another machine while the rest do not. Hallucinated seeds are checked on their own too.
            m = min(int(b.get("check_n", 5000)), n)
            gck = torch.Generator(device="cpu").manual_seed(12345)
            idx = torch.randperm(n, generator=gck)[:m].to(Z.device)
            agree = (lab(Z[idx]) == L[idx]).float().mean().item()
            hidx = torch.nonzero(L < 0).flatten()
            hidx = hidx[torch.randperm(hidx.numel(), generator=gck)[:min(1000, hidx.numel())].to(hidx.device)]
            h_agree = (lab(Z[hidx]) < 0).float().mean().item() if hidx.numel() else 1.0
            print(f"[hall_bench]   gt vs his sampler: {100*agree:.2f}% of labels agree on {m} random seeds; "
                  f"{100*h_agree:.1f}% of his hallucinated seeds hallucinate here")
            agree = min(agree, h_agree)
            if agree < float(b.get("min_agree", 0.9)):
                raise ValueError(f"only {100*agree:.1f}% of labels agree: seeds not reproduced")
            del gt
            how = f"gt_cache:{os.path.basename(path)} ({n_gt} seeds)"
            if cap is not None and n_gt < cap:                  # top up to bench.n seeds
                extra = cap - n_gt
                Zx = (proc.seeds(extra, d, seed + 10007) if proc is not None
                      else utils.seeds(extra, d, seed + 10007, device))
                Z, L = torch.cat([Z, Zx]), torch.cat([L, lab(Zx)])
                how += f" + {extra} model-labelled"
            return Z, L, how
        except Exception as e:
            print(f"[hall_bench]   gt cache unusable ({path}): {e}. Labelling with his sampler instead.")
            gt = None
            src = "learned"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    n = cap or 200000
    Z = proc.seeds(n, d, int(b.seed) + 17) if proc is not None else utils.seeds(n, d, int(b.seed) + 17, device)
    SS = S_true if src == "true" else S
    L = torch.cat([gmm.label_fate(SS.G(c), means_t, R99) for c in Z.split(int(b.chunk))])
    return Z, L, f"{src}-sampler on {n} seeds"


# ------------------------------------------------------------------ RODS
def score(S, x, i):
    return -S.F.eps(x, i) / torch.sqrt(1 - S.ab[i])


def grad_score_norm(S, x, i):
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad(score(S, x, i).norm(dim=1).sum(), x)
    return g


def sample_rods(S, z, rho, kind, thresh, window, chunk):
    """RODS-SAS / RODS-CAS on the DDIM sampler; correction only in the step window (fractions
    of the run, 0 = first step)."""
    T = S.T
    lo, hi = float(window[0]), float(window[1])
    out = []
    for x in z.split(chunk):
        for k, i in enumerate(range(T - 1, 0, -1)):
            frac = k / max(1, T - 2)
            if rho <= 0 or not (lo <= frac <= hi):
                with torch.no_grad():
                    x = S.step(x, i)
                continue
            gn = grad_score_norm(S, x, i)
            u = gn / gn.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if thresh > 0:                                   # curvature index, eq. (8)
                H = (grad_score_norm(S, x + rho * u, i) - gn).norm(dim=1)
                on = H >= thresh
            else:
                on = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            with torch.no_grad():
                if kind == "sas":
                    s = score(S, x, i)
                    delta = -rho * s / s.norm(dim=1, keepdim=True).clamp_min(1e-12)
                else:                                        # cas
                    delta = rho * u
                e_hat = S.F.eps(x + delta, i)
                e = torch.where(on[:, None], e_hat, S.F.eps(x, i))
                x = S.p._ddim_step(x, i, e)
        out.append(x)
    return torch.cat(out)


# ------------------------------------------------------------------ one cell
def bench_one(cfg, d, K, device):
    b = cfg.bench
    if str(b.get("backend", "legacy")) == "pranav":
        got = load_pranav(cfg, d, K, device)
        if got is None:
            print(f"[hall_bench] d={d} K={K}: no checkpoint at {_fmt(b.ckpt_fmt, cfg, d, K)}, skipped")
            return None
        model, ck, proc, means_t, R99, path = got
    else:                                                    # legacy: this repo's own checkpoints
        path = _ckpt_path(cfg, d, K)
        if not os.path.exists(path):
            print(f"[hall_bench] d={d} K={K}: no checkpoint at {path}, skipped")
            return None
        model, ck = checkpoint.load(path, device)
        for prm in model.parameters():
            prm.requires_grad_(False)
        means_t, R99 = ck["means"], float(ck["R99"])
        proc = make_process(str(b.ckpt_process or cfg.process), means_t, ck["variance"], ck["T"], device, cfg)
        use_ckpt_schedule(proc, ck)
    S = Sampler(proc, Field(proc, model, "learned"))
    S_true = Sampler(proc, Field(proc, model, "true"))
    t0 = time.time()
    C = int(b.chunk)                                         # plain sampling
    Cg = int(b.get("grad_chunk", 64))                        # anything with autograd

    Z, L, how = ground_truth(cfg, d, K, S, S_true, means_t, R99, device, proc)
    hall = L < 0
    zh = Z[hall]
    N, n_h = int(Z.shape[0]), int(hall.sum())
    print(f"[hall_bench] d={d:>3} K={K:>2} T={S.T}: {n_h}/{N} ground-truth hallucinations "
          f"({100*n_h/max(N,1):.2f}%)  [{how}]", flush=True)
    row = {"d": d, "K": K, "T": int(S.T), "N": N, "gt": how, "n_hall": n_h}
    if n_h == 0:
        return row
    if n_h / max(N, 1) > float(b.get("max_hall_rate", 0.5)):
        print(f"[hall_bench]   {100*n_h/N:.1f}% hallucinating is above bench.max_hall_rate="
              f"{float(b.get('max_hall_rate', 0.5)):g}: the model or its labelling is broken here, cell skipped")
        row["skipped"] = "hall rate too high"
        return None

    def G(z):
        return torch.cat([S.G(c) for c in z.split(C)])

    def still(z_end):                                        # still hallucinating
        return int((gmm.label_fate(z_end, means_t, R99) < 0).sum())

    X_orig = G(zh)
    row["n_hall_plain"] = still(X_orig)                      # sanity: GT vs the learned sampler
    tgt = torch.cdist(X_orig, means_t).argmin(1)
    mu_t = means_t[tgt]
    g = torch.Generator(device=device).manual_seed(int(b.seed) + 3)
    tune = torch.randperm(n_h, generator=g, device=device)[: min(int(b.tune_n), n_h)]

    # ours: fixed eps along -n   (bench.ours=false skips it)
    if bool(b.get("ours", True)):
        nrm = torch.cat([pulled_normal(S, c, S.T - 1, m)[0] for c, m in zip(zh.split(Cg), mu_t.split(Cg))])
        ours = lambda e, idx=slice(None): still(G(zh[idx] - e * nrm[idx]))
        scan_e = {float(e): ours(float(e), tune) for e in b.eps_grid}
        eps = min(scan_e, key=lambda e: (scan_e[e], e))
        row["n_ours"], row["eps"] = ours(eps), eps

    # IQ at one or several windows (bench.iq_windows, e.g. [0.5, 0.4, 0.3, 0.2]; IQ acts on
    # every step with t = (i+1)/T <= window). The noise draws are taken either way so the RNG
    # stream, and every later choice, is the same whichever methods run.
    i0 = int(np.clip(round(float(b.iq_t0) * S.T - 1), 0, S.T - 1))
    MC = torch.randn(int(b.n_mc) // 2, d, generator=g, device=device)
    if bool(b.get("iq", True)):
        gradE = make_energy(S, S.F, i0, torch.cat([MC, -MC]))
        windows = [float(w) for w in (b.get("iq_windows", None) or [b.iq_window])]

        for w in windows:
            def iq(lam, idx=slice(None), w=w):
                return still(torch.cat([sample_iq(S, gradE, c, lam, w)[0][-1]
                                        for c in zh[idx].split(Cg)]))
            scan_l = {float(l): iq(float(l), tune) for l in b.lam_grid}
            lam = min(scan_l, key=lambda l: (scan_l[l], l))
            row[f"n_iq@{w:g}"], row[f"lam@{w:g}"] = iq(lam), lam
            if len(windows) == 1:                            # keep the old column names too
                row["n_iq"], row["lam"] = row[f"n_iq@{w:g}"], lam

    # RODS-SAS and RODS-CAS (bench.rods=false skips them)
    for kind in (("sas", "cas") if bool(b.get("rods", True)) else ()):
        def rods(rho, idx=slice(None)):
            return still(sample_rods(S, zh[idx], rho, kind, float(b.rods_thresh), b.rods_window, Cg))
        scan_r = {float(r): rods(float(r), tune) for r in b.rho_grid}
        rho = min(scan_r, key=lambda r: (scan_r[r], r))
        row[f"n_{kind}"], row[f"rho_{kind}"] = rods(rho), rho

    row["secs"] = round(time.time() - t0, 1)
    parts = []
    if "n_ours" in row:
        parts.append(f"ours {row['n_ours']} (eps={row['eps']:g})")
    for k in sorted([k for k in row if k.startswith("n_iq@")], key=lambda k: -float(k[5:])):
        w = k[5:]
        parts.append(f"IQ(t<={w}) {row[k]} (lam={row['lam@' + w]:g})")
    if "n_sas" in row:
        parts.append(f"SAS {row['n_sas']} (rho={row['rho_sas']:g})  CAS {row['n_cas']} (rho={row['rho_cas']:g})")
    print(f"[hall_bench]   {n_h} hallucinated -> " + "  ".join(parts) + f"   [{row['secs']}s]", flush=True)
    return row


KEYS = ["d", "K", "T", "N", "n_hall", "n_hall_plain", "n_ours", "n_iq", "n_sas", "n_cas",
        "eps", "lam", "rho_sas", "rho_cas", "gt", "secs"]


def _save(cfg, rows):
    os.makedirs(bench_dir(cfg), exist_ok=True)
    json.dump(rows, open(os.path.join(bench_dir(cfg), "cells.json"), "w"), indent=2)
    path = os.path.join(bench_dir(cfg), "summary.csv")
    extra = sorted({k for r in rows for k in r if k not in KEYS and k != "gt"})
    keys = [k for k in KEYS if k != "gt"] + extra + ["gt"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k) for k in keys])
    return path


def run(cfg):
    device = utils.get_device(cfg.device)
    prev = os.path.join(bench_dir(cfg), "cells.json")
    rows = json.load(open(prev)) if os.path.exists(prev) else []   # resume a crashed run
    done = {(r["d"], r["K"]) for r in rows}
    for d in cfg.bench.d:
        for K in cfg.bench.K:
            if (int(d), int(K)) in done:
                print(f"[hall_bench] d={d} K={K}: already done, skipped")
                continue
            r = bench_one(cfg, int(d), int(K), device)
            if r and r.get("n_hall"):
                rows.append(r)
                _save(cfg, rows)                            # after every cell
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    path = _save(cfg, rows)
    print(f"[hall_bench] wrote {path}")
    return {"summary": path, "n_cells": len(rows)}


# ------------------------------------------------------------------ table (stage hall_bench_viz)
def viz(cfg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(os.path.join(bench_dir(cfg), "cells.json")))
    if not rows:
        print("[hall_bench_viz] nothing to draw")
        return {}
    rows.sort(key=lambda r: (r["d"], r["K"]))
    out = os.path.join(viz_root(cfg), cfg.run_tag, cfg.process, "hall_bench")
    os.makedirs(out, exist_ok=True)
    iqk = sorted({k for r in rows for k in r if k.startswith("n_iq@")}, key=lambda k: -float(k[5:]))
    cand = ([("n_ours", "Ours")] + [(k, f"IQ t<={k[5:]}") for k in iqk]
            + ([] if iqk else [("n_iq", "IQ")]) + [("n_sas", "RODS-SAS"), ("n_cas", "RODS-CAS")])
    meth = [m for m in cand if all(m[0] in r for r in rows)]
    N = rows[0]["N"]

    nr = len(rows) + 1
    fig, ax = plt.subplots(figsize=(6.5 + 1.45 * len(meth), 0.42 * nr + 1.5))
    ax.axis("off")
    colx = [0.04, 0.11] + list(np.linspace(0.30, 0.97, 1 + len(meth)))
    unit = 0.93 / (nr + 1.4)
    top = 0.975
    rowy = lambda i: top - unit * (1.6 + i)
    Lh = lambda y, lw, col="black": ax.plot([0.01, 0.99], [y, y], color=col, lw=lw,
                                            transform=ax.transAxes, clip_on=False)
    Tx = lambda x, y, t, **k: ax.text(x, y, t, transform=ax.transAxes, va="center", **k)
    Lh(top + unit * 0.2, 1.3)
    for j, h in enumerate(["$d$", "$K$", "original"] + [m[1] for m in meth]):
        Tx(colx[j], top - unit * 0.55, h, ha="right" if j > 1 else "center", fontsize=11,
           fontweight="bold")
    Lh(top - unit * 1.05, 0.8)
    prev = None
    tot = {k: 0 for k in ["n_hall"] + [m[0] for m in meth]}
    for i, r in enumerate(rows):
        y = rowy(i)
        if prev is not None and r["d"] != prev:
            Lh(y + unit * 0.5, 0.6, "0.75")
        prev = r["d"]
        Tx(colx[0], y, str(r["d"]), ha="center", fontsize=10.5)
        Tx(colx[1], y, str(r["K"]), ha="center", fontsize=10.5)
        Tx(colx[2], y, f"{r['n_hall']:,}", ha="right", fontsize=10.5, color="0.35")
        best = min(r[m[0]] for m in meth)
        for j, (k, _) in enumerate(meth):
            Tx(colx[3 + j], y, f"{r[k]:,}", ha="right", fontsize=10.5,
               fontweight="bold" if r[k] == best else "normal")
        for k in tot:
            tot[k] += r[k]
    y = rowy(len(rows))
    Lh(y + unit * 0.5, 0.8)
    Tx(colx[1], y, "total", ha="center", fontsize=10.5)
    Tx(colx[2], y, f"{tot['n_hall']:,}", ha="right", fontsize=10.5, color="0.35")
    bt = min(tot[m[0]] for m in meth)
    for j, (k, _) in enumerate(meth):
        Tx(colx[3 + j], y, f"{tot[k]:,}", ha="right", fontsize=10.5,
           fontweight="bold" if tot[k] == bt else "normal")
    Lh(y - unit * 0.55, 1.3)
    Tx(0.01, y - unit * 1.4, f"Hallucinated samples out of N = {N:,} seeds per cell: the ground-truth "
       f"count, and how many remain after each repair (fewest per row in bold).", ha="left",
       fontsize=8.8, style="italic", color="0.3")
    Tx(0.01, y - unit * 2.05, "IQ t<=w: IQ on every step with t = (i+1)/T <= w; each method's step size "
       "/ strength chosen on a subset of the hallucinated seeds.", ha="left",
       fontsize=8.8, style="italic", color="0.3")
    fig.savefig(os.path.join(out, "table_hall_bench.png"), dpi=220, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    o = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Hallucinated samples out of $N={N:,}$ seeds per cell: the ground-truth count "
         r"and the number remaining after each repair, applied to the ground-truth hallucinations "
         r"only. IQ $t\le w$: IQ on every sampling step with $t=(i+1)/T \le w$. Fewest per row in bold.}",
         r"\label{tab:hall-bench}", r"\begin{tabular}{rr r " + "r" * len(meth) + "}", r"\toprule",
         r"$d$ & $K$ & original & " + " & ".join(m[1] for m in meth) + r" \\", r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["d"] != prev:
            o.append(r"\midrule")
        prev = r["d"]
        best = min(r[m[0]] for m in meth)
        cells = [(r"\textbf{%d}" % r[k]) if r[k] == best else str(r[k]) for k, _ in meth]
        o.append(f"{r['d']} & {r['K']} & {r['n_hall']} & " + " & ".join(cells) + r" \\")
    o.append(r"\midrule")
    cells = [(r"\textbf{%d}" % tot[k]) if tot[k] == bt else str(tot[k]) for k, _ in meth]
    o.append(r"\multicolumn{2}{r}{total} & " + f"{tot['n_hall']} & " + " & ".join(cells) + r" \\")
    o += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "table_hall_bench.tex"), "w").write("\n".join(o) + "\n")
    print(f"[hall_bench_viz] wrote {out}/table_hall_bench.png/.tex")
    return {"figures": [os.path.join(out, "table_hall_bench")]}


# ==================================================================================== #
#  stage: hall_bench_anim — replay benchmark seeds with every step recorded (d = 2)     #
# ==================================================================================== #
"""Reads the finished benchmark (output/<run_tag>/<process>/hall_bench/cells.json), rebuilds
each cell's hallucinated seeds (cells with d in bench.anim_d, default [2]) exactly as the benchmark did (same ground truth, same RNG
stream), takes the first bench.anim_n of them (no cherry-picking), and re-runs every method
with the settings the benchmark chose (eps, lam per IQ window, rho), recording full paths.
d = 2 is drawn directly; d > 2 in the (a, r) plane of each seed's two modes, where distances to
both centres are exact. Writes one animation (gif, + mp4 with ffmpeg) and one still per seed to
visualization/<run_tag>/<process>/hall_bench/.

    python code/ddim/main.py 'stages=[hall_bench_anim]' run_tag=<benchmark run_tag> +bench.variant=...
"""
from common.stages.pullback_iq import C_OG, C_IQ, C_US


def sample_rods_traj(S, z, rho, kind, thresh, window):
    """sample_rods, recording every state; returns (T, B, d)."""
    T = S.T
    lo, hi = float(window[0]), float(window[1])
    x, xs = z, [z]
    for k, i in enumerate(range(T - 1, 0, -1)):
        frac = k / max(1, T - 2)
        if rho <= 0 or not (lo <= frac <= hi):
            with torch.no_grad():
                x = S.step(x, i)
        else:
            gn = grad_score_norm(S, x, i)
            u = gn / gn.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if thresh > 0:
                on = (grad_score_norm(S, x + rho * u, i) - gn).norm(dim=1) >= thresh
            else:
                on = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            with torch.no_grad():
                if kind == "sas":
                    s_ = score(S, x, i)
                    delta = -rho * s_ / s_.norm(dim=1, keepdim=True).clamp_min(1e-12)
                else:
                    delta = rho * u
                e = torch.where(on[:, None], S.F.eps(x + delta, i), S.F.eps(x, i))
                x = S.p._ddim_step(x, i, e)
        xs.append(x.detach())
    return torch.stack(xs)


def _anim_multi(paths, t, MU, R99, zh, zo, nn, eps, title, path, fps=25, hold=20, intro=20,
                max_frames=120, dpi=110):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    from matplotlib.colors import ListedColormap
    import shutil
    K = MU.shape[0]
    pale = ["#fde0dd", "#deebf7", "#e5f5e0", "#fff7bc", "#efedf5", "#fee6ce", "#e0f3f8",
            "#f2f0f7", "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc",
            "#e5d8bd", "#fddaec"]
    cols = {"original": C_OG, "ours": C_US, "RODS-SAS": "#9467bd", "RODS-CAS": "#2ca02c"}
    iq_cols = ["#1f77b4", "#4a98c9", "#7ab6d9", "#a6cde3"]
    names = list(paths)
    for j, nm in enumerate([n for n in names if n.startswith("IQ")]):
        cols[nm] = iq_cols[j % len(iq_cols)]
    n = len(t)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    pts = np.concatenate([p for p in paths.values()] + [MU])
    lo_, hi_ = pts.min(0), pts.max(0)
    c, w = 0.5 * (lo_ + hi_), 0.62 * max(hi_ - lo_) + 2 * R99
    gx, gy = np.linspace(c[0] - w, c[0] + w, 300), np.linspace(c[1] - w, c[1] + w, 300)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    near = np.argmin(((P[:, None, :] - MU[None]) ** 2).sum(-1), 1).reshape(GX.shape)
    ax[0].pcolormesh(GX, GY, near, cmap=ListedColormap(pale[:K]), shading="auto",
                     vmin=0, vmax=K - 1, alpha=0.65)
    ax[0].contour(GX, GY, near, levels=np.arange(K) + 0.5, colors="0.55", linestyles="--",
                  linewidths=0.9)
    th = np.linspace(0, 2 * np.pi, 240)
    for k in range(K):
        ax[0].plot(MU[k, 0] + R99 * np.cos(th), MU[k, 1] + R99 * np.sin(th), color="0.35", lw=1.2)
        ax[0].plot(*MU[k], "k*", ms=12)
    ax[0].plot(*paths["original"][0], "X", color="k", ms=11, zorder=5)
    ax[0].set_xlim(c[0] - w, c[0] + w); ax[0].set_ylim(c[1] - w, c[1] + w)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$")
    wi = max(2.2 * eps, 1e-6)                                # inset: the pullback step
    ins = ax[0].inset_axes([1.0 - 0.30, 0.0, 0.30, 0.30])   # bottom-right corner, small
    ins.set_xlim(zh[0] - wi, zh[0] + wi); ins.set_ylim(zh[1] - wi, zh[1] + wi)
    ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")
    ins.annotate("", xy=zh - 0.8 * wi * nn, xytext=zh, arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
    ins.text(0.05, 0.05, rf"$-n$,  $\epsilon$={eps:.3g}", transform=ins.transAxes, fontsize=8)
    ins.plot(*zh, "X", color="k", ms=9); ins.plot(*zo, "o", color=C_US, ms=7, mec="k")
    ins.set_title("pullback step at $x_T$", fontsize=8)
    lines = {k: ax[0].plot([], [], color=cols[k], lw=2 if k in ("original", "ours") else 1.5,
                           label=k, zorder=3)[0] for k in names}
    dots = {k: ax[0].plot([], [], "o", color=cols[k], ms=8, mec="k", zorder=4)[0] for k in names}
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
                 framealpha=0.92, facecolor="white")
    D = {k: np.linalg.norm(paths[k] - paths["original"], axis=1) for k in names if k != "original"}
    for k in D:
        ax[1].plot(t, D[k], color=cols[k], lw=1, alpha=0.18)
    dl = {k: ax[1].plot([], [], color=cols[k], lw=2, label=k)[0] for k in D}
    ax[1].axhline(R99, color="0.5", lw=1, ls=":", label=r"$R_{99}$")
    now = ax[1].axvline(t[0], color="0.3", lw=1)
    allD = np.concatenate([v[1:] for v in D.values()])
    ax[1].set_yscale("log"); ax[1].set_ylim(max(1e-4, allD[allD > 0].min()) * 0.5, allD.max() * 2)
    ax[1].set_xlim(t[0], t[-1]); ax[1].set_xlabel("t  (sampling runs 1 → 0)")
    ax[1].set_ylabel(r"$\|x_t-x_t^{\mathrm{orig}}\|$")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                 framealpha=0.92, facecolor="white")
    sup = fig.suptitle("")
    idx = (np.unique(np.linspace(0, n - 1, max_frames).astype(int)) if n > max_frames else np.arange(n))

    def frame(f):
        if f < intro:
            i = 0
            sup.set_text(f"{title}:  pullback step, " + rf"$\epsilon$ = {eps:.3g}")
        else:
            i = int(idx[min(f - intro, len(idx) - 1)])
            sup.set_text(f"{title}:  reverse step {i}/{n-1}   (t = {t[i]:.3f})")
        for k in names:
            lines[k].set_data(paths[k][: i + 1, 0], paths[k][: i + 1, 1])
            dots[k].set_data([paths[k][i, 0]], [paths[k][i, 1]])
        for k in D:
            dl[k].set_data(t[: i + 1], D[k][: i + 1])
        now.set_xdata([t[i], t[i]])
        return list(lines.values()) + list(dots.values()) + list(dl.values()) + [now, sup]

    fig.tight_layout()
    frame(intro + len(idx) - 1)                              # still of the final state
    fig.savefig(f"{path}.png", dpi=170)
    an = FuncAnimation(fig, frame, frames=intro + len(idx) + hold, interval=1000 / fps, blit=False)
    if shutil.which("ffmpeg"):
        an.save(f"{path}.mp4", writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    an.save(f"{path}.gif", writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def _anim_meridian(paths_x0, paths_x, t, mu1, mu2, R99, eps, title, path, fps=25, hold=20,
                   max_frames=120, dpi=110):
    """d > 2: plot the predicted clean sample x0_hat(x_t) in (a, r) coordinates, where a is the
    position along the axis mu1 -> mu2 and r the distance from that axis. Distances to BOTH mode
    centres are exact in these coordinates, so both cores are exact half-disks of radius R99
    and the boundary between the two modes is exact."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import shutil
    e = (mu2 - mu1) / np.linalg.norm(mu2 - mu1)
    L = float(np.linalg.norm(mu2 - mu1))

    def ar(X):                                              # (T, d) -> (T, 2)
        Y = X - mu1
        a = Y @ e
        r = np.linalg.norm(Y - a[:, None] * e[None], axis=1)
        return np.stack([a, r], 1)

    P = {k: ar(v) for k, v in paths_x0.items()}
    cols = {"original": C_OG, "ours": C_US, "RODS-SAS": "#9467bd", "RODS-CAS": "#2ca02c"}
    iq_cols = ["#1f77b4", "#4a98c9", "#7ab6d9", "#a6cde3"]
    names = list(P)
    for j, nm in enumerate([n for n in names if n.startswith("IQ")]):
        cols[nm] = iq_cols[j % len(iq_cols)]
    n = len(t)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    xlo, xhi = -2.5 * R99, L + 2.5 * R99
    yhi = max(2.5 * R99, 0.6 * L)
    ax[0].axvspan(xlo, L / 2, color="#deebf7", alpha=0.65, lw=0)
    ax[0].axvspan(L / 2, xhi, color="#fde0dd", alpha=0.65, lw=0)
    ax[0].axvline(L / 2, color="0.55", ls="--", lw=0.9)
    th = np.linspace(0, np.pi, 200)
    for cx, lab in ((0.0, r"$\mu_{target}$"), (L, r"$\mu_{other}$")):
        ax[0].plot(cx + R99 * np.cos(th), R99 * np.sin(th), color="0.35", lw=1.2)
        ax[0].plot(cx, 0, "k*", ms=12, clip_on=False)
        ax[0].annotate(lab, (cx, 0), textcoords="offset points", xytext=(6, 6), fontsize=10)
    ax[0].set_xlim(xlo, xhi); ax[0].set_ylim(0, yhi); ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$a$: position along the axis $\mu_{target} \to \mu_{other}$")
    ax[0].set_ylabel(r"$r$: distance from that axis")
    ax[0].set_title(r"predicted clean sample $\hat x_0(x_t)$; distances to both modes exact", fontsize=10)
    lines = {k: ax[0].plot([], [], color=cols[k], lw=2 if k in ("original", "ours") else 1.5,
                           label=k, zorder=3)[0] for k in names}
    dots = {k: ax[0].plot([], [], "o", color=cols[k], ms=8, mec="k", zorder=4)[0] for k in names}
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                 framealpha=0.92, facecolor="white")
    D = {k: np.linalg.norm(paths_x[k] - paths_x["original"], axis=1) for k in names if k != "original"}
    for k in D:
        ax[1].plot(t, D[k], color=cols[k], lw=1, alpha=0.18)
    dl = {k: ax[1].plot([], [], color=cols[k], lw=2, label=k)[0] for k in D}
    ax[1].axhline(R99, color="0.5", lw=1, ls=":", label=r"$R_{99}$")
    now = ax[1].axvline(t[0], color="0.3", lw=1)
    allD = np.concatenate([v[1:] for v in D.values()])
    ax[1].set_yscale("log"); ax[1].set_ylim(max(1e-4, allD[allD > 0].min()) * 0.5, allD.max() * 2)
    ax[1].set_xlim(t[0], t[-1]); ax[1].set_xlabel("t  (sampling runs 1 → 0)")
    ax[1].set_ylabel(r"$\|x_t-x_t^{\mathrm{orig}}\|$ (full $d$-dim distance)")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
                 framealpha=0.92, facecolor="white")
    sup = fig.suptitle("")
    idx = (np.unique(np.linspace(0, n - 1, max_frames).astype(int)) if n > max_frames else np.arange(n))

    def frame(f):
        i = int(idx[min(f, len(idx) - 1)])
        sup.set_text(f"{title}:  reverse step {i}/{n-1}   (t = {t[i]:.3f})   " + rf"$\epsilon$ = {eps:.3g}")
        for k in names:
            lines[k].set_data(P[k][: i + 1, 0], P[k][: i + 1, 1])
            dots[k].set_data([P[k][i, 0]], [P[k][i, 1]])
        for k in D:
            dl[k].set_data(t[: i + 1], D[k][: i + 1])
        now.set_xdata([t[i], t[i]])
        return list(lines.values()) + list(dots.values()) + list(dl.values()) + [now, sup]

    fig.tight_layout()
    frame(len(idx) - 1)
    fig.savefig(f"{path}.png", dpi=170)
    an = FuncAnimation(fig, frame, frames=len(idx) + hold, interval=1000 / fps, blit=False)
    if shutil.which("ffmpeg"):
        an.save(f"{path}.mp4", writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    an.save(f"{path}.gif", writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def anim(cfg):
    b = cfg.bench
    device = utils.get_device(cfg.device)
    cells = json.load(open(os.path.join(bench_dir(cfg), "cells.json")))
    out = os.path.join(viz_root(cfg), cfg.run_tag, cfg.process, "hall_bench")
    os.makedirs(out, exist_ok=True)
    made = []
    want = [int(x) for x in (b.get("anim_d", None) or [2])]
    for r in [r for r in cells if int(r["d"]) in want]:
        d, K = int(r["d"]), int(r["K"])
        path = _ckpt_path(cfg, d, K)
        model, ck = checkpoint.load(path, device)
        for prm in model.parameters():
            prm.requires_grad_(False)
        means_t, R99 = ck["means"], float(ck["R99"])
        proc = make_process(str(b.ckpt_process or cfg.process), means_t, ck["variance"], ck["T"], device, cfg)
        use_ckpt_schedule(proc, ck)
        S = Sampler(proc, Field(proc, model, "learned"))
        S_true = Sampler(proc, Field(proc, model, "true"))
        Z, L, how = ground_truth(cfg, d, K, S, S_true, means_t, R99, device, proc)
        zh_all = Z[L < 0]
        if zh_all.shape[0] != int(r["n_hall"]):
            print(f"[hall_bench_anim] d={d} K={K}: {zh_all.shape[0]} hallucinated seeds here vs "
                  f"{r['n_hall']} in the benchmark; check the run settings (variant, n)")
        g = torch.Generator(device=device).manual_seed(int(b.seed) + 3)   # same RNG stream
        torch.randperm(zh_all.shape[0], generator=g, device=device)
        MC = torch.randn(int(b.n_mc) // 2, d, generator=g, device=device)
        zh = zh_all[: int(b.get("anim_n", 3))]
        with torch.no_grad():
            tgt = torch.cdist(S.G(zh), means_t).argmin(1)
        mu_t = means_t[tgt]
        paths = {"original": S.traj(zh)}
        eps = float(r.get("eps", 0.0) or 0.0)
        nrm = pulled_normal(S, zh, S.T - 1, mu_t)[0]
        z_ours = zh - eps * nrm
        if "n_ours" in r:
            paths["ours"] = S.traj(z_ours)
        i0 = int(np.clip(round(float(b.iq_t0) * S.T - 1), 0, S.T - 1))
        gradE = make_energy(S, S.F, i0, torch.cat([MC, -MC]))
        for k in sorted([k for k in r if k.startswith("n_iq@")], key=lambda k: -float(k[5:])):
            w = k[5:]
            paths[f"IQ t<={w}"] = sample_iq(S, gradE, zh, float(r[f"lam@{w}"]), float(w))[0]
        for kind in ("sas", "cas"):
            if f"n_{kind}" in r:
                paths[f"RODS-{kind.upper()}"] = sample_rods_traj(
                    S, zh, float(r[f"rho_{kind}"]), kind, float(b.rods_thresh), b.rods_window)
        t = np.array([S.t_of(level(k, S.T)) for k in range(S.T)])
        cpu = lambda x: x.detach().cpu().numpy()
        MU = cpu(means_t)
        frames = int(cfg.pullback.get("anim_frames", 120))
        if d > 2:                                           # predicted clean samples along each path
            with torch.no_grad():
                paths_x0 = {k: torch.stack([S.tweedie(v[kk], level(kk, S.T)) for kk in range(S.T)])
                            for k, v in paths.items()}
            with torch.no_grad():                           # the other end of each seed's channel
                two = torch.cdist(paths["original"][-1], means_t).topk(2, dim=1, largest=False).indices
        for j in range(zh.shape[0]):
            stem = os.path.join(out, f"anim_d{d}_K{K}_seed{j}")
            if d == 2:
                P = {k: cpu(v[:, j]) for k, v in paths.items()}
                _anim_multi(P, t, MU, R99, cpu(zh[j]), cpu(z_ours[j]), cpu(nrm[j]), eps,
                            f"d={d}, K={K}, benchmark seed {j}", stem, max_frames=frames)
            else:
                t_i = int(tgt[j]); o_i = int(two[j, 1] if int(two[j, 0]) == t_i else two[j, 0])
                _anim_meridian({k: cpu(v[:, j]) for k, v in paths_x0.items()},
                               {k: cpu(v[:, j]) for k, v in paths.items()}, t,
                               MU[t_i], MU[o_i], R99, eps, f"d={d}, K={K}, benchmark seed {j}",
                               stem, max_frames=frames)
            made.append(stem)
            print(f"[hall_bench_anim] wrote {stem}.gif", flush=True)
    return {"figures": made}