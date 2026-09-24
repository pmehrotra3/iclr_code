"""baselines/basin.py — stage `basin`: does each seed reach the basin it should?

Why this stage exists
---------------------
`evaluate` asks how PREDICTABLE a sampler's fates are from the seed: it
picks one ground-truth sampler and scores classifiers against it. That is the
right question for the paper's main claim, but it is not the question Abhinav
asked about the baselines, which is whether an intervention sends each seed to
the basin it *should* reach.

So this stage fixes one seed set and one reference, and compares samplers on it:

    reference   the analytic-field sampler (exact GMM score).  The basin a seed
                belongs to is defined by where the exact reverse map takes it --
                this is the paper's notion, not a distance threshold.
    candidates  the learnt sampler (`ddim`) and each intervention
                (`rods_cas`, `rods_sas`, `iq`).

Every metric is agreement between two fate vectors on the SAME seeds, so no
k*sigma / chi^2 / 3-sigma threshold appears anywhere. That also removes the
reason our numbers, Pranav's R99 numbers and the LID paper's 3-sigma numbers
have not been comparable.

Reported per (d, K), per sampler:

    basin_acc     fraction of ALL seeds sent to the basin the reference gives
                  (the headline: "does it go where it should")
    mode_acc      same, restricted to seeds the reference sends to a real mode
    misassigned   reference says mode i, sampler says mode j != i.  A wrong
                  class, NOT a hallucination -- the k*sigma rule cannot see this
                  at all, which is the main thing it was missing.
    hallucinated  reference says mode i, sampler says -1 (fell in the intermodal
                  intersection)
    spurious      reference says -1, sampler says a mode
    hall_rate     share of -1 under this sampler, for continuity with the old tables
    d_hall        hall_rate minus the reference's, signed

Against the learnt baseline, so an intervention is judged on what it changed:

    preserved     of seeds the BASELINE placed in a real basin, the fraction this
                  sampler still places in the SAME basin
    broke         baseline placed it, this sampler hallucinates it
    rescued       baseline hallucinated it, this sampler places it
    moved         baseline placed it in basin i, this sampler in basin j != i

Every sampler runs the SAME network: the ddim checkpoint of each (d, K, seed) cell
(seeds: basin.seeds, default the run's repeat seeds), each baseline with the knobs of
its conf/process/<name>.yaml. Seeds are the cell's eval seeds, proc.seeds(n, d, seed+1).

Writes output/<run_id>/ddim/<variant>/basin/basin[_<tag>].{json,csv} and a per-cell
console table. It loops over `basin.samplers` itself, so a single invocation covers
every method:

    python code/baselines/main.py 'stages=[basin]'
    python code/baselines/main.py 'stages=[basin]' 'basin.d=[2,8]' 'basin.K=[4]' basin.n_eval=20000
"""
from __future__ import annotations

import os
import json

import torch
from omegaconf import OmegaConf

import core
from evaluate import load_ckpt
from processes.factory import make_process
from train import ckpt_path, n_eval_for, variant_of
from baselines.process import process_cfg

DEFAULT_SAMPLERS = ("ddim", "rods_cas", "rods_sas", "iq")


@torch.no_grad()
def _label(proc, model, X, R99, chunk=200000):
    out = []
    for j in range(0, X.shape[0], chunk):
        out.append(proc.label(model, X[j:j + chunk], R99))
    return torch.cat(out)


def _agree(ref, got):
    """Agreement of `got` against the reference fates `ref`. Fractions of ALL seeds."""
    n = ref.numel()
    ref_mode = ref >= 0
    return dict(
        basin_acc=float((got == ref).float().mean()),
        mode_acc=(float((got[ref_mode] == ref[ref_mode]).float().mean())
                  if bool(ref_mode.any()) else float("nan")),
        misassigned=float(((got >= 0) & ref_mode & (got != ref)).sum()) / n,
        hallucinated=float(((got == -1) & ref_mode).sum()) / n,
        spurious=float(((got >= 0) & (ref == -1)).sum()) / n,
        hall_rate=float((got == -1).float().mean()),
    )


def _vs_baseline(base, got):
    """What this sampler changed relative to the learnt baseline."""
    ok = base >= 0
    bad = ~ok
    f = lambda m: float(m.sum()) / max(int(ok.sum()), 1)          # noqa: E731
    return dict(
        preserved=(float((got[ok] == base[ok]).float().mean())
                   if bool(ok.any()) else float("nan")),
        broke=f((got == -1) & ok),
        moved=f((got >= 0) & ok & (got != base)),
        rescued=(float((got[bad] >= 0).float().mean())
                 if bool(bad.any()) else float("nan")),
    )


def _one(cfg, d, K, seed, device, samplers):
    """All samplers on one (d, K, seed) cell, sharing seeds, a reference and the network."""
    path = ckpt_path(cfg.paths.checkpoints, "ddim", d, K, seed, variant_of(cfg))
    if not os.path.exists(path):
        return None
    model, ck = load_ckpt(path, device)
    means_t, R99, variance, T = ck["means"], float(ck["R99"]), ck["variance"], int(ck["T"])
    weights = ck.get("weights")

    n = int(cfg.basin.get("n_eval", None) or n_eval_for(cfg, K))
    ddim_cfg = process_cfg(cfg, ck, "ddim")
    X = make_process("ddim", means_t, variance, T, device, ddim_cfg, weights).seeds(n, d, seed + 1)

    # reference: the analytic field at T_true. No model, no threshold on top.
    ref_proc = make_process("ddim", means_t, variance, int(ddim_cfg.process.T_true), device,
                            ddim_cfg, weights)
    ref = _label(ref_proc, None, X, R99)

    row = {"d": d, "K": K, "seed": seed, "n_eval": n,
           "ref_hall_rate": float((ref == -1).float().mean()),
           "samplers": {}}

    base_fate = None
    for nm in samplers:
        proc = make_process(nm, means_t, variance, T, device, process_cfg(cfg, ck, nm), weights)
        got = _label(proc, model, X, R99)
        if nm == "ddim":
            base_fate = got
        r = _agree(ref, got)
        r["d_hall"] = r["hall_rate"] - row["ref_hall_rate"]
        if base_fate is not None and nm != "ddim":
            r.update(_vs_baseline(base_fate, got))
        row["samplers"][nm] = r
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def _print(row):
    d, K = row["d"], row["K"]
    print(f"\n=== d={d}  K={K}  seed={row['seed']}  n={row['n_eval']}  "
          f"reference (analytic) hall={100*row['ref_hall_rate']:.2f}% ===")
    print(f"  {'sampler':<10} {'basin acc':>10} {'mode acc':>9} {'misasgn':>8} "
          f"{'halluc':>7} {'spur':>6} {'HR':>7} {'preserved':>10} {'broke':>7} "
          f"{'moved':>7} {'rescued':>8}")
    for nm, r in row["samplers"].items():
        g = lambda k: (f"{100*r[k]:.2f}" if k in r else "-")      # noqa: E731
        print(f"  {nm:<10} {100*r['basin_acc']:>9.2f}% {100*r['mode_acc']:>8.2f}% "
              f"{100*r['misassigned']:>7.2f}% {100*r['hallucinated']:>6.2f}% "
              f"{100*r['spurious']:>5.2f}% {100*r['hall_rate']:>6.2f}% "
              f"{g('preserved'):>10} {g('broke'):>7} {g('moved'):>7} "
              f"{g('rescued'):>8}")


def run(cfg):
    device = core.get_device(cfg.device)
    b = cfg.basin
    samplers = tuple(b.get("samplers", None) or DEFAULT_SAMPLERS)
    seeds = [int(x) for x in (b.get("seeds", None) or core.seed_list(cfg))]
    out_dir = os.path.join(cfg.paths.output, str(cfg.run_id), "ddim", variant_of(cfg), "basin")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for d in b.d:
        for K in b.K:
            for seed in seeds:
                r = _one(cfg, int(d), int(K), seed, device, samplers)
                if r is None:
                    print(f"[basin] d={d:>2} K={K:>2} seed={seed} -> no ddim checkpoint, skipped")
                    continue
                _print(r)
                rows.append(r)

    tag = b.get("tag", None)
    js = os.path.join(out_dir, f"basin{'_' + str(tag) if tag else ''}.json")
    with open(js, "w") as f:
        json.dump({"reference": "analytic field (exact GMM score)",
                   "samplers": list(samplers), "seeds": seeds,
                   "basin": OmegaConf.to_container(b, resolve=True),
                   "results": rows}, f, indent=2)

    cols = ["basin_acc", "mode_acc", "misassigned", "hallucinated", "spurious",
            "hall_rate", "d_hall", "preserved", "broke", "moved", "rescued"]
    csv = js[:-5] + ".csv"
    with open(csv, "w") as f:
        f.write("d,K,seed,n_eval,ref_hall_rate,sampler," + ",".join(cols) + "\n")
        for r in rows:
            pre = f"{r['d']},{r['K']},{r['seed']},{r['n_eval']},{r['ref_hall_rate']:.5f}"
            for nm, s_ in r["samplers"].items():
                f.write(f"{pre},{nm}," + ",".join(
                    (f"{s_[c]:.5f}" if c in s_ else "") for c in cols) + "\n")
    print(f"\n[basin] wrote {js} and {csv}")
    return {"json": js, "csv": csv, "n_cells": len(rows)}
